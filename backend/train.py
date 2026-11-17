"""Fit every model, score them on houses and whole districts they never saw, write artifacts/.  python -m backend.train"""
import json
import time
from datetime import datetime, timezone

import numpy as np

from . import config, data
from .bma import family_weights, subset_bma
from .combine import BASE, CROWD, KIND, LABEL, Ensemble
from .ensemble import AdaBoost, Forest, GradientBoosting, Stacker
from .gam import GAM
from .linear import Logistic
from .metrics import auc, bootstrap_auc_gain, calibration, capacity_curve, confusion, log_loss, roc_points, summary
from .tree import Tree

SHIFTS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
SHARES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


UNIT = {"pred_intensity": "", "count_floors_pre_eq": "", "age_building": " yrs", "plinth_area_sq_ft": " sq ft", "height_ft_pre_eq": " ft"}


def _num(src, v):
    return f"{v:.2f}" if src == "pred_intensity" else f"{v:.0f}"


def rule_text(D, j, thr, side):
    """Plain words for one branch of a split."""
    name, kind, src = D.names[j], D.kind[j], D.source[j]
    if kind in ("bin", "cat"):
        return f"not {name}" if side == "left" else name
    short = config.SHORT.get(src, name)
    if src == "count_floors_pre_eq":
        return f"{short} ≤ {int(thr)}" if side == "left" else f"{short} ≥ {int(thr) + 1}"
    return f"{short} ≤ {_num(src, thr)}{UNIT[src]}" if side == "left" else f"{short} > {_num(src, thr)}{UNIT[src]}"


def merged_rule(D, conds):
    """Join the splits on the way to a leaf, merging repeated cuts on one number into a single range."""
    order, lo, hi, flag = [], {}, {}, {}
    for j, thr, side in conds:
        if j not in order:
            order.append(j)
        if D.kind[j] == "num":
            if side == "left":
                hi[j] = min(hi.get(j, np.inf), thr)
            else:
                lo[j] = max(lo.get(j, -np.inf), thr)
        else:
            flag[j] = side
    out = []
    for j in order:
        if D.kind[j] != "num":
            out.append(rule_text(D, j, 0, flag[j]))
            continue
        src = D.source[j]
        short = config.SHORT.get(src, D.names[j])
        lo_v, hi_v = lo.get(j), hi.get(j)
        if src == "count_floors_pre_eq":
            a_ = int(lo_v) + 1 if lo_v is not None else None
            c_ = int(hi_v) if hi_v is not None else None
            if a_ is not None and c_ is not None:
                out.append(f"{short} {a_}" if a_ == c_ else f"{short} {a_}–{c_}")
            else:
                out.append(f"{short} ≥ {a_}" if a_ is not None else f"{short} ≤ {c_}")
        elif lo_v is not None and hi_v is not None:
            out.append(f"{short} {_num(src, lo_v)}–{_num(src, hi_v)}{UNIT[src]}")
        else:
            out.append(f"{short} > {_num(src, lo_v)}{UNIT[src]}" if lo_v is not None else f"{short} ≤ {_num(src, hi_v)}{UNIT[src]}")
    return out


def within_ward(y, p, wid, share=0.2, min_n=30):
    """Ranking inside each ward, where every house had the same estimated shaking: weighted AUC, and the share of
    collapses caught if each ward strengthens its own riskiest `share` of houses."""
    order = np.argsort(wid, kind="mergesort")
    ys, ps, ws = y[order], p[order], wid[order]
    cuts = np.flatnonzero(np.diff(ws)) + 1
    aucs, ns, hit, tot = [], [], 0, 0
    for a, c in zip(np.r_[0, cuts], np.r_[cuts, len(ws)]):
        yy, pp = ys[a:c], ps[a:c]
        if c - a < min_n or yy.min() == yy.max():
            continue
        aucs.append(auc(yy, pp))
        ns.append(c - a)
        k = max(1, int(round(share * (c - a))))
        hit += int(yy[np.argsort(-pp, kind="mergesort")[:k]].sum())
        tot += int(yy.sum())
    return {"auc": float(np.average(aucs, weights=ns)), "recall": hit / max(tot, 1), "wards": len(ns)}


def tree_nodes(t, D):
    """A small tree as nodes the page can draw: question, share of houses, collapse rate."""
    keep = t._reachable()
    total = t.weight[0]
    out = []
    for i in keep:
        node = {"id": int(i), "n": float(t.weight[i]), "share": float(t.weight[i] / total), "p": float(t.value[i][1]), "leaf": bool(t.feature[i] < 0)}
        if t.feature[i] >= 0:
            j = int(t.feature[i])
            node.update({"left": int(t.left[i]), "right": int(t.right[i]), "feature": D.names[j],
                         "yes": rule_text(D, j, t.threshold[i], "left"), "no": rule_text(D, j, t.threshold[i], "right")})
        out.append(node)
    return out


def tree_rules(t, D):
    """Every leaf as a plain rule: the conditions on the way down, how many houses, how many collapsed."""
    rules = []

    def walk(i, conds):
        if t.feature[i] < 0:
            rules.append({"if": merged_rule(D, conds), "leaf": int(i), "share": float(t.weight[i] / t.weight[0]), "p": float(t.value[i][1])})
            return
        j = int(t.feature[i])
        walk(t.left[i], conds + [(j, float(t.threshold[i]), "left")])
        walk(t.right[i], conds + [(j, float(t.threshold[i]), "right")])
    walk(0, [])
    return sorted(rules, key=lambda r: -r["p"])


def grade_metrics(g, P, majority):
    pred = P.argmax(1)
    expected = (P * np.arange(1, 6)).sum(1)
    M = confusion(g - 1, pred, 5)
    return {"exact": float((pred + 1 == g).mean()), "within_one": float((np.abs(pred + 1 - g) <= 1).mean()),
            "majority_exact": float((g == majority).mean()), "majority_within_one": float((np.abs(g - majority) <= 1).mean()),
            "mae_expected": float(np.abs(expected - g).mean()), "confusion": M.tolist(),
            "confusion_rows": (M / np.maximum(M.sum(1, keepdims=True), 1)).round(4).tolist(),
            "log_loss": float(-np.mean(np.log(np.clip(P[np.arange(len(g)), g - 1], 1e-9, 1))))}


def main():
    t_start = time.time()
    config.ARTIFACTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(config.SEED)

    # ------------------------------------------------------------------ data
    b, wards = data.load()
    parts = data.split(b)
    levels = data.fit_levels(b, parts["train"])
    D = data.design(b, levels)
    D0 = data.design(b, levels, with_shaking=False)
    y = b["collapse"].to_numpy()
    grade = b["damage_grade"].to_numpy()
    Xtr, Xva, Xs, Xu = (D.X[parts[k]] for k in ("train", "val", "test_seen", "test_unseen"))
    ytr, yva, ys, yu = (y[parts[k]] for k in ("train", "val", "test_seen", "test_unseen"))
    J_SHAKE = D.source.index("pred_intensity")
    log(f"{len(b):,} buildings, {D.X.shape[1]} columns; train {len(ytr):,}, val {len(yva):,}, seen test {len(ys):,}, unseen test {len(yu):,}")
    sub_u = np.sort(rng.choice(len(yu), 30_000, replace=False))       # for curves that need many re-predictions

    # ------------------------------------------------------------------ single models
    lg = Logistic(ridge=1.0).fit(Xtr, ytr)
    log("logistic regression")
    full = Tree(16, 20, smooth=1).fit(Xtr, ytr)
    path = full.pruning_path()
    path_rows = []
    for alpha, coll, leaves in path:
        pv = full.pruned(coll).predict_proba(Xva)[:, 1]
        path_rows.append({"alpha": alpha, "leaves": leaves, "val_brier": float(np.mean((pv - yva) ** 2)), "val_auc": auc(yva, pv)})
    best_i = int(np.argmin([r["val_brier"] for r in path_rows]))
    cart = full.pruned(path[best_i][1])
    small_i = min(i for i, r in enumerate(path_rows) if r["leaves"] <= config.CHECKLIST_LEAVES)
    checklist = full.pruned(path[small_i][1])
    log(f"CART: full tree {full.n_leaves()} leaves, pruned to {cart.n_leaves()}; checklist {checklist.n_leaves()} leaves")

    lams = (0.3, 1, 3, 10, 30, 100)
    gams = []
    for lam in lams:
        g = GAM(lam=lam).fit(Xtr, ytr, D.kind, D.names)
        gams.append({"lam": lam, "model": g, "val_log_loss": log_loss(yva, g.predict_proba(Xva)), "edf": float(sum(g.edf))})
    gi = int(np.argmin([r["val_log_loss"] for r in gams]))
    gam = gams[gi]["model"]
    log(f"GAM: lambda {lams[gi]} chosen on validation")

    # ------------------------------------------------------------------ ensembles
    bag = Forest(config.N_BAG, None, config.FOREST_LEAF, seed=1).fit(Xtr, ytr)
    log(f"bagging: {config.N_BAG} trees")
    rf = Forest(config.N_FOREST, "sqrt", config.FOREST_LEAF, seed=2).fit(Xtr, ytr)
    log(f"random forest: {config.N_FOREST} trees")
    ada = AdaBoost(config.N_ADA, 1, seed=3).fit(Xtr, ytr)
    log(f"AdaBoost: {len(ada.alphas)} stumps")
    gb = GradientBoosting(config.N_GB, 4, config.GB_RATE, 0.8, 20, seed=4).fit(Xtr, ytr, Xva, yva)
    log(f"gradient boosting: best {gb.best_n} of {config.N_GB} trees")
    models = {"checklist": checklist, "cart": cart, "bagging": bag, "forest": rf, "adaboost": ada, "gboost": gb, "gam": gam, "logit": lg}

    P_val = Ensemble(models, None).base(Xva)
    stacker = Stacker(ridge=1.0).fit(np.column_stack([P_val[k] for k in BASE]), yva)
    ens = Ensemble(models, stacker)
    val_ll = {k: {"total": float(np.sum(yva * np.log(np.clip(P_val[k], 1e-9, 1)) + (1 - yva) * np.log(np.clip(1 - P_val[k], 1e-9, 1)))), "n": len(yva)} for k in BASE}
    fam = family_weights(val_ll)
    log(f"stacking weights {dict(zip(BASE, np.round(stacker.weights, 2)))}")

    # ------------------------------------------------------------------ every building, every model
    P_all = ens.predict(D.X)
    log("predicted all buildings")
    Ps = {k: v[parts["test_seen"]] for k, v in P_all.items()}
    Pu = {k: v[parts["test_unseen"]] for k, v in P_all.items()}

    # Subset BMA: all 4,096 logistic models over 12 groups of facts
    groups = [(name, [j for j, s in enumerate(D.source) if s in srcs]) for name, srcs in config.BMA_GROUPS]
    bma_rows = np.sort(rng.choice(parts["train"], config.N_BMA, replace=False))
    bma_eval = np.r_[parts["test_seen"], parts["test_unseen"]]
    bm = subset_bma(D.X[bma_rows], y[bma_rows], D.X[bma_eval], groups)
    Ps["bma"], Pu["bma"] = bm["averaged"][:len(ys)], bm["averaged"][len(ys):]
    bma_best_s, bma_best_u = bm["best_single"][:len(ys)], bm["best_single"][len(ys):]
    log(f"BMA over {bm['n_models']} models; {bm['models_for_99']} hold 99% of the weight")

    # Baselines a planner could use without any model
    fragile = b[["has_superstructure_mud_mortar_stone", "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_brick", "has_superstructure_stone_flag"]].to_numpy().max(1)
    shake = b["pred_intensity"].to_numpy()
    jitter = rng.random(len(b)) * 1e-6
    for P, rows in ((Ps, parts["test_seen"]), (Pu, parts["test_unseen"])):
        P["shaking"] = shake[rows] + jitter[rows]
        P["rule"] = fragile[rows] * 10 + shake[rows] + jitter[rows]
        P["random"] = rng.random(len(rows))

    # Models without the shaking map, to show what the map adds
    lg0 = Logistic(ridge=1.0).fit(D0.X[parts["train"]], ytr)
    gb0 = GradientBoosting(config.N_GB, 4, config.GB_RATE, 0.8, 20, seed=5).fit(D0.X[parts["train"]], ytr, D0.X[parts["val"]], yva)
    no_shake = {}
    for name, m in (("logit", lg0), ("gboost", gb0)):
        no_shake[name] = {"seen": auc(ys, m.predict_proba(D0.X[parts["test_seen"]])), "unseen": auc(yu, m.predict_proba(D0.X[parts["test_unseen"]]))}
    log("no-shaking comparison")

    # ------------------------------------------------------------------ scores
    order = ["checklist"] + BASE + ["crowd", "stack", "bma"]
    scores = {}
    for name in order:
        scores[name] = {"label": LABEL[name], "kind": KIND[name], "seen": summary(ys, Ps[name], config.CAPACITY), "unseen": summary(yu, Pu[name], config.CAPACITY)}
    scores["bma"]["best_single_auc"] = {"seen": auc(ys, bma_best_s), "unseen": auc(yu, bma_best_u)}
    base_scores = {k: {"label": LABEL[k], "seen": auc(ys, Ps[k]), "unseen": auc(yu, Pu[k])} for k in ("shaking", "rule", "random")}

    pairs = [("crowd", "cart"), ("forest", "cart"), ("bagging", "cart"), ("gboost", "cart"), ("adaboost", "cart"), ("stack", "gboost"),
             ("crowd", "gboost"), ("gam", "logit"), ("gboost", "gam"), ("bma", "logit")]
    boot = []
    for setname, yy, P in (("seen", ys, Ps), ("unseen", yu, Pu)):
        idx = np.sort(rng.choice(len(yy), config.N_BOOT_ROWS, replace=False))
        for a, c in pairs:
            r = bootstrap_auc_gain(yy[idx], P[a][idx], P[c][idx], reps=config.N_BOOT, seed=len(boot))
            boot.append({"set": setname, "a": a, "b": c, "label": f"{LABEL[a]} vs {LABEL[c]}", **r})
    log("bootstrap intervals")

    roc = {k: dict(zip(("fpr", "tpr"), roc_points(yu, Pu[k], 80))) for k in ("checklist", "cart", "forest", "gboost", "gam", "logit", "crowd", "stack", "shaking", "rule")}
    calib = {s: {k: calibration(yy, P[k], 10) for k in ("cart", "forest", "adaboost", "gboost", "gam", "crowd", "stack")} for s, yy, P in (("seen", ys, Ps), ("unseen", yu, Pu))}
    capacity = {k: capacity_curve(yu, Pu[k], SHARES) for k in ("crowd", "stack", "gboost", "cart", "checklist", "shaking", "rule", "random")}
    nu, cu = len(yu), int(yu.sum())
    wid_u = b["ward_id"].to_numpy()[parts["test_unseen"]]
    ward_rank = {k: within_ward(yu, Pu[k], wid_u) for k in ("crowd", "stack", "gboost", "gam", "forest", "bagging", "adaboost", "logit", "cart", "checklist", "rule", "random")}
    log(f"within-ward ranking: crowd {ward_rank['crowd']['auc']:.3f}, rule {ward_rank['rule']['auc']:.3f}")
    at20 = {k: capacity[k][SHARES.index(0.2)] for k in capacity}

    # ------------------------------------------------------------------ how each ensemble grows
    yu_s, Xu_s = yu[sub_u], Xu[sub_u]
    steps = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 160, 200, 300, 400]
    f_steps = [s for s in steps if s <= len(rf.trees)]
    rf_staged = rf.staged_proba(Xu_s, set(f_steps))
    bag_steps = [s for s in steps if s <= len(bag.trees)]
    bag_staged = bag.staged_proba(Xu_s, set(bag_steps))
    growth = {
        "forest": {"n": f_steps, "auc": [auc(yu_s, rf_staged[s]) for s in f_steps], "oob_error": rf.oob_curve},
        "bagging": {"n": bag_steps, "auc": [auc(yu_s, bag_staged[s]) for s in bag_steps], "oob_error": bag.oob_curve},
        "adaboost": {"n": [s for s in steps if s <= len(ada.alphas)], "errors": ada.errors, "alphas": ada.alphas},
        "gboost": {"train": gb.train_curve, "val": gb.val_curve, "best_n": gb.best_n},
        "single_tree_auc": [auc(yu_s, t.predict_proba(Xu_s)[:, 1]) for t in rf.trees[:40]],
        "single_bag_auc": [auc(yu_s, t.predict_proba(Xu_s)[:, 1]) for t in bag.trees[:40]],
    }
    growth["adaboost"]["auc"] = [auc(yu_s, ada.predict_proba(Xu_s, s)) for s in growth["adaboost"]["n"]]
    g_steps = sorted(set([s for s in steps if s <= gb.best_n] + [gb.best_n]))
    growth["gboost"]["n"] = g_steps
    growth["gboost"]["auc"] = [auc(yu_s, gb.predict_proba(Xu_s, s)) for s in g_steps]
    stumps = []
    for a_, t in list(zip(ada.alphas, ada.trees))[:12]:
        j = int(t.feature[0])
        if j < 0:
            continue
        right_collapse = t.value[t.right[0]][1] > t.value[t.left[0]][1]
        stumps.append({"feature": D.names[j], "alpha": a_, "collapse_if": rule_text(D, j, t.threshold[0], "right" if right_collapse else "left")})
    log("growth curves")

    # Adaptive basis functions: one house's log-odds built up tree by tree
    ex_u = int(np.argmax(Pu["gboost"] * (b["district_id"].to_numpy()[parts["test_unseen"]] == 36)))
    x1 = Xu[ex_u:ex_u + 1]
    contrib = [gb.basis(x1, k)[0] for k in range(gb.best_n)]
    basis = {"f0": gb.f0, "cumulative": (gb.f0 + np.cumsum(contrib)).tolist(), "first": contrib[:40], "p_final": float(gb.predict_proba(x1)[0])}

    # ------------------------------------------------------------------ fragility curves
    wt = b["wall_type"].to_numpy()
    lo_i, hi_i = np.floor(shake.min() * 4) / 4, np.ceil(shake.max() * 4) / 4
    band_edges = np.arange(lo_i, hi_i + 1e-9, 0.25)
    mids = ((band_edges[:-1] + band_edges[1:]) / 2).round(3)
    band = np.clip(np.digitize(shake, band_edges) - 1, 0, len(mids) - 1)
    grid = np.round(np.arange(6.25, 8.51, 0.125), 3)
    eval_rows = np.r_[parts["val"], parts["test_seen"], parts["test_unseen"]]
    fragility = {"bands": mids.tolist(), "grid": grid.tolist(), "types": {}}
    for name in config.FRAGILITY_TYPES:
        m = wt == name
        rates, ns = [], []
        for k in range(len(mids)):
            mm = m & (band == k)
            ns.append(int(mm.sum()))
            rates.append(float(y[mm].mean()) if mm.sum() >= config.MIN_CELL else None)
        pool = eval_rows[wt[eval_rows] == name]
        pick = rng.choice(pool, min(len(pool), 1500), replace=False)
        Xp = D.X[pick].copy()
        curve = []
        for gval in grid:
            Xp[:, J_SHAKE] = gval
            curve.append(float(ens.predict(Xp)["crowd"].mean()))
        fragility["types"][name] = {"n": int(m.sum()), "observed": rates, "observed_n": ns, "model": curve}
    gam_shake = gam.curve(J_SHAKE, grid)
    fragility["gam_shaking"] = (gam_shake - gam_shake[0]).tolist()
    fragility["all_observed"] = [float(y[band == k].mean()) if (band == k).sum() >= config.MIN_CELL else None for k in range(len(mids))]
    fragility["all_n"] = [int((band == k).sum()) for k in range(len(mids))]
    wall_rates = {name: {"n": int((wt == name).sum()), "rate": float(y[wt == name].mean())} for name in dict.fromkeys(wt) if (wt == name).sum() >= config.MIN_CELL}
    fl = np.clip(b["count_floors_pre_eq"].to_numpy(), 1, 5)
    floor_rates = {"floors": ["1", "2", "3", "4", "5+"], "rate": [float(y[fl == k].mean()) for k in range(1, 6)], "n": [int((fl == k).sum()) for k in range(1, 6)]}
    log("fragility curves")

    # ------------------------------------------------------------------ GAM shapes and effects
    gam_terms = []
    for j, t in enumerate(gam.terms):
        col = t["col"]
        if t["type"] == "spline":
            x = Xtr[:, col]
            g_ = np.linspace(t["lo"], t["hi"], 60)
            hist, edges = np.histogram(x, bins=30, range=(t["lo"], t["hi"]))
            gam_terms.append({"name": D.names[col], "source": D.source[col], "type": "spline", "grid": g_.tolist(), "curve": gam.curve(j, g_).tolist(),
                              "hist_x": ((edges[:-1] + edges[1:]) / 2).tolist(), "hist_n": hist.tolist(), "edf": gam.edf[j]})
        else:
            gam_terms.append({"name": D.names[col], "source": D.source[col], "type": "linear", "effect": float(gam.betas[j][0] / t["sd"]), "edf": gam.edf[j]})
    log("GAM shapes")

    # ------------------------------------------------------------------ what drives collapse
    perm = []
    Xp = Xu_s.copy()
    base_auc = auc(yu_s, gb.predict_proba(Xp))
    for src in dict.fromkeys(D.source):
        cols = D.columns_of(src)
        drops = []
        for r in range(2):
            Xp2 = Xp.copy()
            Xp2[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
            drops.append(base_auc - auc(yu_s, gb.predict_proba(Xp2)))
        perm.append({"source": src, "name": config.NICE[src], "drop": float(np.mean(drops))})
    perm.sort(key=lambda r: -r["drop"])
    imp_rf = {}
    for j, src in enumerate(D.source):
        imp_rf[config.NICE[src]] = imp_rf.get(config.NICE[src], 0) + float(rf.importance[j])
    log("importance")

    # ------------------------------------------------------------------ what would an upgrade change?
    upgrade_pool = parts["test_unseen"]
    upgrades = []
    for key, label, applies, change in config_upgrades():
        hit = upgrade_pool[applies(b.iloc[upgrade_pool])]
        if len(hit) < 200:
            continue
        hit = rng.choice(hit, min(len(hit), 6000), replace=False)
        before_df = b.iloc[hit].copy()
        after_df = change(before_df.copy())
        p0 = ens.predict(data.design(before_df, levels).X)["crowd"]
        p1 = ens.predict(data.design(after_df, levels).X)["crowd"]
        upgrades.append({"key": key, "label": label, "n": int(len(hit)), "before": float(p0.mean()), "after": float(p1.mean()),
                         "drop": float(p0.mean() - p1.mean()), "observed_before": float(y[hit].mean())})
    log("upgrade what-ifs")

    # ------------------------------------------------------------------ districts and the ward map
    head = P_all[config.HEADLINE]
    dist = []
    did = b["district_id"].to_numpy()
    in_train = np.zeros(len(b), bool)
    in_train[parts["train"]] = True
    for d_id, name in sorted(wards.drop_duplicates("district_id")[["district_id", "district_name"]].itertuples(index=False)):
        m = did == d_id
        if not m.any():
            continue
        dist.append({"id": int(d_id), "name": name, "unseen": d_id in config.UNSEEN_DISTRICTS, "n": int(m.sum()), "collapsed": int(y[m].sum()),
                     "predicted": float(head[m].sum()), "predicted_out_of_sample": float(head[m & ~in_train].sum() / max((m & ~in_train).sum(), 1) * m.sum()),
                     "rate": float(y[m].mean()), "shaking": float(shake[m].mean()),
                     "auc": auc(y[m & ~in_train], head[m & ~in_train])})
    wid = b["ward_id"].to_numpy()
    per_ward = {}
    for i in rng.permutation(len(b)):
        lst = per_ward.setdefault(wid[i], [])
        if len(lst) < config.WARD_SAMPLE:
            lst.append(i)
    ward_rows = np.concatenate([np.array(v) for v in per_ward.values()])
    ward_of = wid[ward_rows]
    Xw = D.X[ward_rows].copy()
    base_shake = Xw[:, J_SHAKE].copy()
    by_shift = {}
    for sft in SHIFTS:
        Xw[:, J_SHAKE] = base_shake + sft
        by_shift[sft] = ens.predict(Xw)[config.HEADLINE]
    ward_list = []
    wmeta = wards.set_index("ward_id")
    for w_ in sorted(per_ward):
        m = wid == w_
        mm = ward_of == w_
        r = wmeta.loc[w_]
        ward_list.append({"id": int(w_), "lat": float(r["latitude"]), "lon": float(r["longitude"]), "place": str(r["vdcmun_name"]), "district": str(r["district_name"]),
                          "unseen": int(r["district_id"]) in config.UNSEEN_DISTRICTS, "shaking": float(r["pred_intensity"]), "n": int(m.sum()),
                          "actual": float(y[m].mean()), "model": float(head[m].mean()), "shift": [float(by_shift[s][mm].mean()) for s in SHIFTS]})
    shift_totals = [float(sum(w["shift"][i] * w["n"] for w in ward_list)) for i in range(len(SHIFTS))]
    ward_fit = {}
    for name, flag in (("seen", False), ("unseen", True)):
        ws_ = [w for w in ward_list if w["unseen"] == flag]
        a_, m_ = np.array([w["actual"] for w in ws_]), np.array([w["model"] for w in ws_])
        ward_fit[name] = {"wards": len(ws_), "r": float(np.corrcoef(a_, m_)[0, 1]), "mae": float(np.average(np.abs(a_ - m_), weights=[w["n"] for w in ws_]))}
    log(f"{len(ward_list)} wards mapped at {len(SHIFTS)} shaking levels")

    # ------------------------------------------------------------------ sort houses into all 5 damage grades
    gtr, gva = grade[parts["train"]], grade[parts["val"]]
    t5 = Tree(16, 20, n_classes=5, smooth=1).fit(Xtr, gtr - 1)
    p5 = t5.pruning_path()
    best5 = min(p5, key=lambda a: -np.mean(np.log(np.clip(t5.pruned(a[1]).predict_proba(Xva)[np.arange(len(gva)), gva - 1], 1e-9, 1))))
    cart5 = t5.pruned(best5[1])
    rf5 = Forest(config.N_FOREST5, "sqrt", config.FOREST_LEAF, seed=6, n_classes=5).fit(Xtr, gtr - 1)
    majority = int(np.bincount(gtr).argmax())
    grades = {"labels": [config.GRADES[k] for k in range(1, 6)], "train_counts": np.bincount(gtr, minlength=6)[1:].tolist(), "majority": majority,
              "cart_leaves": cart5.n_leaves(), "forest_trees": len(rf5.trees)}
    for setname, rows in (("seen", parts["test_seen"]), ("unseen", parts["test_unseen"])):
        gg = grade[rows]
        grades[setname] = {"cart": grade_metrics(gg, cart5.predict_proba(D.X[rows]), majority), "forest": grade_metrics(gg, rf5.predict_proba_all(D.X[rows]), majority),
                           "counts": np.bincount(gg, minlength=6)[1:].tolist()}
    log("five-grade CART and forest")

    # ------------------------------------------------------------------ example houses and the builder's choices
    examples = []
    ud = b.iloc[parts["test_unseen"]]
    picks = [("A stone house near the epicentre", (ud["wall_type"] == "Stone in mud mortar") & (ud["district_id"] == 36) & (ud["count_floors_pre_eq"] == 2) & (ud["collapse"] == 1), "high"),
             ("A brick and cement house in Dhading", (ud["wall_type"] == "Brick in cement mortar") & (ud["district_id"] == 30) & (ud["collapse"] == 0), "mid"),
             ("An engineered concrete frame", (ud["wall_type"] == "RC frame, engineered") & (ud["collapse"] == 0), "low")]
    for title, mask, which in picks:
        cand = np.where(mask.to_numpy())[0]
        if not len(cand):
            continue
        pc = Pu[config.HEADLINE][cand]
        k = cand[{"high": np.argsort(pc)[int(len(pc) * .8)], "mid": np.argsort(pc)[len(pc) // 2], "low": np.argsort(pc)[len(pc) // 2]}[which]]
        row = ud.iloc[k]
        examples.append({"title": title, "house": house_dict(row), "district": config.UNSEEN_DISTRICTS.get(int(row["district_id"]), ""),
                         "grade": int(row["damage_grade"]), "collapsed": bool(row["collapse"]), "p": {m: float(Pu[m][k]) for m in ("checklist", "cart", "forest", "gboost", "gam", "crowd", "stack")}})
    train_df = b.iloc[parts["train"]]
    choices = {c: [v for v in train_df[c].value_counts().index if v != "Not recorded"] for c in config.CATEGORICAL}
    numeric_range = {c: {"min": float(np.quantile(train_df[c], .005)), "max": float(np.quantile(train_df[c], .995)), "median": float(train_df[c].median())} for c in config.NUMERIC}
    district_shaking = {d["name"]: round(d["shaking"], 2) for d in dist}

    # ------------------------------------------------------------------ write
    counts = np.bincount(grade, minlength=6)[1:]
    rep = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "train_seconds": time.time() - t_start,
        "data": {"buildings": len(b), "wards": len(ward_list), "districts": len(dist), "collapse_rate": float(y.mean()), "grade_counts": counts.tolist(),
                 "grade_labels": [config.GRADES[k] for k in range(1, 6)], "n_train": len(ytr), "n_val": len(yva), "n_test_seen": len(ys), "n_test_unseen": len(yu),
                 "unseen_districts": list(config.UNSEEN_DISTRICTS.values()), "columns": D.names, "n_columns": len(D.names),
                 "wall_types": {k: int(v) for k, v in zip(*np.unique(wt, return_counts=True))}, "shaking_range": [float(shake.min()), float(shake.max())],
                 "age_unknown": int(b["age_unknown"].sum()), "rate_seen": float(ys.mean()), "rate_unseen": float(yu.mean())},
        "scores": scores, "baselines": base_scores, "no_shaking": no_shake, "bootstrap": boot, "roc": roc, "calibration": calib,
        "capacity": {"shares": SHARES, "curves": {k: [c["recall"] for c in v] for k, v in capacity.items()}, "at20": at20, "unseen_houses": nu, "unseen_collapsed": cu},
        "cart": {"path": path_rows, "chosen": best_i, "leaves": cart.n_leaves(), "full_leaves": full.n_leaves(), "depth": cart.depth(),
                 "checklist": {"leaves": checklist.n_leaves(), "nodes": tree_nodes(checklist, D), "rules": tree_rules(checklist, D)}},
        "gam": {"lams": [{"lam": r["lam"], "val_log_loss": r["val_log_loss"], "edf": r["edf"]} for r in gams], "lam": lams[gi], "terms": gam_terms, "deviance": gam.deviance_path},
        "stack": {"weights": dict(zip(BASE, stacker.weights.tolist())), "intercept": float(stacker.model.beta[0])},
        "family_bma": fam, "crowd_members": CROWD,
        "bma": {"n_models": bm["n_models"], "inclusion": bm["inclusion"], "top": bm["top"], "models_for_99": bm["models_for_99"], "weights_sorted": bm["weights_sorted"],
                "rows": config.N_BMA, "groups": [g[0] for g in groups]},
        "growth": growth, "stumps": stumps, "basis": basis, "fragility": fragility, "importance": {"permutation": perm, "forest": imp_rf},
        "ward_rank": ward_rank, "ward_fit": ward_fit, "wall_rates": wall_rates, "floor_rates": floor_rates, "upgrades": upgrades, "districts": dist, "wards": ward_list, "shifts": SHIFTS, "shift_totals": shift_totals, "grades": grades,
        "examples": examples, "headline": config.HEADLINE, "labels": LABEL, "kinds": KIND, "sources": config.SOURCES,
    }
    model = {"design": {"names": D.names, "source": D.source, "kind": D.kind}, "levels": levels, "choices": choices, "level_nice": config.LEVEL_NICE, "numeric": numeric_range,
             "district_shaking": district_shaking, "age_median": float(b.loc[b["age_unknown"] == 0, "age_building"].median()),
             "ensemble": ens.state(), "checklist_nodes": tree_nodes(checklist, D), "headline": config.HEADLINE}
    (config.ARTIFACTS / "model.json").write_text(json.dumps(clean(model), separators=(",", ":")), encoding="utf-8")
    (config.ARTIFACTS / "report.json").write_text(json.dumps(clean(rep), separators=(",", ":")), encoding="utf-8")
    log(f"model.json {(config.ARTIFACTS / 'model.json').stat().st_size / 1e6:.1f} MB, report.json {(config.ARTIFACTS / 'report.json').stat().st_size / 1e6:.1f} MB")
    try:
        from .charts import write_graphs
        write_graphs(json.loads(json.dumps(clean(rep))))
    except Exception as e:                      # the report is saved; graphs can be rebuilt with python -m backend.charts
        log(f"graphs not written: {e!r}")
    log("done")
    return rep


def house_dict(row):
    h = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in config.NUMERIC + config.CATEGORICAL}
    h["walls"] = [c for c in config.WALLS if int(row[c]) == 1]
    return h


def config_upgrades():
    """(key, label, which houses it applies to, how the house changes). Associations learned from the survey, not proof."""
    def mud_to_cement(df):
        for mud, cem in (("has_superstructure_mud_mortar_stone", "has_superstructure_cement_mortar_stone"), ("has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick")):
            hit = df[mud] == 1
            df.loc[hit, mud] = 0
            df.loc[hit, cem] = 1
        df.loc[df["foundation_type"] == "Mud mortar-Stone/Brick", "foundation_type"] = "Cement-Stone/Brick"
        return df

    def light_roof(df):
        df.loc[df["roof_type"] == "Bamboo/Timber-Heavy roof", "roof_type"] = "Bamboo/Timber-Light roof"
        return df

    def one_floor_less(df):
        per = df["height_ft_pre_eq"] / df["count_floors_pre_eq"]
        df["count_floors_pre_eq"] = df["count_floors_pre_eq"] - 1
        df["height_ft_pre_eq"] = (df["height_ft_pre_eq"] - per).round()
        return df

    def rebuild_rc(df):
        for c in config.WALLS:
            df[c] = 0
        df["has_superstructure_rc_engineered"] = 1
        df["foundation_type"], df["ground_floor_type"], df["roof_type"] = "RC", "RC", "RCC/RB/RBC"
        df.loc[df["count_floors_pre_eq"] > 1, "other_floor_type"] = "RCC/RB/RBC"
        return df

    mud = lambda df: ((df["has_superstructure_mud_mortar_stone"] == 1) | (df["has_superstructure_mud_mortar_brick"] == 1)).to_numpy()
    return [
        ("cement", "Mud mortar replaced by cement mortar", mud, mud_to_cement),
        ("roof", "Heavy roof replaced by a light one", lambda df: (df["roof_type"] == "Bamboo/Timber-Heavy roof").to_numpy(), light_roof),
        ("floor", "One storey fewer", lambda df: (mud(df) & (df["count_floors_pre_eq"] >= 3).to_numpy()), one_floor_less),
        ("rc", "Rebuilt as an engineered concrete frame", mud, rebuild_rc),
    ]


if __name__ == "__main__":
    main()
