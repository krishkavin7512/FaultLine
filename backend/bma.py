"""Bayesian model averaging, in two forms.

1. Over explanations. Every subset of a list of candidate factors is its own logistic model. Each model's evidence
   p(data | model) is approximated by BIC, so with equal prior odds the posterior weight of model m is
   proportional to exp(-BIC_m / 2). Summing weights over the models that include a factor gives the probability that
   the factor matters at all.
2. Over model families. Trees, forests, boosting and the GAM are weighted by how probable they make held-out houses.
   With thousands of houses one model's likelihood dwarfs the rest, so the weights collapse onto it. Averaging
   then turns into choosing, which is why stacking, which learns weights for combining, can do better.
"""
from itertools import combinations

import numpy as np

from .linear import Logistic


def subset_bma(X_tr, y_tr, X_te, groups, max_models_to_average=200):
    """groups: list of (name, [column indices]). Fits all 2^G models on training rows."""
    G = len(groups)
    n = len(y_tr)
    rows = []
    for k in range(G + 1):
        for subset in combinations(range(G), k):
            cols = [c for g in subset for c in groups[g][1]]
            if cols:
                m = Logistic(ridge=1e-6).fit(X_tr[:, cols], y_tr)
                ll, npar = m.loglik, m.n_params()
            else:
                p0 = y_tr.mean()
                ll, npar = float(n * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))), 1
            rows.append({"subset": subset, "loglik": ll, "params": npar, "bic": -2 * ll + npar * np.log(n)})
    bic = np.array([r["bic"] for r in rows])
    logw = -(bic - bic.min()) / 2
    w = np.exp(logw)
    w /= w.sum()
    for r, wi in zip(rows, w):
        r["weight"] = float(wi)
    order = np.argsort(-w)
    inclusion = [float(sum(w[i] for i, r in enumerate(rows) if g in r["subset"])) for g in range(G)]
    # Model-averaged prediction from the models that hold almost all the posterior weight
    keep, mass = [], 0.0
    for i in order:
        keep.append(i)
        mass += w[i]
        if mass > 0.999 or len(keep) >= max_models_to_average:
            break
    preds, ws = [], []
    for i in keep:
        cols = [c for g in rows[i]["subset"] for c in groups[g][1]]
        p = Logistic(ridge=1e-6).fit(X_tr[:, cols], y_tr).predict_proba(X_te[:, cols]) if cols else np.full(len(X_te), y_tr.mean())
        preds.append(p)
        ws.append(w[i])
    ws = np.array(ws) / np.sum(ws)
    averaged = np.sum(np.array(preds) * ws[:, None], 0)
    best_cols = [c for g in rows[order[0]]["subset"] for c in groups[g][1]]
    best = Logistic(ridge=1e-6).fit(X_tr[:, best_cols], y_tr).predict_proba(X_te[:, best_cols]) if best_cols else np.full(len(X_te), y_tr.mean())
    full_cols = [c for g in range(G) for c in groups[g][1]]
    full = Logistic(ridge=1e-6).fit(X_tr[:, full_cols], y_tr).predict_proba(X_te[:, full_cols])
    top = [{"factors": [groups[g][0] for g in rows[i]["subset"]], "weight": float(w[i]), "bic": float(bic[i])} for i in order[:12]]
    return {"n_models": len(rows), "inclusion": dict(zip([g[0] for g in groups], inclusion)), "top": top,
            "models_for_99": int(np.searchsorted(np.cumsum(w[order]), 0.99) + 1), "averaged": averaged, "best_single": best, "full": full,
            "weights_sorted": w[order][:60].tolist()}


def family_weights(val_logliks):
    """Posterior-style weights from total held-out log-likelihood, and the same weights per house
    (dividing by the number of houses), to show how fast they collapse as data grows."""
    names = list(val_logliks)
    ll = np.array([val_logliks[k]["total"] for k in names])
    n = val_logliks[names[0]]["n"]
    w = np.exp(ll - ll.max())
    w /= w.sum()
    by_n = {}
    for m in (1, 5, 10, 25, 50, 100, 250, n):
        llm = ll * (m / n)
        wm = np.exp(llm - llm.max())
        by_n[int(m)] = (wm / wm.sum()).tolist()
    return {"names": names, "weights": w.tolist(), "by_houses": by_n}
