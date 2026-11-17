"""Scoring helpers, written out so each number on screen can be traced by hand."""
import numpy as np


def auc(y, s):
    """Probability that a random house that collapsed is ranked above a random one that stood (ties count half).
    Mann-Whitney form: average the ranks of tied scores, then sum the ranks of the collapsed houses."""
    y = np.asarray(y).astype(bool)
    u, inv, cnt = np.unique(np.asarray(s, float), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    order = np.argsort(-np.asarray(s), kind="mergesort")
    yy = np.asarray(y)[order]
    hits = np.cumsum(yy)
    prec = hits / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p) - y) ** 2))


def at_capacity(y, p, share):
    """If a programme can only reach the riskiest `share` of houses: how many of them really collapsed, and what
    share of all collapses does that cover?"""
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-np.asarray(p), kind="mergesort")[:k]
    hit = int(np.asarray(y)[top].sum())
    return {"flagged": k, "real": hit, "precision": hit / k, "recall": hit / max(int(np.sum(y)), 1)}


def capacity_curve(y, p, shares):
    return [at_capacity(y, p, s) for s in shares]


def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            out.append({"predicted": float(p[m].mean()), "observed": float(y[m].mean()), "n": int(m.sum())})
    return out


def roc_points(y, s, n=60):
    order = np.argsort(-np.asarray(s), kind="mergesort")
    yy = np.asarray(y)[order]
    tpr = np.r_[0, np.cumsum(yy) / yy.sum()]
    fpr = np.r_[0, np.cumsum(1 - yy) / (1 - yy).sum()]
    keep = np.unique(np.linspace(0, len(tpr) - 1, n).astype(int))
    return fpr[keep].tolist(), tpr[keep].tolist()


def confusion(y, pred, k):
    M = np.zeros((k, k), int)
    np.add.at(M, (np.asarray(y), np.asarray(pred)), 1)
    return M


def bootstrap_auc_gain(y, p_a, p_b, reps=400, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b) on the same houses."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(auc(y[i], p_a[i]) - auc(y[i], p_b[i]))
    d = np.array(d)
    return {"mean": float(d.mean()), "low": float(np.quantile(d, .025)), "high": float(np.quantile(d, .975)), "p_below_zero": float((d <= 0).mean())}


def summary(y, p, share):
    c = at_capacity(y, p, share)
    return {"auc": auc(y, p), "ap": average_precision(y, p), "log_loss": log_loss(y, p), "brier": brier(y, p),
            "accuracy": float(((p > 0.5) == y).mean()), "precision_at_capacity": c["precision"], "recall_at_capacity": c["recall"]}
