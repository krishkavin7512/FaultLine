"""Logistic generalized additive model, fitted by local scoring with backfitting (Hastie and Tibshirani).

    log-odds of collapse = alpha + f_1(x_1) + f_2(x_2) + ... + f_p(x_p)

Each numeric variable gets a smooth curve built from cubic B-splines, with a penalty on wiggliness (P-splines, Eilers
and Marx). Flags and categories get a single coefficient each. Because the model is a sum, every house's risk splits
exactly into one contribution per variable, which is what makes it explainable.
"""
import numpy as np

from .linear import sigmoid


def bspline(x, lo, hi, n_seg, deg=3):
    """Cubic B-spline basis on equally spaced knots over [lo, hi], by the Cox-de Boor recursion."""
    dx = (hi - lo) / n_seg
    knots = lo + dx * np.arange(-deg, n_seg + deg + 1)
    x = np.clip(np.asarray(x, float), lo, hi - 1e-9 * max(hi - lo, 1))
    B = ((x[:, None] >= knots[:-1]) & (x[:, None] < knots[1:])).astype(float)
    for d in range(1, deg + 1):
        left = (x[:, None] - knots[:-(d + 1)]) / (knots[d:-1] - knots[:-(d + 1)])
        right = (knots[d + 1:] - x[:, None]) / (knots[d + 1:] - knots[1:-d])
        B = left * B[:, :-1] + right * B[:, 1:]
    return B


class GAM:
    def __init__(self, lam=3.0, n_seg=8, min_unique=7, ridge=1.0, n_outer=30, n_inner=30, tol=1e-6):
        self.lam, self.n_seg, self.min_unique, self.ridge = lam, n_seg, min_unique, ridge
        self.n_outer, self.n_inner, self.tol = n_outer, n_inner, tol

    # ------------------------------------------------------------------ terms
    def _make_terms(self, X, kinds):
        terms = []
        for j in range(X.shape[1]):
            x = X[:, j]
            if kinds[j] == "num" and len(np.unique(x)) >= self.min_unique:
                lo, hi = float(np.quantile(x, 0.005)), float(np.quantile(x, 0.995))
                if hi <= lo:
                    lo, hi = float(x.min()), float(x.max())
                nb = self.n_seg + 3
                D = np.diff(np.eye(nb), 2, axis=0)
                terms.append({"col": j, "type": "spline", "lo": lo, "hi": hi, "P": self.lam * D.T @ D + 1e-6 * np.eye(nb)})
            else:
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                terms.append({"col": j, "type": "linear", "mu": mu, "sd": sd, "P": np.array([[self.ridge]])})
        return terms

    def _basis(self, t, x):
        if t["type"] == "spline":
            return bspline(x, t["lo"], t["hi"], self.n_seg)
        return ((x - t["mu"]) / t["sd"])[:, None]

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y, kinds, names=None):
        self.names = names
        self.terms = self._make_terms(X, kinds)
        Bs = [self._basis(t, X[:, t["col"]]) for t in self.terms]
        n = len(y)
        self.alpha = float(np.log(y.mean() / (1 - y.mean())))
        f = [np.zeros(n) for _ in self.terms]
        betas = [np.zeros(B.shape[1]) for B in Bs]
        shifts = [0.0] * len(Bs)
        self.deviance_path = []
        for _ in range(self.n_outer):
            eta = self.alpha + np.sum(f, 0)
            mu = sigmoid(eta)
            w = np.clip(mu * (1 - mu), 1e-6, None)
            z = eta + (y - mu) / w                         # working response for this local-scoring step
            for _ in range(self.n_inner):                   # backfitting: refit one term at a time
                biggest = 0.0
                total = np.sum(f, 0)
                self.alpha = float(np.sum(w * (z - total)) / w.sum())
                for j, (B, t) in enumerate(zip(Bs, self.terms)):
                    r = z - self.alpha - (total - f[j])
                    BW = B * w[:, None]
                    betas[j] = np.linalg.solve(B.T @ BW + t["P"], BW.T @ r)
                    new = B @ betas[j]
                    shifts[j] = float(np.sum(w * new) / w.sum())   # centre each curve so alpha carries the level
                    new = new - shifts[j]
                    biggest = max(biggest, float(np.abs(new - f[j]).max()))
                    total = total - f[j] + new
                    f[j] = new
                if biggest < 1e-7:
                    break
            p = sigmoid(self.alpha + np.sum(f, 0))
            dev = float(-2 * np.sum(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))
            self.deviance_path.append(dev)
            if len(self.deviance_path) > 1 and abs(self.deviance_path[-2] - dev) < self.tol * max(dev, 1):
                break
        self.betas, self.shifts = betas, shifts
        mu = sigmoid(self.alpha + np.sum(f, 0))
        w = mu * (1 - mu)
        self.edf = [float(np.trace(np.linalg.solve(B.T @ (B * w[:, None]) + t["P"], B.T @ (B * w[:, None])))) for B, t in zip(Bs, self.terms)]
        return self

    # ------------------------------------------------------------------ use
    def contributions(self, X):
        """Each term's share of the log-odds for every row (columns follow the input columns)."""
        out = np.zeros((len(X), len(self.terms)))
        for j, t in enumerate(self.terms):
            out[:, j] = self._basis(t, X[:, t["col"]]) @ self.betas[j] - self.shifts[j]
        return out

    def decision(self, X):
        return self.alpha + self.contributions(X).sum(1)

    def predict_proba(self, X):
        return sigmoid(self.decision(X))

    def curve(self, j, grid):
        t = self.terms[j]
        return self._basis(t, np.asarray(grid, float)) @ self.betas[j] - self.shifts[j]

    def state(self):
        terms = []
        for t, b, s in zip(self.terms, self.betas, self.shifts):
            d = {k: v for k, v in t.items() if k != "P"}
            d.update({"beta": b.tolist(), "shift": s})
            terms.append(d)
        return {"alpha": self.alpha, "n_seg": self.n_seg, "terms": terms}

    @classmethod
    def from_state(cls, s):
        g = cls(n_seg=s["n_seg"])
        g.alpha = s["alpha"]
        g.terms = [{k: v for k, v in t.items() if k not in ("beta", "shift")} for t in s["terms"]]
        g.betas = [np.array(t["beta"]) for t in s["terms"]]
        g.shifts = [t["shift"] for t in s["terms"]]
        return g
