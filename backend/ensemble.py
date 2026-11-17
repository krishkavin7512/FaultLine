"""Combining models: bagging, random forests, AdaBoost, gradient boosting and stacking, written out in NumPy.

Every model here is a sum or average of simple trees, which makes each one an adaptive basis function model:
    f(x) = w0 + sum_m  w_m * phi_m(x)
where each basis function phi_m is a tree whose splits are learned from the data, not fixed in advance.
"""
import numpy as np

from .linear import Logistic, sigmoid
from .tree import Binner, Tree


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Forest:
    """Bagging when max_features is None; a random forest when each split only sees a random subset of features.
    Works for two classes (collapsed or not) and for several (the five damage grades)."""

    def __init__(self, n_trees=200, max_features="sqrt", min_leaf=3, max_depth=30, seed=0, n_classes=2):
        self.n_trees, self.max_features, self.min_leaf, self.max_depth, self.seed = n_trees, max_features, min_leaf, max_depth, seed
        self.K = n_classes

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        mf = {"sqrt": max(1, int(np.sqrt(d))), None: None}.get(self.max_features, self.max_features)
        self.binner = Binner().fit(X)
        B = self.binner.transform(X)
        self.trees = []
        oob_sum, oob_n = np.zeros((n, self.K)), np.zeros(n)
        self.oob_curve = []
        for t in range(self.n_trees):
            boot = rng.integers(0, n, n)
            out = np.setdiff1d(np.arange(n), boot)
            tree = Tree(self.max_depth, self.min_leaf, mf, n_classes=self.K, seed=int(rng.integers(1 << 30)), binner=self.binner)
            tree.fit(X[boot], y[boot], B=B[boot])
            self.trees.append(tree)
            oob_sum[out] += tree.predict_proba(X[out])
            oob_n[out] += 1
            seen = oob_n > 0
            self.oob_curve.append(float(np.mean(oob_sum[seen].argmax(1) != y[seen])))
        self.oob_proba = np.where(oob_n[:, None] > 0, oob_sum / np.maximum(oob_n, 1)[:, None], np.nan)
        self.importance = np.mean([t.importance for t in self.trees], 0)
        return self

    def predict_proba_all(self, X, n=None):
        trees = self.trees[:n] if n else self.trees
        return np.mean([t.predict_proba(X) for t in trees], 0)

    def predict_proba(self, X, n=None):
        """Probability of class 1 (collapse) for the two-class forest."""
        return self.predict_proba_all(X, n)[:, 1]

    def staged_proba(self, X, steps):
        acc = np.zeros(len(X))
        out = {}
        for i, t in enumerate(self.trees, 1):
            acc += t.predict_proba(X)[:, 1]
            if i in steps:
                out[i] = acc / i
        return out

    def state(self):
        return {"kind": "forest", "trees": [t.state(compact=self.K == 2) for t in self.trees]}

    @classmethod
    def from_state(cls, s):
        f = cls()
        f.trees = [Tree.from_state(t) for t in s["trees"]]
        f.K = f.trees[0].K
        return f


class AdaBoost:
    """Discrete AdaBoost. Each round fits a small tree to re-weighted data, then up-weights the houses it got wrong.
    The combined score F estimates half the log-odds, so p = 1 / (1 + exp(-2F))."""

    def __init__(self, n_rounds=200, depth=1, seed=0):
        self.n_rounds, self.depth, self.seed = n_rounds, depth, seed

    def fit(self, X, y):
        self.binner = Binner().fit(X)
        B = self.binner.transform(X)
        n = len(y)
        w = np.full(n, 1 / n)
        ys = 2 * y - 1
        self.trees, self.alphas, self.errors = [], [], []
        for _ in range(self.n_rounds):
            t = Tree(self.depth, 1, binner=self.binner).fit(X, y, w, B=B)
            h = 2 * t.predict(X) - 1
            err = float(np.clip(w[h != ys].sum(), 1e-10, 1 - 1e-10))
            if err >= 0.5:
                break
            a = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-a * ys * h)
            w /= w.sum()
            self.trees.append(t)
            self.alphas.append(float(a))
            self.errors.append(err)
        return self

    def score(self, X, n=None):
        F = np.zeros(len(X))
        for a, t in list(zip(self.alphas, self.trees))[:n]:
            F += a * (2 * t.predict(X) - 1)
        return F

    def predict_proba(self, X, n=None):
        return sigmoid(2 * self.score(X, n))

    def state(self):
        return {"kind": "adaboost", "alphas": self.alphas, "trees": [t.state(compact=True) for t in self.trees]}

    @classmethod
    def from_state(cls, s):
        a = cls()
        a.alphas, a.trees = list(s["alphas"]), [Tree.from_state(t) for t in s["trees"]]
        return a


class GradientBoosting:
    """Gradient boosting for log-loss. Start from the overall log-odds, then repeatedly fit a small regression tree to
    the residuals y - p and add it, shrunk by the learning rate. Leaves take a Newton step: sum(r) / sum(p(1-p))."""

    def __init__(self, n_trees=600, depth=3, learning_rate=0.05, subsample=0.8, min_leaf=10, seed=0):
        self.n_trees, self.depth, self.lr, self.subsample, self.min_leaf, self.seed = n_trees, depth, learning_rate, subsample, min_leaf, seed

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        self.binner = Binner().fit(X)
        B = self.binner.transform(X)
        n = len(y)
        self.f0 = float(np.log(y.mean() / (1 - y.mean())))
        F = np.full(n, self.f0)
        Fv = None if X_val is None else np.full(len(X_val), self.f0)
        self.trees, self.train_curve, self.val_curve = [], [], []
        for _ in range(self.n_trees):
            p = sigmoid(F)
            r, h = y - p, p * (1 - p)
            rows = rng.choice(n, int(self.subsample * n), replace=False)
            t = Tree(self.depth, self.min_leaf, task="regress", binner=self.binner).fit(X[rows], r[rows], B=B[rows], hess=h[rows])
            self.trees.append(t)
            F += self.lr * t.predict_value(X)[:, 0]
            self.train_curve.append(log_loss(y, sigmoid(F)))
            if Fv is not None:
                Fv += self.lr * t.predict_value(X_val)[:, 0]
                self.val_curve.append(log_loss(y_val, sigmoid(Fv)))
        self.best_n = int(np.argmin(self.val_curve)) + 1 if self.val_curve else self.n_trees
        return self

    def decision(self, X, n=None):
        n = n or self.best_n
        F = np.full(len(X), self.f0)
        for t in self.trees[:n]:
            F += self.lr * t.predict_value(X)[:, 0]
        return F

    def predict_proba(self, X, n=None):
        return sigmoid(self.decision(X, n))

    def basis(self, X, k):
        """The k-th learned basis function (one tree's contribution), for showing what boosting adds."""
        return self.lr * self.trees[k].predict_value(X)[:, 0]

    def state(self):
        return {"kind": "gboost", "f0": self.f0, "lr": self.lr, "trees": [t.state() for t in self.trees[:self.best_n]]}

    @classmethod
    def from_state(cls, s):
        g = cls(learning_rate=s["lr"])
        g.f0, g.trees = s["f0"], [Tree.from_state(t) for t in s["trees"]]
        g.best_n = len(g.trees)
        return g


class Stacker:
    """Stacked generalisation: a logistic regression learns how to weigh each base model's log-odds.
    It is fitted on validation predictions, never on the data the base models trained on."""

    def __init__(self, ridge=1.0):
        self.model = Logistic(ridge=ridge, standardise=False)

    @staticmethod
    def _logit(P):
        P = np.clip(P, 1e-6, 1 - 1e-6)
        return np.log(P / (1 - P))

    def fit(self, P_val, y_val):
        self.model.fit(self._logit(P_val), y_val)
        self.weights = self.model.beta[1:]
        return self

    def predict_proba(self, P):
        return self.model.predict_proba(self._logit(P))

    def state(self):
        return self.model.state()

    @classmethod
    def from_state(cls, s):
        st = cls()
        st.model = Logistic.from_state(s)
        st.weights = st.model.beta[1:]
        return st
