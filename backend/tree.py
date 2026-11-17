"""Classification and regression trees (CART), written out in NumPy.

Splits are searched over quantile bins of each feature, so one split search is a few `bincount` calls instead of a
sort. A classification tree grows by Gini impurity and can be pruned back by cost-complexity (weakest-link) pruning.
A regression tree fits gradient-boosting residuals and sets each leaf by a Newton step.
"""
import numpy as np


class Binner:
    """Quantile edges per feature. A value x falls in bin b = number of edges strictly below x, so
    "bin <= k" is exactly "x <= edges[k]" and every split has a real-valued threshold."""

    def __init__(self, max_bins=48):
        self.max_bins = max_bins

    def fit(self, X):
        self.edges = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= self.max_bins:
                e = u[:-1]
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            self.edges.append(e.astype(float))
        return self

    def transform(self, X):
        return np.column_stack([np.searchsorted(e, X[:, j], side="left") for j, e in enumerate(self.edges)]).astype(np.int16)


class Tree:
    """One binary tree stored as flat arrays: feature, threshold, left, right, value (-1 children mean a leaf)."""

    def __init__(self, max_depth=6, min_leaf=5, max_features=None, task="classify", n_classes=2, seed=0, binner=None, smooth=0.0):
        """smooth adds that many pseudo-houses of each class to every leaf (Laplace smoothing), so a pure leaf
        says 'very likely' instead of an impossible 0% or 100%. Leave it at 0 for weighted fits such as AdaBoost."""
        self.max_depth, self.min_leaf, self.max_features = max_depth, min_leaf, max_features
        self.task, self.K, self.rng, self.binner, self.smooth = task, n_classes, np.random.default_rng(seed), binner, smooth

    # ------------------------------------------------------------------ growing
    def fit(self, X, y, w=None, B=None, hess=None):
        """Classification: y holds class ids, w sample weights. Regression: y holds residuals, hess the Newton weights."""
        if self.binner is None:
            self.binner = Binner().fit(X)
        B = self.binner.transform(X) if B is None else B
        n = len(y)
        w = np.ones(n) if w is None else np.asarray(w, float)
        self.n_features = B.shape[1]
        self.nbins = np.array([len(e) + 1 for e in self.binner.edges])
        self.feature, self.threshold, self.left, self.right, self.value, self.risk, self.weight = [], [], [], [], [], [], []
        self.importance = np.zeros(self.n_features)
        self._grow(B, np.asarray(y), w, None if hess is None else np.asarray(hess, float), np.arange(n), 0)
        for a in ("feature", "left", "right"):
            setattr(self, a, np.array(getattr(self, a), dtype=np.int32))
        self.threshold, self.value = np.array(self.threshold, float), np.array(self.value, float)
        self.risk, self.weight = np.array(self.risk, float), np.array(self.weight, float)
        s = self.importance.sum()
        self.importance = self.importance / s if s > 0 else self.importance
        return self

    def _leaf_value(self, y, w, hess, idx):
        if self.task == "classify":
            counts = np.bincount(y[idx], w[idx], minlength=self.K) + self.smooth
            return counts / max(counts.sum(), 1e-12)
        h = hess[idx].sum() if hess is not None else len(idx)
        return np.array([y[idx].sum() / max(h, 1e-12)])

    def _node_risk(self, y, w, idx):
        """Resubstitution risk used by pruning: weighted misclassification (classify) or squared error (regress)."""
        if self.task == "classify":
            counts = np.bincount(y[idx], w[idx], minlength=self.K)
            return counts.sum() - counts.max()
        r = y[idx]
        return float(((r - r.mean()) ** 2).sum())

    def _new(self, value, risk, weight):
        self.feature.append(-1); self.threshold.append(np.nan); self.left.append(-1); self.right.append(-1)
        self.value.append(value); self.risk.append(risk); self.weight.append(weight)
        return len(self.feature) - 1

    def _grow(self, B, y, w, hess, idx, depth):
        node = self._new(self._leaf_value(y, w, hess, idx), self._node_risk(y, w, idx), w[idx].sum())
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return node
        best = self._best_split(B, y, w, idx)
        if best is None:
            return node
        gain, j, k = best
        self.importance[j] += gain
        mask = B[idx, j] <= k
        self.feature[node], self.threshold[node] = j, self.binner.edges[j][k]
        self.left[node] = self._grow(B, y, w, hess, idx[mask], depth + 1)
        self.right[node] = self._grow(B, y, w, hess, idx[~mask], depth + 1)
        return node

    def _best_split(self, B, y, w, idx):
        feats = np.arange(self.n_features)
        if self.max_features is not None and self.max_features < self.n_features:
            feats = self.rng.choice(self.n_features, self.max_features, replace=False)
        best, bw = None, 1e-12
        for j in feats:
            nb = self.nbins[j]
            if nb < 2:
                continue
            b = B[idx, j].astype(np.int64)
            cnt = np.cumsum(np.bincount(b, minlength=nb))[:-1]
            ok = (cnt >= self.min_leaf) & (len(idx) - cnt >= self.min_leaf)
            if not ok.any():
                continue
            if self.task == "classify":
                hist = np.bincount(b * self.K + y[idx], w[idx], minlength=nb * self.K).reshape(nb, self.K)
                L = np.cumsum(hist, 0)[:-1]
                T = hist.sum(0)
                R = T - L
                nL, nR, nT = L.sum(1), R.sum(1), T.sum()
                with np.errstate(divide="ignore", invalid="ignore"):
                    gL = 1 - ((L / nL[:, None]) ** 2).sum(1)
                    gR = 1 - ((R / nR[:, None]) ** 2).sum(1)
                gT = 1 - ((T / nT) ** 2).sum()
                gain = nT * gT - (nL * gL + nR * gR)          # weighted Gini decrease
            else:
                s = np.cumsum(np.bincount(b, y[idx], minlength=nb))[:-1]
                S, n = y[idx].sum(), len(idx)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain = s ** 2 / cnt + (S - s) ** 2 / (n - cnt) - S ** 2 / n    # squared-error decrease
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > bw:
                best, bw = (float(gain[k]), int(j), k), gain[k]
        return best

    # ------------------------------------------------------------------ prediction
    def apply(self, X):
        """Leaf index reached by each row."""
        node = np.zeros(len(X), dtype=np.int32)
        while True:
            f = self.feature[node]
            active = f >= 0
            if not active.any():
                return node
            rows = np.where(active)[0]
            go_left = X[rows, f[rows]] <= self.threshold[node[rows]]
            node[rows] = np.where(go_left, self.left[node[rows]], self.right[node[rows]])

    def predict_value(self, X):
        return self.value[self.apply(X)]

    def predict_proba(self, X):
        return self.predict_value(X)

    def predict(self, X):
        return self.predict_value(X).argmax(1)

    def n_leaves(self):
        """Leaves reachable from the root (pruning leaves orphaned nodes in the arrays)."""
        return sum(1 for i in self._reachable() if self.feature[i] < 0)

    def depth(self):
        best, stack = 0, [(0, 0)]
        while stack:
            i, d = stack.pop()
            best = max(best, d)
            if self.feature[i] >= 0:
                stack += [(self.left[i], d + 1), (self.right[i], d + 1)]
        return best

    # ------------------------------------------------------------------ cost-complexity pruning
    def pruning_path(self):
        """Weakest-link pruning (Breiman et al.). Returns a list of (alpha, collapsed node set, leaves).
        At each step the internal node with the smallest g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1) is collapsed."""
        collapsed = set()
        path = [(0.0, frozenset(), self.n_leaves())]
        while True:
            stats = {}

            def walk(i):
                if self.feature[i] < 0 or i in collapsed:
                    return self.risk[i], 1
                rl, nl = walk(self.left[i])
                rr, nr = walk(self.right[i])
                stats[i] = (self.risk[i], rl + rr, nl + nr)
                return rl + rr, nl + nr
            walk(0)
            if not stats:
                break
            g = {i: (r - rs) / max(n - 1, 1) for i, (r, rs, n) in stats.items()}
            alpha = min(g.values())
            for i, gi in g.items():
                if gi <= alpha + 1e-12:
                    collapsed.add(i)
            leaves = self._leaves_after(collapsed)
            path.append((float(alpha), frozenset(collapsed), leaves))
            if 0 in collapsed:
                break
        return path

    def _leaves_after(self, collapsed):
        def count(i):
            if self.feature[i] < 0 or i in collapsed:
                return 1
            return count(self.left[i]) + count(self.right[i])
        return count(0)

    def pruned(self, collapsed):
        """A copy of the tree with the given internal nodes turned into leaves."""
        t = Tree(self.max_depth, self.min_leaf, task=self.task, n_classes=self.K, binner=self.binner)
        t.feature, t.threshold, t.left, t.right = self.feature.copy(), self.threshold.copy(), self.left.copy(), self.right.copy()
        t.value, t.risk, t.weight, t.importance = self.value.copy(), self.risk.copy(), self.weight.copy(), self.importance.copy()
        for i in collapsed:
            t.feature[i], t.left[i], t.right[i], t.threshold[i] = -1, -1, -1, np.nan
        return t

    # ------------------------------------------------------------------ saving
    def state(self, compact=False):
        """compact=True (two-class trees only) keeps just P(class 1) per node and rounds numbers, to keep forests small."""
        keep = self._reachable()
        remap = {old: new for new, old in enumerate(keep)}
        if compact:
            value = [round(float(np.atleast_1d(self.value[i])[-1]), 4) for i in keep]
            thr = [None if self.feature[i] < 0 else float(f"{self.threshold[i]:.6g}") for i in keep]
        else:
            value = [[round(float(v), 6) for v in np.atleast_1d(self.value[i])] for i in keep]
            thr = [None if self.feature[i] < 0 else float(self.threshold[i]) for i in keep]
        return {"task": self.task, "feature": [int(self.feature[i]) for i in keep], "threshold": thr,
                "left": [remap.get(int(self.left[i]), -1) for i in keep], "right": [remap.get(int(self.right[i]), -1) for i in keep],
                "value": value}

    def _reachable(self):
        out, stack = [], [0]
        while stack:
            i = stack.pop()
            out.append(i)
            if self.feature[i] >= 0:
                stack += [self.right[i], self.left[i]]
        return sorted(out)

    @classmethod
    def from_state(cls, s):
        t = cls(task=s["task"])
        t.feature = np.array(s["feature"], np.int32)
        t.threshold = np.array([np.nan if v is None else v for v in s["threshold"]], float)
        t.left, t.right = np.array(s["left"], np.int32), np.array(s["right"], np.int32)
        v = np.array(s["value"], float)
        t.value = np.column_stack([1 - v, v]) if v.ndim == 1 and s["task"] == "classify" else v.reshape(len(v), -1)
        t.K = t.value.shape[1]
        return t

    def path(self, x):
        """Node ids visited by one row, root first."""
        out, i = [0], 0
        while self.feature[i] >= 0:
            i = self.left[i] if x[self.feature[i]] <= self.threshold[i] else self.right[i]
            out.append(int(i))
        return out
