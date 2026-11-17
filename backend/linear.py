"""Logistic regression fitted by iteratively reweighted least squares (Newton's method).

Used as the plain baseline, inside Bayesian model averaging (thousands of small fits) and as the stacking combiner.
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


class Logistic:
    def __init__(self, ridge=1e-4, n_iter=50, tol=1e-8, standardise=True):
        self.ridge, self.n_iter, self.tol, self.standardise = ridge, n_iter, tol, standardise

    def _prep(self, X):
        return (X - self.mu) / self.sd

    def fit(self, X, y, w=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        if self.standardise:
            self.mu, self.sd = X.mean(0), X.std(0)
            self.sd[self.sd == 0] = 1
        else:
            self.mu, self.sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        Z = np.c_[np.ones(len(X)), self._prep(X)]
        beta = np.zeros(Z.shape[1])
        beta[0] = np.log((y @ w + .5) / ((1 - y) @ w + .5))
        P = self.ridge * np.eye(Z.shape[1])
        P[0, 0] = 0
        for _ in range(self.n_iter):
            p = sigmoid(Z @ beta)
            s = w * p * (1 - p)
            H = Z.T @ (Z * s[:, None]) + P
            g = Z.T @ (w * (y - p)) - P @ beta
            step = np.linalg.solve(H, g)
            beta += step
            if np.abs(step).max() < self.tol:
                break
        self.beta, self.hessian = beta, H
        p = sigmoid(Z @ beta)
        self.loglik = float((w * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))).sum())
        return self

    def decision(self, X):
        return np.c_[np.ones(len(X)), self._prep(np.asarray(X, float))] @ self.beta

    def predict_proba(self, X):
        return sigmoid(self.decision(X))

    def n_params(self):
        return len(self.beta)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_state(cls, s):
        m = cls()
        m.mu, m.sd, m.beta = np.array(s["mu"]), np.array(s["sd"]), np.array(s["beta"])
        return m
