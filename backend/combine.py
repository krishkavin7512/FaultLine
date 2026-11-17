"""Every fitted model together. One call gives each model's collapse probability plus the two combinations:
the crowd (a plain average of five different kinds of model) and the stack (a learned weighting of all seven)."""
import numpy as np

from .ensemble import AdaBoost, Forest, GradientBoosting, Stacker
from .gam import GAM
from .linear import Logistic
from .tree import Tree

BASE = ["cart", "bagging", "forest", "adaboost", "gboost", "gam", "logit"]
CROWD = ["forest", "adaboost", "gboost", "gam", "logit"]
ALL = ["checklist"] + BASE + ["crowd", "stack"]
LABEL = {"checklist": "Inspector's checklist", "cart": "CART", "bagging": "Bagging", "forest": "Random forest", "adaboost": "AdaBoost",
         "gboost": "Gradient boosting", "gam": "GAM", "logit": "Logistic regression", "crowd": "Crowd average", "stack": "Stacking",
         "bma": "BMA (4,096 models)", "shaking": "Shaking only", "rule": "Mud walls first", "random": "Random"}
KIND = {"checklist": "single", "cart": "single", "logit": "single", "gam": "single", "bagging": "ensemble", "forest": "ensemble",
        "adaboost": "ensemble", "gboost": "ensemble", "crowd": "combined", "stack": "combined", "bma": "combined"}
LOADERS = {"checklist": Tree.from_state, "cart": Tree.from_state, "bagging": Forest.from_state, "forest": Forest.from_state,
           "adaboost": AdaBoost.from_state, "gboost": GradientBoosting.from_state, "gam": GAM.from_state, "logit": Logistic.from_state}


class Ensemble:
    def __init__(self, models, stacker):
        self.m, self.stacker = models, stacker

    def base(self, X):
        m = self.m
        return {"checklist": m["checklist"].predict_proba(X)[:, 1], "cart": m["cart"].predict_proba(X)[:, 1],
                "bagging": m["bagging"].predict_proba(X), "forest": m["forest"].predict_proba(X),
                "adaboost": m["adaboost"].predict_proba(X), "gboost": m["gboost"].predict_proba(X),
                "gam": m["gam"].predict_proba(X), "logit": m["logit"].predict_proba(X)}

    def predict(self, X, chunk=200_000):
        if len(X) > chunk:
            parts = [self.predict(X[i:i + chunk], chunk) for i in range(0, len(X), chunk)]
            return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        P = self.base(X)
        P["crowd"] = np.mean([P[k] for k in CROWD], 0)
        P["stack"] = self.stacker.predict_proba(np.column_stack([P[k] for k in BASE]))
        return P

    def state(self):
        return {"models": {k: v.state() for k, v in self.m.items()}, "stacker": self.stacker.state()}

    @classmethod
    def from_state(cls, s):
        return cls({k: LOADERS[k](v) for k, v in s["models"].items()}, Stacker.from_state(s["stacker"]))
