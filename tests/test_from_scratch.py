"""Each from-scratch model, checked against scikit-learn or against a property it must have."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from backend import config, data
from backend.bma import family_weights, subset_bma
from backend.ensemble import AdaBoost, Forest, GradientBoosting, Stacker
from backend.gam import GAM
from backend.linear import Logistic, sigmoid
from backend.metrics import at_capacity, auc, bootstrap_auc_gain, log_loss
from backend.train import merged_rule, within_ward
from backend.tree import Tree


def toy(n=3000, seed=0):
    """Collapse depends on shaking (a wave a straight line cannot follow), a fragile-wall flag, storeys, and an interaction."""
    r = np.random.default_rng(seed)
    shake = r.uniform(6, 8.5, n)
    wall = r.integers(0, 2, n).astype(float)
    floors = r.integers(1, 5, n).astype(float)
    noise = r.normal(size=n)
    logit = -1 + 1.8 * np.sin(2.2 * (shake - 6)) + 1.2 * wall + 0.3 * floors + 0.8 * wall * (shake > 7.6) + 0 * noise
    y = (r.random(n) < sigmoid(logit)).astype(int)
    return np.column_stack([shake, wall, floors, noise]), y


X, Y = toy()
Xt, Yt = toy(seed=1)


def test_auc_matches_sklearn_with_ties():
    r = np.random.default_rng(3)
    y = r.integers(0, 2, 5000)
    s = np.round(r.random(5000) + 0.4 * y, 1)          # heavy ties
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_capacity_counts():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    p = np.arange(10)[::-1] / 10
    c = at_capacity(y, p, 0.3)
    assert c["flagged"] == 3 and c["real"] == 2 and c["recall"] == pytest.approx(2 / 3)


def test_tree_first_split_matches_sklearn():
    Xs = np.round(X, 1)                                  # few unique values, so every possible cut is searched
    ours = Tree(1, 1).fit(Xs, Y)
    sk = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xs, Y)
    assert ours.feature[0] == sk.tree_.feature[0]
    left_ours = Xs[:, ours.feature[0]] <= ours.threshold[0]
    left_sk = Xs[:, sk.tree_.feature[0]] <= sk.tree_.threshold[0]
    assert np.array_equal(left_ours, left_sk)


def test_pruning_path_is_nested():
    t = Tree(10, 5, smooth=1).fit(X, Y)
    path = t.pruning_path()
    alphas = [a for a, _, _ in path]
    leaves = [l for _, _, l in path]
    assert alphas == sorted(alphas)
    assert leaves == sorted(leaves, reverse=True) and leaves[-1] == 1
    assert all(prev <= nxt for (_, prev, _), (_, nxt, _) in zip(path, path[1:]))   # each step only collapses more nodes
    small = t.pruned(path[len(path) // 2][1])
    assert small.n_leaves() == path[len(path) // 2][2]


def test_tree_save_and_load_gives_same_predictions():
    t = Tree(8, 10, smooth=1).fit(X, Y)
    for compact in (False, True):
        back = Tree.from_state(t.state(compact=compact))
        assert np.allclose(back.predict_proba(Xt)[:, 1], t.predict_proba(Xt)[:, 1], atol=1e-4)


def test_forest_beats_one_deep_tree():
    single = Tree(30, 3).fit(X, Y).predict_proba(Xt)[:, 1]
    rf = Forest(60, "sqrt", 3, seed=1).fit(X, Y)
    assert auc(Yt, rf.predict_proba(Xt)) > auc(Yt, single) + 0.02
    assert len(rf.oob_curve) == 60 and rf.oob_curve[-1] < rf.oob_curve[0]
    back = Forest.from_state(rf.state())
    assert np.allclose(back.predict_proba(Xt), rf.predict_proba(Xt), atol=1e-3)


def test_multiclass_forest_probabilities_sum_to_one():
    g = np.clip((X[:, 0] - 6) * 2, 0, 4).astype(int)
    f = Forest(10, "sqrt", 5, n_classes=5).fit(X, g)
    P = f.predict_proba_all(Xt)
    assert P.shape == (len(Xt), 5) and np.allclose(P.sum(1), 1)


def test_adaboost_errors_and_gain():
    ab = AdaBoost(120, 1).fit(X, Y)
    assert all(e < 0.5 for e in ab.errors)
    assert auc(Yt, ab.predict_proba(Xt)) > auc(Yt, ab.predict_proba(Xt, 1)) + 0.05
    assert np.allclose(AdaBoost.from_state(ab.state()).predict_proba(Xt), ab.predict_proba(Xt), atol=1e-6)


def test_gradient_boosting_close_to_sklearn():
    gb = GradientBoosting(200, 3, 0.1, 1.0, 1).fit(X, Y, Xt, Yt)
    sk = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0).fit(X, Y)
    ours, theirs = auc(Yt, gb.predict_proba(Xt, 200)), roc_auc_score(Yt, sk.predict_proba(Xt)[:, 1])
    assert abs(ours - theirs) < 0.02
    assert gb.train_curve == sorted(gb.train_curve, reverse=True)
    back = GradientBoosting.from_state(gb.state())
    assert np.allclose(back.predict_proba(Xt), gb.predict_proba(Xt), atol=1e-6)


def test_logistic_matches_sklearn():
    ours = Logistic(ridge=1e-8).fit(X, Y)
    sk = LogisticRegression(C=1e8, max_iter=20000, tol=1e-12).fit(X, Y)
    assert np.allclose(ours.predict_proba(Xt), sk.predict_proba(Xt)[:, 1], atol=2e-4)


def test_gam_recovers_a_curve_the_logit_cannot():
    kinds = ["num", "bin", "num", "num"]
    g = GAM(lam=3).fit(X, Y, kinds)
    lg = Logistic(ridge=1e-6).fit(X, Y)
    assert log_loss(Yt, g.predict_proba(Xt)) < log_loss(Yt, lg.predict_proba(Xt)) - 0.01
    grid = np.linspace(6.2, 8.3, 40)
    truth = 1.8 * np.sin(2.2 * (grid - 6))
    assert np.corrcoef(g.curve(0, grid), truth)[0, 1] > 0.95
    assert np.allclose(GAM.from_state(g.state()).predict_proba(Xt), g.predict_proba(Xt))
    assert np.allclose(g.alpha + g.contributions(Xt).sum(1), g.decision(Xt))


def test_stacker_and_family_weights():
    P = np.column_stack([Forest(20, "sqrt", 5).fit(X, Y).predict_proba(Xt), Logistic().fit(X, Y).predict_proba(Xt)])
    st = Stacker().fit(P, Yt)
    assert np.all(np.isfinite(st.weights)) and np.allclose(Stacker.from_state(st.state()).predict_proba(P), st.predict_proba(P))
    fw = family_weights({"a": {"total": -100.0, "n": 1000}, "b": {"total": -110.0, "n": 1000}})
    assert sum(fw["weights"]) == pytest.approx(1) and fw["weights"][0] > 0.99
    assert fw["by_houses"][1][0] < fw["by_houses"][1000][0]           # weights collapse as houses are added


def test_subset_bma_finds_the_signal():
    groups = [("shaking", [0]), ("wall", [1]), ("floors", [2]), ("noise", [3])]
    r = subset_bma(X, Y, Xt, groups)
    assert r["n_models"] == 16
    assert r["inclusion"]["shaking"] > 0.99 and r["inclusion"]["wall"] > 0.99
    assert r["inclusion"]["noise"] < 0.5
    assert len(r["averaged"]) == len(Xt) and np.all((r["averaged"] > 0) & (r["averaged"] < 1))


def test_bootstrap_interval_contains_mean():
    p_a = Forest(20, "sqrt", 5).fit(X, Y).predict_proba(Xt)
    p_b = np.random.default_rng(0).random(len(Xt))
    r = bootstrap_auc_gain(Yt, p_a, p_b, reps=100)
    assert r["low"] <= r["mean"] <= r["high"] and r["low"] > 0


def test_one_house_encodes_like_the_training_data():
    rows = []
    for i, (wall, roof) in enumerate([("has_superstructure_mud_mortar_stone", "Bamboo/Timber-Light roof"), ("has_superstructure_rc_engineered", "RCC/RB/RBC"),
                                      ("has_superstructure_adobe_mud", "Bamboo/Timber-Heavy roof")] * 600):
        r = {c: 0 for c in config.WALLS}
        r.update({wall: 1, "count_floors_pre_eq": 2, "age_building": 10 + i % 50, "plinth_area_sq_ft": 300 + i, "height_ft_pre_eq": 15, "pred_intensity": 7 + (i % 10) / 10,
                  "age_unknown": 0, "land_surface_condition": "Flat", "foundation_type": "Mud mortar-Stone/Brick", "roof_type": roof, "ground_floor_type": "Mud",
                  "other_floor_type": "TImber/Bamboo-Mud", "position": "Not attached", "plan_configuration": "Rectangular" if i % 900 else "U-shape"})
        rows.append(r)
    df = pd.DataFrame(rows)
    levels = data.fit_levels(df, np.arange(len(df)))
    full = data.design(df, levels)
    one = data.design(df.iloc[[5]], levels)
    assert one.X.shape[1] == full.X.shape[1] and np.allclose(one.X[0], full.X[5])
    assert "Roof: reinforced concrete" in full.names


def test_rules_merge_repeated_cuts():
    D = data.Design(np.zeros((1, 3)), ["Ground shaking (intensity)", "Walls: stone in mud mortar", "Floors"], ["pred_intensity", "has_superstructure_mud_mortar_stone", "count_floors_pre_eq"], ["num", "bin", "num"])
    out = merged_rule(D, [(0, 7.62, "right"), (1, 0.0, "right"), (0, 7.91, "left"), (0, 7.78, "left"), (2, 2.0, "right"), (2, 3.0, "left")])
    assert out == ["Shaking 7.62–7.78", "Walls: stone in mud mortar", "Storeys 3"]


def test_within_ward_ranking():
    r = np.random.default_rng(0)
    wid = np.repeat(np.arange(50), 100)
    y = r.integers(0, 2, len(wid))
    assert abs(within_ward(y, r.random(len(wid)), wid)["auc"] - 0.5) < 0.03
    assert within_ward(y, y + r.random(len(wid)) * 0.1, wid)["auc"] == pytest.approx(1.0)
