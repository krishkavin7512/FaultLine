"""Load the building survey, attach each ward's ground shaking, build a numeric matrix and split once.

Only facts known before the earthquake are used: floors, age, area, height, ground, foundation, roof, floors, walls,
position and plan, plus the shaking estimated for the building's ward. The damage grade is the answer, used for
training and scoring only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


def load():
    b = pd.read_csv(config.BUILDINGS)
    w = pd.read_csv(config.WARDS)
    b = b.merge(w[["ward_id", "pred_intensity"]], on="ward_id", how="inner")    # drops the 2,096 buildings with no ward estimate
    b["age_unknown"] = (b["age_building"] == config.AGE_UNKNOWN).astype(int)
    known = b.loc[b["age_unknown"] == 0, "age_building"]
    b.loc[b["age_unknown"] == 1, "age_building"] = int(known.median())
    b["age_building"] = b["age_building"].clip(upper=config.AGE_CAP)
    for c in config.CATEGORICAL:
        b[c] = b[c].fillna("Not recorded")
    b["collapse"] = (b["damage_grade"] == 5).astype(int)
    b["wall_type"] = wall_type(b)
    return b.reset_index(drop=True), w


def wall_type(b):
    out = np.full(len(b), "Other", dtype=object)
    done = np.zeros(len(b), bool)
    for name, col in config.WALL_TYPES:
        hit = (b[col].to_numpy() == 1) & ~done
        out[hit] = name
        done |= hit
    return out


def split(b, seed=config.SEED):
    """Seen districts: disjoint random samples for train, validation and test. Unseen districts: every building."""
    rng = np.random.default_rng(seed)
    unseen = b["district_id"].isin(config.UNSEEN_DISTRICTS).to_numpy()
    pool = rng.permutation(np.where(~unseen)[0])
    a, c = config.N_TRAIN, config.N_TRAIN + config.N_VAL
    return {"train": np.sort(pool[:a]), "val": np.sort(pool[a:c]), "test_seen": np.sort(pool[c:c + config.N_TEST_SEEN]),
            "test_unseen": np.where(unseen)[0]}


@dataclass
class Design:
    X: np.ndarray
    names: list
    source: list
    kind: list
    levels: dict = field(default_factory=dict)

    def columns_of(self, var):
        return [i for i, s in enumerate(self.source) if s == var]


def fit_levels(b, rows):
    """For each category: the levels common enough to keep (most common first, used as the reference), and whether
    rarer levels exist and are pooled into one "other" column. Fixed once, so any single house encodes the same way."""
    out = {}
    for c in config.CATEGORICAL:
        counts = b.iloc[rows][c].value_counts()
        kept = list(counts[counts >= config.MIN_LEVEL].index)
        out[c] = {"kept": kept, "other": bool((counts < config.MIN_LEVEL).any())}
    return out


def design(b, levels, with_shaking=True):
    blocks, names, source, kind = [], [], [], []
    numeric = config.NUMERIC if with_shaking else [c for c in config.NUMERIC if c != "pred_intensity"]
    for c in numeric:
        blocks.append(b[c].to_numpy(float)); names.append(config.NICE[c]); source.append(c); kind.append("num")
    for c in ["age_unknown"] + config.WALLS:
        blocks.append(b[c].to_numpy(float)); names.append(config.NICE[c]); source.append(c); kind.append("bin")
    for c in config.CATEGORICAL:
        v = b[c].to_numpy()
        kept = levels[c]["kept"]
        for lev in kept[1:]:
            blocks.append((v == lev).astype(float)); names.append(f"{config.NICE[c]}: {config.LEVEL_NICE.get(lev, lev)}"); source.append(c); kind.append("cat")
        if levels[c]["other"]:
            blocks.append((~np.isin(v, kept)).astype(float)); names.append(f"{config.NICE[c]}: other"); source.append(c); kind.append("cat")
    return Design(np.column_stack(blocks), names, source, kind, levels)


def row_frame(house, levels):
    """One house described in raw survey terms (from the what-if panel) as a one-row frame."""
    row = {c: house.get(c) for c in config.NUMERIC + config.CATEGORICAL + config.WALLS}
    row["age_unknown"] = 0
    for c in config.WALLS:
        row[c] = int(bool(row[c]))
    return pd.DataFrame([row])
