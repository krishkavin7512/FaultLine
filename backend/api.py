"""FastAPI backend for FaultLine. Start with `python run.py`; API docs at http://127.0.0.1:8005/docs"""
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, data
from .combine import ALL, LABEL, Ensemble

app = FastAPI(title="FaultLine API", version="1.0",
              description="Which houses will an earthquake bring down? Trees, forests, boosting, a GAM and model averaging, "
                          "written from scratch and trained on 760,000 houses surveyed after Nepal's 2015 earthquake.")


def _load(name):
    path = config.ARTIFACTS / name
    if not path.exists():
        raise RuntimeError(f"{path} is missing. Run `python -m backend.train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


REPORT = _load("report.json")
GRAPHS = _load("graphs.json")
MODEL = _load("model.json")
ENS = Ensemble.from_state(MODEL["ensemble"])
LEVELS = MODEL["levels"]
NAMES, SOURCE, KIND = MODEL["design"]["names"], MODEL["design"]["source"], MODEL["design"]["kind"]
J_SHAKE = SOURCE.index("pred_intensity")
CHECK_LIST = sorted(MODEL["checklist_nodes"], key=lambda n: n["id"])
HEAD = MODEL["headline"]
WALL_KEYS = {c.replace("has_superstructure_", ""): c for c in config.WALLS}
GRID = [round(6.0 + 0.125 * i, 3) for i in range(21)]


@app.get("/api/overview")
def overview():
    R = REPORT
    keep = ("trained_at", "data", "scores", "baselines", "no_shaking", "capacity", "cart", "stack", "family_bma", "crowd_members", "fragility", "wall_rates", "floor_rates",
            "upgrades", "districts", "shifts", "shift_totals", "grades", "examples", "headline", "labels", "kinds", "sources", "stumps", "ward_rank", "ward_fit")
    out = {k: R[k] for k in keep if k in R}
    out["bootstrap"] = R["bootstrap"]
    out["bma"] = {k: R["bma"][k] for k in ("n_models", "inclusion", "models_for_99", "rows", "top")}
    out["gam"] = {"lam": R["gam"]["lam"], "terms": [{k: t[k] for k in ("name", "source", "type", "effect", "edf") if k in t} for t in R["gam"]["terms"]]}
    out["importance"] = R["importance"]["permutation"]
    out["growth"] = {"gboost_best": R["growth"]["gboost"]["best_n"], "forest_auc": R["growth"]["forest"]["auc"][-1],
                     "single_tree": [min(R["growth"]["single_tree_auc"]), max(R["growth"]["single_tree_auc"])]}
    out["n_graphs"] = len(GRAPHS)
    return out


@app.get("/api/wards")
def wards():
    """Every surveyed ward: position, real and predicted share destroyed, and the prediction at each shaking shift."""
    return {"shifts": REPORT["shifts"], "totals": REPORT["shift_totals"], "wards": REPORT["wards"]}


@app.get("/api/options")
def options():
    """Allowed values for the house builder."""
    return {"choices": MODEL["choices"], "numeric": MODEL["numeric"], "walls": {k: config.NICE[c] for k, c in WALL_KEYS.items()},
            "district_shaking": MODEL["district_shaking"], "examples": REPORT["examples"], "labels": {k: LABEL[k] for k in ALL}, "level_nice": MODEL.get("level_nice", {})}


class House(BaseModel):
    floors: int = Field(2, ge=1, le=9, description="Storeys before the earthquake")
    age: float = Field(20, ge=0, le=120, description="Age in years")
    area: float = Field(350, ge=70, le=5000, description="Floor area (plinth) in square feet")
    height: float = Field(16, ge=6, le=99, description="Height in feet")
    shaking: float = Field(7.5, ge=5.5, le=9.5, description="Ground shaking intensity at the house")
    ground: str = Field("Flat", description="land_surface_condition")
    foundation: str = Field("Mud mortar-Stone/Brick", description="foundation_type")
    roof: str = Field("Bamboo/Timber-Light roof", description="roof_type")
    ground_floor: str = Field("Mud", description="ground_floor_type")
    upper_floors: str = Field("TImber/Bamboo-Mud", description="other_floor_type")
    position: str = Field("Not attached")
    plan: str = Field("Rectangular", description="plan_configuration")
    walls: list[str] = Field(["mud_mortar_stone"], description=f"Any of {list(WALL_KEYS)}")


FIELDS = {"ground": "land_surface_condition", "foundation": "foundation_type", "roof": "roof_type", "ground_floor": "ground_floor_type",
          "upper_floors": "other_floor_type", "position": "position", "plan": "plan_configuration"}


def raw_row(h: House):
    bad = [w for w in h.walls if w not in WALL_KEYS]
    if bad or not h.walls:
        raise HTTPException(400, f"walls must be a non-empty list from {list(WALL_KEYS)}; got {bad or 'nothing'}")
    row = {"count_floors_pre_eq": h.floors, "age_building": h.age, "plinth_area_sq_ft": h.area, "height_ft_pre_eq": h.height,
           "pred_intensity": h.shaking, "age_unknown": 0}
    for f, c in FIELDS.items():
        v = getattr(h, f)
        if v not in MODEL["choices"][c]:
            raise HTTPException(400, f"{f} must be one of {MODEL['choices'][c]}")
        row[c] = v
    for c in config.WALLS:
        row[c] = int(c.replace("has_superstructure_", "") in h.walls)
    return row


def upgrades_for(row):
    """The same house with one change each; only changes that apply to this house."""
    out = []
    mud = row["has_superstructure_mud_mortar_stone"] or row["has_superstructure_mud_mortar_brick"] or row["foundation_type"] == "Mud mortar-Stone/Brick"
    if mud:
        r = dict(row)
        for m, c in (("has_superstructure_mud_mortar_stone", "has_superstructure_cement_mortar_stone"), ("has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick")):
            if r[m]:
                r[m], r[c] = 0, 1
        if r["foundation_type"] == "Mud mortar-Stone/Brick":
            r["foundation_type"] = "Cement-Stone/Brick"
        out.append(("cement", "Use cement mortar instead of mud", r))
    if row["roof_type"] == "Bamboo/Timber-Heavy roof":
        out.append(("roof", "Swap the heavy roof for a light one", {**row, "roof_type": "Bamboo/Timber-Light roof"}))
    if row["count_floors_pre_eq"] >= 3:
        # Only from three storeys down to two, as in the training report. Single-storey stone houses in this survey
        # fell more often than two-storey ones with the same facts, which reflects who built them, not the storey.
        per = row["height_ft_pre_eq"] / row["count_floors_pre_eq"]
        out.append(("floor", "Build one storey fewer", {**row, "count_floors_pre_eq": row["count_floors_pre_eq"] - 1, "height_ft_pre_eq": max(6, round(row["height_ft_pre_eq"] - per))}))
    if not row["has_superstructure_rc_engineered"]:
        r = {**row, **{c: 0 for c in config.WALLS}, "has_superstructure_rc_engineered": 1, "foundation_type": "RC", "ground_floor_type": "RC", "roof_type": "RCC/RB/RBC"}
        if r["count_floors_pre_eq"] > 1:
            r["other_floor_type"] = "RCC/RB/RBC"
        out.append(("rc", "Rebuild as an engineered concrete frame", r))
    return out


def checklist_path(x):
    """Walk the saved checklist tree. Saving renumbers the tree's nodes in the same sorted order the node
    descriptions were written in, so position i in CHECK_LIST describes node i of the loaded tree."""
    tree = ENS.m["checklist"]
    ids = tree.path(x)
    steps = [CHECK_LIST[a]["yes"] if b == tree.left[a] else CHECK_LIST[a]["no"] for a, b in zip(ids, ids[1:])]
    leaf = CHECK_LIST[ids[-1]]
    return {"steps": steps, "p": leaf["p"], "share": leaf["share"]}


@app.post("/api/predict")
def predict(h: House):
    """Every model's collapse probability for one house, why (GAM breakdown and checklist path), what an upgrade would change,
    and how the risk grows with shaking."""
    row = raw_row(h)
    ups = upgrades_for(row)
    rows = [row] + [r for _, _, r in ups] + [{**row, "pred_intensity": s} for s in GRID]
    X = data.design(pd.DataFrame(rows), LEVELS).X
    P = ENS.predict(X)
    p = {k: float(P[k][0]) for k in ALL}
    k0 = 1 + len(ups)
    gam = ENS.m["gam"]
    contrib = gam.contributions(X[:1])[0]
    by = {}
    for t, c in zip(gam.terms, contrib):
        src = SOURCE[t["col"]]
        by[src] = by.get(src, 0.0) + float(c)
    why = sorted(({"fact": config.NICE[s], "log_odds": v} for s, v in by.items()), key=lambda r: -abs(r["log_odds"]))
    return {"p": p, "headline": HEAD, "label": LABEL[HEAD], "spread": [min(p[k] for k in ALL), max(p[k] for k in ALL)],
            "why": {"base": float(gam.alpha), "terms": why, "p_gam": p["gam"]}, "checklist": checklist_path(X[0]),
            "upgrades": [{"key": k, "label": lab, "p": float(P[HEAD][i + 1]), "drop": p[HEAD] - float(P[HEAD][i + 1])} for i, (k, lab, _) in enumerate(ups)],
            "curve": {"shaking": GRID, "p": P[HEAD][k0:].round(4).tolist()}}


@app.get("/api/graphs")
def graphs():
    return {"graphs": GRAPHS}


@app.get("/api/graphs/{gid}")
def graph(gid: str):
    for g in GRAPHS:
        if g["id"] == gid:
            return g
    raise HTTPException(404, "no such graph")


@app.middleware("http")
async def revalidate_assets(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/assets") or request.url.path in ("/", "/graphs"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/assets", StaticFiles(directory=config.FRONTEND / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND / "index.html")


@app.get("/graphs", include_in_schema=False)
def graphs_page():
    return FileResponse(config.FRONTEND / "graphs.html")
