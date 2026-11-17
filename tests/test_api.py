"""The running app: every route, on the real trained artifacts."""
import pytest
from fastapi.testclient import TestClient

from backend.api import REPORT, app

client = TestClient(app)
STONE = {"floors": 3, "age": 30, "area": 350, "height": 24, "shaking": 7.8, "ground": "Flat", "foundation": "Mud mortar-Stone/Brick",
         "roof": "Bamboo/Timber-Heavy roof", "ground_floor": "Mud", "upper_floors": "TImber/Bamboo-Mud", "position": "Not attached",
         "plan": "Rectangular", "walls": ["mud_mortar_stone"]}


def test_pages_load():
    assert client.get("/").status_code == 200
    assert client.get("/graphs").status_code == 200
    assert client.get("/assets/js/main.js").status_code == 200


def test_overview():
    r = client.get("/api/overview").json()
    assert r["n_graphs"] >= 40
    assert {"crowd", "stack", "cart", "gboost", "gam", "bma"} <= set(r["scores"])
    assert len(r["data"]["unseen_districts"]) == 3
    for s in r["scores"].values():
        assert 0.5 < s["unseen"]["auc"] < 1 and 0.5 < s["seen"]["auc"] < 1


def test_wards():
    r = client.get("/api/wards").json()
    assert len(r["wards"]) > 900 and len(r["shifts"]) == len(r["totals"])
    assert r["totals"] == sorted(r["totals"])                      # stronger shaking never forecasts fewer losses
    assert all(0 <= w["actual"] <= 1 and 0 <= w["model"] <= 1 for w in r["wards"])


def test_options():
    r = client.get("/api/options").json()
    assert "Mud mortar-Stone/Brick" in r["choices"]["foundation_type"] and len(r["walls"]) == 11 and r["examples"]


def test_predict_stone_house():
    r = client.post("/api/predict", json=STONE).json()
    assert all(0 <= v <= 1 for v in r["p"].values()) and r["headline"] in r["p"]
    assert r["checklist"]["steps"] and r["why"]["terms"]
    assert {u["key"] for u in r["upgrades"]} == {"cement", "roof", "floor", "rc"}
    assert all(u["p"] <= r["p"][r["headline"]] + 0.02 for u in r["upgrades"])
    assert r["curve"]["p"][-1] > r["curve"]["p"][0]


def test_engineered_frame_is_safer_than_stone():
    rc = {**STONE, "walls": ["rc_engineered"], "foundation": "RC", "roof": "RCC/RB/RBC", "ground_floor": "RC", "upper_floors": "RCC/RB/RBC"}
    p_stone = client.post("/api/predict", json=STONE).json()["p"]
    p_rc = client.post("/api/predict", json=rc).json()["p"]
    assert p_rc["crowd"] < p_stone["crowd"] / 2
    assert not any(u["key"] == "rc" for u in client.post("/api/predict", json=rc).json()["upgrades"])


@pytest.mark.parametrize("bad", [{"walls": ["gold"]}, {"walls": []}, {"roof": "Thatch"}, {"floors": 0}, {"shaking": 12}])
def test_predict_rejects_bad_input(bad):
    assert client.post("/api/predict", json={**STONE, **bad}).status_code in (400, 422)


def test_examples_score_like_the_report():
    for e in REPORT["examples"]:
        h = e["house"]
        body = {"floors": h["count_floors_pre_eq"], "age": h["age_building"], "area": h["plinth_area_sq_ft"], "height": h["height_ft_pre_eq"], "shaking": h["pred_intensity"],
                "ground": h["land_surface_condition"], "foundation": h["foundation_type"], "roof": h["roof_type"], "ground_floor": h["ground_floor_type"],
                "upper_floors": h["other_floor_type"], "position": h["position"], "plan": h["plan_configuration"], "walls": [w.replace("has_superstructure_", "") for w in h["walls"]]}
        r = client.post("/api/predict", json=body).json()
        assert r["p"]["crowd"] == pytest.approx(e["p"]["crowd"], abs=0.01)


def test_graphs():
    gs = client.get("/api/graphs").json()["graphs"]
    assert len({g["id"] for g in gs}) == len(gs)
    for g in gs:
        assert g["insight"] and all(g["explain"][k] for k in ("what", "how", "read", "where"))
    assert client.get(f"/api/graphs/{gs[0]['id']}").status_code == 200
    assert client.get("/api/graphs/nope").status_code == 404
