"""Paths, fixed settings, and the plain names used for the Nepal 2015 building survey."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"
BUILDINGS = DATA / "buildings.csv.gz"
WARDS = DATA / "wards.csv"

SEED = 7
# Three whole districts are kept out of training to test the model somewhere it has never seen.
# Gorkha held the epicentre of the 25 April 2015 earthquake.
UNSEEN_DISTRICTS = {36: "Gorkha", 30: "Dhading", 12: "Okhaldhunga"}
N_TRAIN = 60_000          # buildings the models learn from (sampled from the other 8 districts)
N_VAL = 20_000            # tuning, stacking and model averaging
N_TEST_SEEN = 100_000     # random unseen buildings from the 8 training districts
N_BMA = 15_000            # subsample for fitting all 4,096 explanation models
CAPACITY = 0.20           # default share of houses a programme can strengthen or inspect first

# Model sizes (kept small enough that every fitted model ships inside the site)
CHECKLIST_LEAVES = 10     # the inspector's checklist is the largest pruned tree with at most this many leaves
N_BAG, N_FOREST, N_FOREST5, FOREST_LEAF = 40, 120, 60, 60
N_ADA, N_GB, GB_RATE = 400, 800, 0.1
N_BOOT, N_BOOT_ROWS = 300, 50_000
MIN_CELL = 150            # fewest houses behind one point on an observed fragility curve
WARD_SAMPLE = 100         # houses per ward re-scored for the shaking slider
HEADLINE = "crowd"        # the combined model behind the map, district totals and house builder

# The 12 groups of facts that subset Bayesian model averaging switches on and off (2^12 = 4,096 models)
BMA_GROUPS = [
    ("Ground shaking", ["pred_intensity"]),
    ("Floors and height", ["count_floors_pre_eq", "height_ft_pre_eq"]),
    ("Age", ["age_building", "age_unknown"]),
    ("Floor area", ["plinth_area_sq_ft"]),
    ("Wall material", ["has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone", "has_superstructure_stone_flag",
                       "has_superstructure_cement_mortar_stone", "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
                       "has_superstructure_timber", "has_superstructure_bamboo", "has_superstructure_rc_non_engineered",
                       "has_superstructure_rc_engineered", "has_superstructure_other"]),
    ("Ground slope", ["land_surface_condition"]),
    ("Foundation", ["foundation_type"]),
    ("Roof", ["roof_type"]),
    ("Ground floor", ["ground_floor_type"]),
    ("Upper floors", ["other_floor_type"]),
    ("Position", ["position"]),
    ("Plan shape", ["plan_configuration"]),
]

NUMERIC = ["count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq", "pred_intensity"]
CATEGORICAL = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration"]
WALLS = ["has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone", "has_superstructure_stone_flag",
         "has_superstructure_cement_mortar_stone", "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
         "has_superstructure_timber", "has_superstructure_bamboo", "has_superstructure_rc_non_engineered",
         "has_superstructure_rc_engineered", "has_superstructure_other"]
MIN_LEVEL = 500           # category levels rarer than this are pooled into "Other"
AGE_UNKNOWN = 999         # the survey's code for an unrecorded age
AGE_CAP = 120

NICE = {
    "count_floors_pre_eq": "Floors", "age_building": "Age (years)", "plinth_area_sq_ft": "Floor area (sq ft)",
    "height_ft_pre_eq": "Height (ft)", "pred_intensity": "Ground shaking (intensity)", "age_unknown": "Age not recorded",
    "land_surface_condition": "Ground", "foundation_type": "Foundation", "roof_type": "Roof", "ground_floor_type": "Ground floor",
    "other_floor_type": "Upper floors", "position": "Position", "plan_configuration": "Plan shape",
    "has_superstructure_adobe_mud": "Walls: adobe / mud", "has_superstructure_mud_mortar_stone": "Walls: stone in mud mortar",
    "has_superstructure_stone_flag": "Walls: stone slabs", "has_superstructure_cement_mortar_stone": "Walls: stone in cement mortar",
    "has_superstructure_mud_mortar_brick": "Walls: brick in mud mortar", "has_superstructure_cement_mortar_brick": "Walls: brick in cement mortar",
    "has_superstructure_timber": "Walls: timber", "has_superstructure_bamboo": "Walls: bamboo",
    "has_superstructure_rc_non_engineered": "Frame: RC, non-engineered", "has_superstructure_rc_engineered": "Frame: RC, engineered",
    "has_superstructure_other": "Walls: other",
}

# Plain names for the survey's category codes (the raw codes stay in the data and the API)
LEVEL_NICE = {
    "RCC/RB/RBC": "reinforced concrete", "Bamboo/Timber-Light roof": "light bamboo/timber", "Bamboo/Timber-Heavy roof": "heavy bamboo/timber",
    "Mud mortar-Stone/Brick": "stone/brick in mud", "Cement-Stone/Brick": "stone/brick in cement", "Bamboo/Timber": "bamboo/timber", "RC": "reinforced concrete",
    "TImber/Bamboo-Mud": "timber/bamboo with mud", "Timber-Planck": "timber planks", "Not applicable": "none (single storey)", "Brick/Stone": "brick/stone",
    "Mud": "mud", "Timber": "timber", "Other": "other", "Flat": "flat", "Moderate slope": "moderate slope", "Steep slope": "steep slope",
    "Not attached": "detached", "Attached-1 side": "attached on 1 side", "Attached-2 side": "attached on 2 sides", "Attached-3 side": "attached on 3 sides",
    "Rectangular": "rectangular", "Square": "square", "L-shape": "L-shape", "T-shape": "T-shape", "U-shape": "U-shape", "E-shape": "E-shape", "H-shape": "H-shape",
    "Multi-projected": "multi-projected", "Others": "other", "Building with Central Courtyard": "central courtyard",
}
SHORT = {"pred_intensity": "Shaking", "count_floors_pre_eq": "Storeys", "age_building": "Age", "plinth_area_sq_ft": "Floor area", "height_ft_pre_eq": "Height"}

# One main construction type per house, strongest first, for the fragility curves
WALL_TYPES = [
    ("RC frame, engineered", "has_superstructure_rc_engineered"),
    ("RC frame, non-engineered", "has_superstructure_rc_non_engineered"),
    ("Brick in cement mortar", "has_superstructure_cement_mortar_brick"),
    ("Stone in cement mortar", "has_superstructure_cement_mortar_stone"),
    ("Brick in mud mortar", "has_superstructure_mud_mortar_brick"),
    ("Stone in mud mortar", "has_superstructure_mud_mortar_stone"),
    ("Adobe / mud", "has_superstructure_adobe_mud"),
    ("Stone slabs", "has_superstructure_stone_flag"),
    ("Timber", "has_superstructure_timber"),
    ("Bamboo", "has_superstructure_bamboo"),
]
FRAGILITY_TYPES = ["Stone in mud mortar", "Adobe / mud", "Brick in mud mortar", "Brick in cement mortar", "RC frame, non-engineered", "RC frame, engineered"]

GRADES = {1: "Grade 1: negligible to slight", 2: "Grade 2: moderate", 3: "Grade 3: substantial to heavy", 4: "Grade 4: very heavy", 5: "Grade 5: destroyed"}

SOURCES = {
    "data": "https://www.kaggle.com/datasets/sirorororo/nepal-earthquake-ward-level-intensities-and-impact",
    "portal": "https://kathmandulivinglabs.org/our-work/earthquake-data-portal",
    "paper": "https://zenodo.org/records/8027701",
}
