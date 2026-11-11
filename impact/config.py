"""Configuration for the independent Wildfire Impact Assessment module.

The module is intentionally standalone: it does not import Streamlit,
``app_core`` or ``dashboard.model_loader``.  The study-area constants below
mirror the model grid and are verified against ``dashboard.config`` and
``app_core`` by ``tests/test_impact_integrity.py``.

Only *read-only* inputs are referenced here.  Nothing in this package may
write to the model, training or pipeline directories.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMPACT_DATA_DIR = PROJECT_ROOT / "data" / "impact"
IMPACT_RAW_DIR = IMPACT_DATA_DIR / "raw"

# Read-only existing dataset, used only to read the panel's date range.
FIRE_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "fire_panel.parquet"

# Derived exposure artifacts produced by scripts/build_impact_layers.py.
CELL_EXPOSURE_PATH = IMPACT_DATA_DIR / "cell_exposure.parquet"
RING_EXPOSURE_PATH = IMPACT_DATA_DIR / "cell_ring_exposure.parquet"
SETTLEMENTS_PATH = IMPACT_DATA_DIR / "cell_settlements.parquet"
INFRASTRUCTURE_PATH = IMPACT_DATA_DIR / "cell_infrastructure.parquet"
ADMIN_UNITS_PATH = IMPACT_DATA_DIR / "admin_units.parquet"
BURNED_AREA_PATH = IMPACT_DATA_DIR / "burned_area_monthly.parquet"
MANIFEST_PATH = IMPACT_DATA_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Study area / model grid (must match the trained model exactly)
# ---------------------------------------------------------------------------
MODEL_LAT_MIN, MODEL_LAT_MAX = 32.0, 37.5
MODEL_LON_MIN, MODEL_LON_MAX = -2.0, 9.0
MODEL_LAT_STEP = 0.5
MODEL_LON_STEP = 0.625

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius

# ---------------------------------------------------------------------------
# Canonical exposure schema
# ---------------------------------------------------------------------------
CELL_ID_COLUMN = "cell_id"
CELL_AREA_COLUMN = "cell_area_km2"

# ESA WorldCover class codes -> short keys stored in the exposure artifact.
WORLDCOVER_CLASSES = {
    10: ("tree", "Tree cover"),
    20: ("shrub", "Shrubland"),
    30: ("grass", "Grassland"),
    40: ("crop", "Cropland"),
    50: ("built", "Built-up"),
    60: ("bare", "Bare / sparse vegetation"),
    70: ("snow", "Snow and ice"),
    80: ("water", "Permanent water bodies"),
    90: ("wetland", "Herbaceous wetland"),
    95: ("mangrove", "Mangroves"),
    100: ("moss", "Moss and lichen"),
}

# Land-cover classes that can carry a vegetation fire.
FLAMMABLE_LC_KEYS = ("tree", "shrub", "grass", "crop")

# Land-cover area columns required for the environmental dimension.
# Areas are in km²; the build stage aggregates WorldCover pixels by cell.
LC_AREA_COLUMNS = {
    "tree": "lc_tree_km2",
    "shrub": "lc_shrub_km2",
    "grass": "lc_grass_km2",
    "crop": "lc_crop_km2",
    "bare": "lc_bare_km2",
    "built": "lc_built_km2",
    "water": "lc_water_km2",
    "other": "lc_other_km2",
}

ENV_REQUIRED_COLUMNS = (CELL_AREA_COLUMN,) + tuple(LC_AREA_COLUMNS.values())

POP_REQUIRED_COLUMNS = (
    CELL_AREA_COLUMN,
    "pop_cell",
    "pop_5km",
    "pop_10km",
    "pop_25km",
)

INFRA_REQUIRED_COLUMNS = (
    CELL_AREA_COLUMN,
    "road_km_total",
    "building_count",
    "building_footprint_km2",
    "critical_count_total",
)

RING_DISTANCES_KM = (5, 10, 25)

# Optional columns the build stage may provide; their absence is reported
# rather than replaced with zeros.
OPTIONAL_POP_COLUMNS = ("settlement_count", "settlement_population")
OPTIONAL_INFRA_COLUMNS = (
    "road_km_primary",
    "road_km_secondary",
    "road_km_tertiary",
    "road_km_track",
    "built_surface_km2",
    "critical_hospital_count",
    "critical_school_count",
    "critical_power_count",
    "critical_airport_count",
    "critical_water_count",
    "critical_emergency_count",
)

# Required schemas for the auxiliary derived files (Stage 2 build outputs).
RING_REQUIRED_COLUMNS = (
    "lat_cell",
    "lon_cell",
    "ring_km",
    CELL_AREA_COLUMN,
    "population",
    "road_km",
    "building_count",
    "building_footprint_km2",
    "critical_count",
)
SETTLEMENTS_REQUIRED_COLUMNS = (
    "lat_cell",
    "lon_cell",
    "settlement_name",
    "population",
    "distance_km",
)
INFRASTRUCTURE_REQUIRED_COLUMNS = (
    "lat_cell",
    "lon_cell",
    "category",
    "asset_name",
)
ADMIN_UNITS_REQUIRED_COLUMNS = ("admin_code", "admin_name", "lat_cell", "lon_cell")
BURNED_AREA_REQUIRED_COLUMNS = (
    "date",
    "lat_cell",
    "lon_cell",
    "burned_area_km2",
)

# ---------------------------------------------------------------------------
# Status vocabulary — missing data is never represented by zeros
# ---------------------------------------------------------------------------
STATUS_AVAILABLE = "AVAILABLE"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_MISSING = "MISSING"
STATUS_INVALID = "INVALID"

# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------
REPORT_TIMEZONE = "UTC"
