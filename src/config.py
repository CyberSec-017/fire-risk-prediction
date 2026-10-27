"""
Shared settings for the whole pipeline. Change values here once;
every other script imports from this file.
"""

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Study area -- change this box to your region of interest.
# This default is northern Algeria.
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 32.0, 37.5
LON_MIN, LON_MAX = -2.0, 9.0

START_YEAR, END_YEAR = 2024, 2026

# POWER's native grid cell size. Do not change unless NASA changes it.
LAT_STEP = 0.5
LON_STEP = 0.625

# Weather variables to pull. Regional requests accept one parameter per
# call, so this list determines how many API calls fetch_power.py makes.
PARAMETERS = [
    "T2M",                  # mean air temp at 2m           (deg C)
    "T2M_MAX",              # daily max air temp             (deg C)
    "RH2M",                 # relative humidity at 2m        (%)
    "WS2M",                 # wind speed at 2m                (m/s)
    "PRECTOTCORR",          # corrected total precipitation   (mm/day)
    "ALLSKY_SFC_SW_DWN",    # solar radiation reaching ground (MJ/m2/day)
    "GWETROOT",             # root-zone soil wetness          (0-1)
]

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"

# NASA POWER regional requests must respect BOTH limits, otherwise the API
# returns HTTP 422:
#   - a box bigger than ~4.5 x 4.5 degrees is refused (too many grid points)
#   - a box smaller than 2 x 2 degrees is refused too
# The study area is therefore split into tiles whose sides are between
# MIN_TILE_DEG and MAX_TILE_DEG.
MIN_TILE_DEG = 2.0
MAX_TILE_DEG = 4.0  # a bit under 4.5 for safety margin


def _split_axis(vmin, vmax, min_size, max_size):
    """Split [vmin, vmax] into equal segments, each between min_size and max_size."""
    span = vmax - vmin
    if span < min_size:
        raise ValueError(
            f"Study area too small ({span} deg): POWER needs at least {min_size} deg."
        )
    n = math.ceil(span / max_size)      # fewest segments that respect max_size
    step = span / n                     # equal segments -> none is too small
    return [(vmin + i * step, vmin + (i + 1) * step) for i in range(n)]


def generate_tiles(lat_min=LAT_MIN, lat_max=LAT_MAX,
                   lon_min=LON_MIN, lon_max=LON_MAX,
                   max_size=MAX_TILE_DEG, min_size=MIN_TILE_DEG):
    """Split a bounding box into a grid of sub-boxes, each between min_size
    and max_size degrees on a side, that together cover the original box
    exactly (no overlap)."""
    lat_tiles = _split_axis(lat_min, lat_max, min_size, max_size)
    lon_tiles = _split_axis(lon_min, lon_max, min_size, max_size)
    return [(round(a, 4), round(b, 4), round(c, 4), round(d, 4))
            for (a, b) in lat_tiles for (c, d) in lon_tiles]


# ---------------------------------------------------------------------------
# Folder layout -- see README.md for the full tree
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
POWER_CACHE = RAW_DIR / "power_cache"
FIRMS_RAW = RAW_DIR / "firms_raw.csv"

INTERIM_DIR = ROOT / "data" / "interim"
FIRES_DAILY = INTERIM_DIR / "fires_daily.parquet"

PROCESSED_DIR = ROOT / "data" / "processed"
FIRE_PANEL = PROCESSED_DIR / "fire_panel.parquet"

for d in (RAW_DIR, POWER_CACHE, INTERIM_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)