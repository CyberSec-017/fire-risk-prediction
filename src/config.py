"""
Shared settings for the whole pipeline. Change values here once;
every other script imports from this file.
"""

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

# NASA POWER regional requests are capped at roughly 4.5 x 4.5 degrees
# (~100 grid points). A box bigger than that returns HTTP 422. Since our
# study area is usually bigger, we tile it into chunks under that limit
# and fetch/merge each chunk separately.
MAX_TILE_DEG = 4.0  # a bit under 4.5 for safety margin


def generate_tiles(lat_min=LAT_MIN, lat_max=LAT_MAX,
                    lon_min=LON_MIN, lon_max=LON_MAX,
                    max_size=MAX_TILE_DEG):
    """Split a bounding box into a grid of sub-boxes, each <= max_size degrees
    on a side, that together cover the original box exactly (no overlap)."""
    tiles = []
    lat = lat_min
    while lat < lat_max:
        lat_end = min(lat + max_size, lat_max)
        lon = lon_min
        while lon < lon_max:
            lon_end = min(lon + max_size, lon_max)
            tiles.append((round(lat, 4), round(lat_end, 4),
                          round(lon, 4), round(lon_end, 4)))
            lon = lon_end
        lat = lat_end
    return tiles

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
