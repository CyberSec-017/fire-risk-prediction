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

START_YEAR, END_YEAR = 2018, 2024

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
    "GWETROOT",             # root-zone soil wetness          (0-1)
]

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"

# NASA POWER regional requests must be between roughly 2 degrees (minimum)
# and 4.5 degrees / ~100 grid points (maximum) per side, or the API
# returns HTTP 422. Since our study area is usually outside that range,
# we tile it into evenly-sized chunks that respect both bounds.
MAX_TILE_DEG = 4.0
MIN_TILE_DEG = 2.0


def _split_axis(lo, hi, max_size=MAX_TILE_DEG, min_size=MIN_TILE_DEG):
    """Split [lo, hi] into evenly-sized pieces, each between min_size and
    max_size degrees (as close to equal as possible), not left-to-right
    greedy chunks -- greedy chunking can leave an undersized remainder
    tile at the edge, which POWER also rejects."""
    total = hi - lo
    if total <= max_size:
        return [(round(lo, 4), round(hi, 4))]

    n = max(1, -(-total // max_size))  # ceil division
    while n > 1 and total / n < min_size:
        n -= 1
    n = int(n)

    size = total / n
    edges = [lo + i * size for i in range(n)] + [hi]
    return [(round(edges[i], 4), round(edges[i + 1], 4)) for i in range(n)]


def generate_tiles(lat_min=LAT_MIN, lat_max=LAT_MAX,
                    lon_min=LON_MIN, lon_max=LON_MAX):
    """Split a bounding box into a grid of sub-boxes that together cover
    the original box exactly, each respecting POWER's min/max size."""
    lat_ranges = _split_axis(lat_min, lat_max)
    lon_ranges = _split_axis(lon_min, lon_max)
    return [
        (la, lb, lo, lohi)
        for (la, lb) in lat_ranges
        for (lo, lohi) in lon_ranges
    ]

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

H = 3   # forecast horizon in days: "fire in the next H days"