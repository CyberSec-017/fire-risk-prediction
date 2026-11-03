
"""
Shared settings for the whole pipeline.
All scripts import their configuration from this file.
"""

import math
from pathlib import Path


# ============================================================
# 1. STUDY AREA — NORTHERN ALGERIA
# ============================================================

LAT_MIN, LAT_MAX = 32.0, 37.5
LON_MIN, LON_MAX = -2.0, 9.0

START_YEAR = 2024
END_YEAR = 2026


# ============================================================
# 2. NASA POWER GRID RESOLUTION
# ============================================================

LAT_STEP = 0.5
LON_STEP = 0.625


# ============================================================
# 3. METEOROLOGICAL PARAMETERS
# ============================================================

PARAMETERS = [
    "T2M",                  # Mean temperature at 2m (°C)
    "T2M_MAX",              # Maximum temperature at 2m (°C)
    "RH2M",                 # Relative humidity (%)
    "WS2M",                 # Wind speed (m/s)
    "PRECTOTCORR",          # Corrected precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN",    # Solar radiation (MJ/m²/day)
    "GWETROOT",             # Root zone soil wetness
]


# ============================================================
# 4. NASA POWER API
# ============================================================

POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/regional"
)

MIN_TILE_DEG = 2.0
MAX_TILE_DEG = 4.0


def _split_axis(vmin, vmax, min_size, max_size):
    """Split an axis into segments respecting POWER size limits."""

    span = vmax - vmin

    if span < min_size:
        raise ValueError(
            f"Study area too small: {span} degrees. "
            f"Minimum allowed size is {min_size} degrees."
        )

    n = math.ceil(span / max_size)
    step = span / n

    segments = [
        (vmin + i * step, vmin + (i + 1) * step)
        for i in range(n)
    ]

    if any((b - a) < min_size for a, b in segments):
        raise ValueError(
            "Unable to split study area within tile size limits."
        )

    return segments


def generate_tiles(
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    lon_min=LON_MIN,
    lon_max=LON_MAX,
    max_size=MAX_TILE_DEG,
    min_size=MIN_TILE_DEG,
):
    """
    Divide the study area into geographic tiles.
    Returns (lat_min, lat_max, lon_min, lon_max).
    """

    lat_tiles = _split_axis(
        lat_min, lat_max, min_size, max_size
    )

    lon_tiles = _split_axis(
        lon_min, lon_max, min_size, max_size
    )

    return [
        (
            round(lat1, 4),
            round(lat2, 4),
            round(lon1, 4),
            round(lon2, 4),
        )
        for lat1, lat2 in lat_tiles
        for lon1, lon2 in lon_tiles
    ]


# ============================================================
# 5. PROJECT DIRECTORIES AND FILES
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
POWER_CACHE = RAW_DIR / "power_cache"
FIRMS_RAW = RAW_DIR / "firms_raw.csv"

INTERIM_DIR = ROOT / "data" / "interim"
FIRES_DAILY = INTERIM_DIR / "fires_daily.parquet"

PROCESSED_DIR = ROOT / "data" / "processed"
FIRE_PANEL = PROCESSED_DIR / "fire_panel.parquet"


# ============================================================
# 6. AUTOMATIC DIRECTORY CREATION
# ============================================================

for directory in (
    RAW_DIR,
    POWER_CACHE,
    INTERIM_DIR,
    PROCESSED_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
