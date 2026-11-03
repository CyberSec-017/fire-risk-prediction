
"""
Shared configuration for the entire Fire Risk Prediction pipeline.

All scripts import their settings from this file.
Changing a parameter here updates the configuration used by the other scripts.
"""

import math
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. GEOGRAPHIC LIMITS — STUDY AREA
# ---------------------------------------------------------------------------
# Define the geographic bounding box of the study area.
# The project focuses on northern Algeria and the surrounding region.
#
# LAT_MIN: minimum latitude (southern boundary).
# LAT_MAX: maximum latitude (northern boundary).
# LON_MIN: minimum longitude (western boundary).
# LON_MAX: maximum longitude (eastern boundary).

LAT_MIN, LAT_MAX = 32.0, 37.5
LON_MIN, LON_MAX = -2.0, 9.0


# ---------------------------------------------------------------------------
# 2. DATA COLLECTION PERIOD
# ---------------------------------------------------------------------------
# Define the first and last years of the data collection period.
# The project uses fire detections and meteorological data from 2024 to 2026.
#
# Note: 2026 data may be incomplete because the year is still in progress.

START_YEAR, END_YEAR = 2024, 2026


# ---------------------------------------------------------------------------
# 3. NASA POWER GRID CELL SIZE
# ---------------------------------------------------------------------------
# Define the spatial resolution of the NASA POWER native grid.
#
# LAT_STEP: latitude increment between grid cells (degrees).
# LON_STEP: longitude increment between grid cells (degrees).
#
# These values are used to associate fire detections with weather grid cells.
# Do not change them unless the spatial resolution used by the project changes.

LAT_STEP = 0.5
LON_STEP = 0.625


# ---------------------------------------------------------------------------
# 4. NASA POWER METEOROLOGICAL VARIABLES
# ---------------------------------------------------------------------------
# List of meteorological variables downloaded from NASA POWER.
# Each variable represents a different weather or soil condition.
#
# fetch_power.py uses this list to determine which parameters to request.
# Regional requests accept one parameter per call.

PARAMETERS = [
    "T2M",                  # Mean air temperature at 2m (°C)
    "T2M_MAX",              # Daily maximum air temperature at 2m (°C)
    "RH2M",                 # Relative humidity at 2m (%)
    "WS2M",                 # Wind speed at 2m (m/s)
    "PRECTOTCORR",          # Corrected total precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN",    # Solar radiation reaching the ground (MJ/m²/day)
    "GWETROOT",             # Root-zone soil wetness (0-1)
]


# NASA POWER API endpoint used to retrieve daily regional meteorological data.
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"


# ---------------------------------------------------------------------------
# 5. GEOGRAPHIC TILE GENERATION
# ---------------------------------------------------------------------------
# NASA POWER regional requests have geographic size restrictions:
# - A tile must not exceed approximately 4.5 x 4.5 degrees.
# - A tile must not be smaller than 2 x 2 degrees.
#
# To respect these restrictions, the study area is divided into smaller tiles.
#
# MIN_TILE_DEG: minimum allowed tile dimension (degrees).
# MAX_TILE_DEG: maximum tile dimension (degrees).
# A value of 4.0 provides a safety margin below the API's approximate limit.

MIN_TILE_DEG = 2.0
MAX_TILE_DEG = 4.0


def _split_axis(vmin, vmax, min_size, max_size):
    """
    Divide a geographic axis into equal segments.

    Each segment must have a size between min_size and max_size.

    Parameters:
        vmin: minimum coordinate.
        vmax: maximum coordinate.
        min_size: minimum permitted segment size.
        max_size: maximum permitted segment size.

    Returns:
        A list of coordinate intervals.
    """
    span = vmax - vmin

    if span < min_size:
        raise ValueError(
            f"Study area too small ({span} deg): POWER needs at least {min_size} deg."
        )

    # Calculate the minimum number of segments needed to respect max_size.
    n = math.ceil(span / max_size)

    # Calculate equal segment sizes so that all segments have the same width.
    step = span / n

    return [
        (vmin + i * step, vmin + (i + 1) * step)
        for i in range(n)
    ]


def generate_tiles(
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    lon_min=LON_MIN,
    lon_max=LON_MAX,
    max_size=MAX_TILE_DEG,
    min_size=MIN_TILE_DEG
):
    """
    Divide the study area's bounding box into smaller geographic tiles.

    The generated tiles:
    - Respect the minimum and maximum geographic size.
    - Cover the original study area completely.
    - Do not overlap.

    Returns:
        A list of tuples in the following order:
        (latitude_min, latitude_max, longitude_min, longitude_max).

    With the default study area, the function generates 6 geographic tiles.
    """

    # Divide the latitude range into smaller intervals.
    lat_tiles = _split_axis(lat_min, lat_max, min_size, max_size)

    # Divide the longitude range into smaller intervals.
    lon_tiles = _split_axis(lon_min, lon_max, min_size, max_size)

    # Combine latitude and longitude intervals to create rectangular tiles.
    # Round coordinates to 4 decimal places for consistency.
    return [
        (round(a, 4), round(b, 4), round(c, 4), round(d, 4))
        for (a, b) in lat_tiles
        for (c, d) in lon_tiles
    ]


# ---------------------------------------------------------------------------
# 6. PROJECT FILE AND FOLDER PATHS
# ---------------------------------------------------------------------------
# Define the paths used by the entire data processing pipeline.
#
# ROOT: main project directory.
# RAW_DIR: original downloaded datasets.
# POWER_CACHE: NASA POWER downloaded CSV files.
# FIRMS_RAW: merged raw NASA FIRMS CSV file.
# INTERIM_DIR: intermediate datasets after initial preprocessing.
# FIRES_DAILY: cleaned fire detections grouped by cell and day.
# PROCESSED_DIR: final datasets ready for analysis and modeling.
# FIRE_PANEL: merged fire and meteorological dataset.
#
# pathlib.Path is used to build paths in a platform-independent way.

ROOT = Path(__file__).resolve().parent.parent

# Raw data directories and files.
RAW_DIR = ROOT / "data" / "raw"
POWER_CACHE = RAW_DIR / "power_cache"
FIRMS_RAW = RAW_DIR / "firms_raw.csv"

# Intermediate data directories and files.
INTERIM_DIR = ROOT / "data" / "interim"
FIRES_DAILY = INTERIM_DIR / "fires_daily.parquet"

# Processed data directories and files.
PROCESSED_DIR = ROOT / "data" / "processed"
FIRE_PANEL = PROCESSED_DIR / "fire_panel.parquet"


# ---------------------------------------------------------------------------
# 7. AUTOMATIC DIRECTORY CREATION
# ---------------------------------------------------------------------------
# Create the required directories if they do not already exist.
# parents=True also creates missing parent directories.
# exist_ok=True prevents an error if a directory already exists.

for d in (RAW_DIR, POWER_CACHE, INTERIM_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)
