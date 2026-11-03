
"""
STEP 1 — Data Collection: NASA POWER weather data.

Downloads daily meteorological data for the study area.
Each file corresponds to one parameter, one year and one geographic tile.

Files are stored in:
    data/raw/power_cache/

Existing files are skipped by default.

Use --refresh-partial to update the current year's partial files.
"""

import argparse
import datetime
import sys
import time

import requests

from config import (
    START_YEAR,
    END_YEAR,
    PARAMETERS,
    POWER_URL,
    POWER_CACHE,
    generate_tiles,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

TILES = generate_tiles()

# NASA POWER data publication delay
POWER_LAG_DAYS = 7

MAX_RETRIES = 4
REQUEST_TIMEOUT = 300
PAUSE_BETWEEN = 1.0

RETRY_STATUS = {429, 500, 502, 503, 504}


# ============================================================
# 2. DATE MANAGEMENT
# ============================================================

def _year_bounds(year: int) -> tuple[str, str]:
    """
    Return the available date range for a year.
    Dates use YYYYMMDD format.
    """

    today = datetime.date.today()

    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)

    latest_available = today - datetime.timedelta(
        days=POWER_LAG_DAYS
    )

    if end > latest_available:
        end = latest_available

    if start > end:
        raise ValueError(
            f"No available data for year {year}. "
            f"Latest requested date: {latest_available}."
        )

    return (
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
    )


# ============================================================
# 3. CACHE FILE MANAGEMENT
# ============================================================

def _cache_file(param: str, year: int, tile_idx: int):
    """
    Generate the cache filename.

    The current year uses the suffix '_partial'.
    """

    suffix = (
        "_partial"
        if year == datetime.date.today().year
        else ""
    )

    return POWER_CACHE / (
        f"{param}_{year}_tile{tile_idx}{suffix}.csv"
    )


# ============================================================
# 4. DOWNLOAD ONE POWER FILE
# ============================================================

def fetch_one(
    param: str,
    year: int,
    tile_idx: int,
    box,
    force=False,
) -> str:
    """
    Download one POWER file.

    Returns:
        'cached'  : file already exists
        'ok'      : successful download
        'failed'  : download failed
    """

    lat_min, lat_max, lon_min, lon_max = box

    out = _cache_file(param, year, tile_idx)

    POWER_CACHE.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"  skip {out.name} (already cached)")
        return "cached"

    start_str, end_str = _year_bounds(year)

    request_params = {
        "parameters": param,
        "community": "AG",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": start_str,
        "end": end_str,
        "format": "CSV",
    }

    print(
        f"  fetching {out.name} "
        f"[{lat_min},{lat_max}] x "
        f"[{lon_min},{lon_max}] "
        f"{start_str}-{end_str} ...",
        end=" ",
        flush=True,
    )

    for attempt in range(1, MAX_RETRIES + 1):

        wait = 5 * attempt

        try:
            response = requests.get(
                POWER_URL,
                params=request_params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as exc:

            print(
                f"\n    Network error ({type(exc).__name__}), "
                f"retry in {wait}s",
                end=" ",
                flush=True,
            )

            if attempt < MAX_RETRIES:
                time.sleep(wait)

            continue

        if response.status_code == 422:
            print(
                f"FAILED (HTTP 422: {response.text[:200]})"
            )
            return "failed"

        if response.status_code in RETRY_STATUS:

            print(
                f"\n    HTTP {response.status_code}, "
                f"retry in {wait}s",
                end=" ",
                flush=True,
            )

            if attempt < MAX_RETRIES:
                time.sleep(wait)

            continue

        if response.status_code != 200:
            print(f"FAILED (HTTP {response.status_code})")
            return "failed"

        if "-END HEADER-" not in response.text:
            print(
                "FAILED (response is not a valid NASA POWER CSV)"
            )
            return "failed"

        # Save through a temporary file to avoid incomplete files.
        tmp = out.with_name(out.name + ".tmp")

        try:
            tmp.write_text(
                response.text,
                encoding="utf-8",
            )

            # Replace the old cache only after successful download.
            tmp.replace(out)

        except OSError as exc:

            if tmp.exists():
                tmp.unlink()

            print(f"FAILED (cannot write cache: {exc})")
            return "failed"

        print("ok")

        time.sleep(PAUSE_BETWEEN)

        return "ok"

    print(f"FAILED after {MAX_RETRIES} attempts")

    return "failed"


# ============================================================
# 5. MAIN DOWNLOAD PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Download NASA POWER daily weather data."
    )

    parser.add_argument(
        "--refresh-partial",
        action="store_true",
        help=(
            "Refresh current-year partial files "
            "without deleting old files first."
        ),
    )

    args = parser.parse_args()

    print(
        f"Study area split into {len(TILES)} geographic tiles:"
    )

    for index, box in enumerate(TILES):
        print(
            f"  tile{index}: "
            f"latitude [{box[0]}, {box[1]}], "
            f"longitude [{box[2]}, {box[3]}]"
        )

    years = range(START_YEAR, END_YEAR + 1)

    total = (
        len(PARAMETERS)
        * len(years)
        * len(TILES)
    )

    print(
        f"\nFetching {len(PARAMETERS)} parameters x "
        f"{len(years)} years x "
        f"{len(TILES)} tiles = {total} files ..."
    )

    counts = {
        "cached": 0,
        "ok": 0,
        "failed": 0,
    }

    failed = []

    current_year = datetime.date.today().year

    for param in PARAMETERS:

        for year in years:

            for tile_idx, box in enumerate(TILES):

                refresh_this_file = (
                    args.refresh_partial
                    and year == current_year
                )

                status = fetch_one(
                    param,
                    year,
                    tile_idx,
                    box,
                    force=refresh_this_file,
                )

                counts[status] += 1

                if status == "failed":
                    failed.append(
                        _cache_file(
                            param,
                            year,
                            tile_idx,
                        ).name
                    )

    print(
        f"\nSummary: {counts['ok']} downloaded, "
        f"{counts['cached']} already cached, "
        f"{counts['failed']} failed "
        f"(total {total})."
    )

    if failed:

        print("Failed files:")

        for name in failed:
            print(f"  - {name}")

        sys.exit(1)

    print(
        "Done. POWER files are stored in "
        "data/raw/power_cache/."
    )


if __name__ == "__main__":
    main()
