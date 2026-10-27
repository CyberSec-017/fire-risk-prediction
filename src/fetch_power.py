"""
STEP 1 -- Data Collection (weather side)

Downloads NASA POWER daily weather data for the whole study area.

NASA POWER's regional endpoint caps requests at roughly 4.5 x 4.5 degrees
(~100 grid points) -- a bigger box returns HTTP 422 regardless of the date
range. So the study area is tiled into sub-boxes under that limit and each
tile is fetched separately. Every (parameter, year, tile) response is cached
to its own CSV under data/raw/power_cache/, and re-running this script skips
files that already exist, so it is safe to resume after a timeout or error.

Improvements over the first version:
  - automatic retries with back-off (network errors, HTTP 429 / 5xx)
  - a response is cached ONLY if it really is a POWER CSV (-END HEADER-)
  - files are written atomically (no half-written CSV if the script is killed)
  - a final summary lists every failed download and exits with code 1,
    so a missing tile is never silent
  - --refresh-partial re-downloads the in-progress year

Usage:
    python src/fetch_power.py
    python src/fetch_power.py --refresh-partial
"""

import argparse
import datetime
import sys
import time

import requests

from config import (
    START_YEAR, END_YEAR, PARAMETERS, POWER_URL, POWER_CACHE,
    generate_tiles,
)

TILES = generate_tiles()

# NASA POWER publishes with a short lag -- the last few days are never
# available yet.
POWER_LAG_DAYS = 7

MAX_RETRIES = 4          # attempts per file
REQUEST_TIMEOUT = 300    # seconds
PAUSE_BETWEEN = 1.0      # seconds, be polite with the API
RETRY_STATUS = {429, 500, 502, 503, 504}


def _year_bounds(year: int) -> tuple[str, str]:
    """Full calendar year as (start, end) in YYYYMMDD, capped so an
    in-progress year is never requested as if it were complete."""
    today = datetime.date.today()
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)

    latest_available = today - datetime.timedelta(days=POWER_LAG_DAYS)
    if end > latest_available:
        end = latest_available

    if start > end:
        raise ValueError(
            f"Year {year} has no available data yet (latest POWER data "
            f"is {latest_available.isoformat()}). Lower END_YEAR in config.py."
        )
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _cache_file(param: str, year: int, tile_idx: int):
    suffix = "_partial" if year == datetime.date.today().year else ""
    return POWER_CACHE / f"{param}_{year}_tile{tile_idx}{suffix}.csv"


def fetch_one(param: str, year: int, tile_idx: int, box) -> str:
    """Returns 'cached', 'ok' or 'failed'."""
    lat_min, lat_max, lon_min, lon_max = box
    out = _cache_file(param, year, tile_idx)

    if out.exists():
        print(f"  skip {out.name} (already cached)")
        return "cached"

    start_str, end_str = _year_bounds(year)
    params = {
        "parameters": param,          # regional endpoint: one parameter per call
        "community": "AG",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": start_str,           # full year: rolling features need a continuous series
        "end": end_str,
        "format": "CSV",
    }

    print(f"  fetching {out.name} "
          f"[{lat_min},{lat_max}] x [{lon_min},{lon_max}] {start_str}-{end_str} ...",
          end=" ", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        wait = 5 * attempt
        try:
            r = requests.get(POWER_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            print(f"\n    network error ({type(e).__name__}), retry in {wait}s",
                  end=" ", flush=True)
            time.sleep(wait)
            continue

        if r.status_code == 422:
            print(f"FAILED (422 -- tile too large or invalid request: {r.text[:200]})")
            return "failed"
        if r.status_code in RETRY_STATUS:
            print(f"\n    HTTP {r.status_code}, retry in {wait}s", end=" ", flush=True)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"FAILED (HTTP {r.status_code})")
            return "failed"
        if "-END HEADER-" not in r.text:
            print("FAILED (response is not a POWER CSV)")
            return "failed"

        tmp = out.with_name(out.name + ".tmp")   # atomic write
        tmp.write_text(r.text)
        tmp.replace(out)
        print("ok")
        time.sleep(PAUSE_BETWEEN)
        return "ok"

    print(f"FAILED after {MAX_RETRIES} attempts")
    return "failed"


def main():
    parser = argparse.ArgumentParser(description="Download NASA POWER weather data")
    parser.add_argument("--refresh-partial", action="store_true",
                        help="delete cached *_partial.csv files and download them again")
    args = parser.parse_args()

    if args.refresh_partial:
        old = list(POWER_CACHE.glob("*_partial.csv"))
        for f in old:
            f.unlink()
        print(f"Removed {len(old)} partial file(s) from the cache.")

    print(f"Study area split into {len(TILES)} tile(s):")
    for i, box in enumerate(TILES):
        print(f"  tile{i}: lat [{box[0]}, {box[1]}], lon [{box[2]}, {box[3]}]")

    n_years = END_YEAR - START_YEAR + 1
    total = len(PARAMETERS) * n_years * len(TILES)
    print(f"\nFetching {len(PARAMETERS)} parameters x {n_years} years x "
          f"{len(TILES)} tile(s) = {total} files ...")

    counts = {"cached": 0, "ok": 0, "failed": 0}
    failed = []
    for param in PARAMETERS:
        for year in range(START_YEAR, END_YEAR + 1):
            for tile_idx, box in enumerate(TILES):
                status = fetch_one(param, year, tile_idx, box)
                counts[status] += 1
                if status == "failed":
                    failed.append(_cache_file(param, year, tile_idx).name)

    print(f"\nSummary: {counts['ok']} downloaded, {counts['cached']} already cached, "
          f"{counts['failed']} failed (total {total}).")
    if failed:
        print("Missing files (re-run the script to retry them):")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    print("Done. Raw files are in data/raw/power_cache/")


if __name__ == "__main__":
    main()