"""
STEP 1 -- Data Collection (weather side)

Downloads NASA POWER daily weather data for the whole study area.

NASA POWER's regional endpoint caps requests at roughly 4.5 x 4.5 degrees
(~100 grid points) -- a bigger box returns HTTP 422 regardless of the date
range. So instead of one request per (parameter, year), we tile the study
area into sub-boxes under that limit and fetch each tile separately. Every
(parameter, year, tile) response is cached to its own CSV under
data/raw/power_cache/, and re-running this script skips files that already
exist, so it's safe to resume after a timeout or an error.

Usage:
    python src/fetch_power.py
"""

import datetime
import time

import requests

from config import (
    START_YEAR, END_YEAR, PARAMETERS, POWER_URL, POWER_CACHE,
    generate_tiles,
)

TILES = generate_tiles()

# NASA POWER publishes with a short lag -- the last few days are never
# available yet. For the current (or a future) year, cap the requested
# end date instead of asking for days that don't exist.
POWER_LAG_DAYS = 7


def _year_bounds(year: int) -> tuple[str, str]:
    """Full calendar year as (start, end) in YYYYMMDD, capped so an
    in-progress year is never requested (and cached) as if it were
    complete."""
    today = datetime.date.today()
    start = datetime.date(year, 1, 1)
    end = datetime.date(year, 12, 31)

    latest_available = today - datetime.timedelta(days=POWER_LAG_DAYS)
    if end > latest_available:
        end = latest_available

    if start > end:
        raise ValueError(
            f"Year {year} has no available data yet (latest POWER data "
            f"is {latest_available.isoformat()}). Remove it from "
            "config.py's END_YEAR until enough days have passed, or "
            "this run will cache an empty/partial file as if it were "
            "final."
        )

    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def fetch_one(param: str, year: int, tile_idx: int, box) -> None:
    lat_min, lat_max, lon_min, lon_max = box

    is_partial_year = year == datetime.date.today().year
    suffix = "_partial" if is_partial_year else ""
    out = POWER_CACHE / f"{param}_{year}_tile{tile_idx}{suffix}.csv"
    if out.exists():
        print(f"  skip {out.name} (already cached)")
        return

    start_str, end_str = _year_bounds(year)

    params = {
        "parameters": param,
        "community": "AG",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        # Full calendar year -- NOT a single day. build_panel.py's
        # rolling 7/14/30-day features need a continuous daily series;
        # a one-day range here would make those windows meaningless.
        # Capped by _year_bounds() so the current year isn't requested
        # past the days POWER has actually published.
        "start": start_str,
        "end": end_str,
        "format": "CSV",
    }
    if is_partial_year:
        print(f"  note: {year} is still in progress -- only fetching "
              f"through {end_str}, cached with a '_partial' suffix so "
              "it's easy to spot and re-fetch once the year is complete.")

    print(f"  fetching {param} {year} tile{tile_idx} "
          f"[{lat_min},{lat_max}] x [{lon_min},{lon_max}] ...", end=" ", flush=True)
    r = requests.get(POWER_URL, params=params, timeout=300)

    if r.status_code == 422:
        print(f"FAILED (422 -- tile still too large? response: {r.text[:200]})")
        return
    r.raise_for_status()

    out.write_text(r.text)
    print("ok")

    time.sleep(1)  # be polite -- POWER throttles aggressive clients


def main():
    print(f"Study area split into {len(TILES)} tile(s) to stay under "
          f"POWER's regional size limit.")
    for i, box in enumerate(TILES):
        print(f"  tile{i}: lat [{box[0]}, {box[1]}], lon [{box[2]}, {box[3]}]")

    print(f"\nFetching {len(PARAMETERS)} parameters x "
          f"{END_YEAR - START_YEAR + 1} years x {len(TILES)} tile(s) ...")
    for param in PARAMETERS:
        for year in range(START_YEAR, END_YEAR + 1):
            for tile_idx, box in enumerate(TILES):
                fetch_one(param, year, tile_idx, box)

    print("Done. Raw files are in data/raw/power_cache/")


if __name__ == "__main__":
    main()