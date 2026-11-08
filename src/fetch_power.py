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

import time

import requests

from config import (
    START_YEAR, END_YEAR, PARAMETERS, POWER_URL,
    generate_tiles, ensure_data_dirs, power_cache_path,
)

TILES = generate_tiles()


def fetch_one(param: str, year: int, tile_idx: int, box) -> None:
    lat_min, lat_max, lon_min, lon_max = box
    out = power_cache_path(param, year, tile_idx)
    if out.exists() and out.stat().st_size > 10_000:
        print(f"  skip {out.name} (already cached)")
        return

    params = {
        "parameters": param,
        "community": "AG",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "CSV",
    }

    print(f"  fetching {param} {year} tile{tile_idx} "
          f"[{lat_min},{lat_max}] x [{lon_min},{lon_max}] ...", end=" ", flush=True)
    r = requests.get(POWER_URL, params=params, timeout=300)
    
    

    if r.status_code == 422:
        print(f"FAILED (422 -- tile still too large? response: {r.text[:200]})")
        return
    r.raise_for_status()
    if r.text.count("\n") < 1000:
        print(f"SUSPICIOUS ({r.text.count(chr(10))} lines), not cached")
        return

    out.write_text(r.text, encoding="utf-8")
    print("ok")

    time.sleep(1)  # be polite -- POWER throttles aggressive clients


def main():
    ensure_data_dirs()
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
