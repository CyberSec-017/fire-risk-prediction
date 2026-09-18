"""
STEP 2 -- Data Preprocessing (fire side)

Reads the raw FIRMS CSV, applies quality filters, snaps each detection
to the nearest NASA POWER grid cell, and collapses same-day detections
in the same cell into one row.

Input:  data/raw/firms_raw.csv          (download manually from FIRMS)
Output: data/interim/fires_daily.parquet

Usage:
    python src/clean_firms.py
"""

import pandas as pd

from config import FIRMS_RAW, FIRES_DAILY, LAT_STEP, LON_STEP


def load_raw() -> pd.DataFrame:
    if not FIRMS_RAW.exists():
        raise FileNotFoundError(
            f"{FIRMS_RAW} not found.\n"
            "Download it from https://firms.modaps.eosdis.nasa.gov/download/ "
            "for your study area and date range, then save it at this path."
        )
    df = pd.read_csv(FIRMS_RAW)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)

    # type: 0 = presumed vegetation fire, 1 = volcano, 2 = other static
    # land source (gas flares etc.), 3 = offshore. Keep vegetation only.
    if "type" in df.columns:
        df = df[df["type"] == 0]
    n1 = len(df)

    # confidence: VIIRS uses l/n/h (low/nominal/high); MODIS uses 0-100.
    if "confidence" in df.columns:
        conf = df["confidence"]
        if conf.dtype == object:
            df = df[conf.str.lower().isin(["n", "h"])]
        else:
            df = df[conf >= 50]
    n2 = len(df)

    print(f"  raw detections        : {n0:,}")
    print(f"  after type filter     : {n1:,}")
    print(f"  after confidence filter: {n2:,}")
    return df


def snap_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_cell"] = (df["latitude"] / LAT_STEP).round() * LAT_STEP
    df["lon_cell"] = (df["longitude"] / LON_STEP).round() * LON_STEP
    df["date"] = pd.to_datetime(df["acq_date"])
    return df


def collapse_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["lat_cell", "lon_cell", "date"])
          .agg(n_detections=("latitude", "size"),
               total_frp=("frp", "sum"))
          .reset_index()
    )

    # Remove cells that burn on more than half of all days in range --
    # these are persistent industrial heat sources, not wildfires.
    days_span = (daily["date"].max() - daily["date"].min()).days + 1
    per_cell = daily.groupby(["lat_cell", "lon_cell"]).size()
    persistent = per_cell[per_cell > 0.5 * days_span].index

    if len(persistent):
        print(f"  dropping {len(persistent)} persistent hotspot cell(s)")
        idx = daily.set_index(["lat_cell", "lon_cell"]).index
        daily = daily[~idx.isin(persistent)]

    return daily


def main():
    print("Loading raw FIRMS data ...")
    df = load_raw()

    print("Applying quality filters ...")
    df = filter_quality(df)

    print("Snapping detections to POWER grid cells ...")
    df = snap_to_grid(df)

    print("Collapsing to one row per (cell, day) ...")
    daily = collapse_to_daily(df)

    daily.to_parquet(FIRES_DAILY, index=False)
    print(f"Saved {FIRES_DAILY} ({len(daily):,} fire-day rows)")


if __name__ == "__main__":
    main()
