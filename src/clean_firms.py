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

from config import FIRMS_RAW, FIRES_DAILY, LAT_STEP, LON_STEP, INTERIM_DIR, RAW_DIR
WINTER_MAX_SHARE = 0.10   # set from the checks.py table

def load_raw() -> pd.DataFrame:
    if not FIRMS_RAW.exists():
        raise FileNotFoundError(
            f"{FIRMS_RAW} not found.\n"
            "Download it from https://firms.modaps.eosdis.nasa.gov/download/ "
            "for your study area and date range, then save it at this path."
        )
    
    raw = pd.read_csv("E:/CS/Projects/Data Mining/fire-risk-prediction/data/raw/firms_raw.csv")
    year = raw["acq_date"].str[:4]
    print(raw.groupby(year).size())                         # detections per year, raw
    print(raw[raw["type"].isna()].groupby(year).size())     # rows with missing type
    print(raw["acq_date"].min(), raw["acq_date"].max())    
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
        if not pd.api.types.is_numeric_dtype(conf):
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
    df["lat_cell"] = ((df["latitude"] / LAT_STEP).round() * LAT_STEP).round(4) + 0.0
    df["lon_cell"] = ((df["longitude"] / LON_STEP).round() * LON_STEP).round(4) + 0.0
    df["date"] = pd.to_datetime(df["acq_date"])
    return df


def collapse_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["lat_cell", "lon_cell", "date"])
          .agg(n_detections=("latitude", "size"),
               total_frp=("frp", "sum"))
          .reset_index()
    )

    # Wildfires are seasonal; industrial heat sources burn all year. Judge each
    # cell by how often it burns in winter, using pre-test years only.
    ref = daily[daily["date"] < "2022-01-01"]
    days = pd.date_range(daily["date"].min(), "2021-12-31")
    n_winter = days.month.isin([12, 1, 2]).sum()
    wdays = ref[ref["date"].dt.month.isin([12, 1, 2])].groupby(["lat_cell", "lon_cell"]).size()
    persistent = (wdays / n_winter)[lambda s: s > WINTER_MAX_SHARE].index

    pd.DataFrame(list(persistent), columns=["lat_cell", "lon_cell"]).to_csv(
        INTERIM_DIR / "excluded_cells.csv", index=False)
    if len(persistent):
        print(f"  excluding {len(persistent)} year-round heat-source cell(s)")
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
