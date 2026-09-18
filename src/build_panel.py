"""
STEP 3 -- Merge (join weather grid with fire labels)
STEP 4 -- Feature Engineering (rolling dryness features)

Reads every cached POWER file, merges them into one wide weather table
(every cell x every day), left-joins the cleaned FIRMS fire-days onto
it to create the binary label, then adds rolling-window features that
capture antecedent dryness -- the real driver of fire risk.

Input:  data/raw/power_cache/*.csv
        data/interim/fires_daily.parquet
Output: data/processed/fire_panel.parquet

Usage:
    python src/build_panel.py
"""

import numpy as np
import pandas as pd

from config import (
    PARAMETERS, START_YEAR, END_YEAR,
    POWER_CACHE, FIRES_DAILY, FIRE_PANEL,
)


# ---------------------------------------------------------------------------
# STEP 3a -- load and merge the POWER weather grid
# ---------------------------------------------------------------------------

def _parse_power_csv(path, param: str) -> pd.DataFrame:
    """
    POWER CSVs start with a metadata block ending at a line containing
    '-END HEADER-'. If your download has a different column layout than
    LAT, LON, YEAR, MO, DY, <value>, this is the only place to fix it --
    open one cached file in a text editor and compare.
    """
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "-END HEADER-" in l) + 1

    df = pd.read_csv(path, skiprows=start)
    df.columns = [c.strip().upper() for c in df.columns]

    if {"YEAR", "MO", "DY"}.issubset(df.columns):
        df["date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY))
        value_col = [c for c in df.columns
                     if c not in {"LAT", "LON", "YEAR", "MO", "DY", "DOY", "date"}][0]
        df = df[["LAT", "LON", "date", value_col]]
        df.columns = ["lat_cell", "lon_cell", "date", param]
    else:
        raise ValueError(
            f"Unexpected POWER column layout in {path.name}: {list(df.columns)}. "
            "Adjust _parse_power_csv to match."
        )

    df[param] = df[param].replace(-999, np.nan)  # POWER's missing-value code
    return df


def load_power() -> pd.DataFrame:
    merged = None
    for param in PARAMETERS:
        pieces = [
            _parse_power_csv(POWER_CACHE / f"{param}_{y}.csv", param)
            for y in range(START_YEAR, END_YEAR + 1)
        ]
        one = pd.concat(pieces, ignore_index=True)
        merged = one if merged is None else merged.merge(
            one, on=["lat_cell", "lon_cell", "date"], how="outer"
        )
        print(f"  merged {param}: {len(merged):,} rows so far")
    return merged


# ---------------------------------------------------------------------------
# STEP 3b -- join fire labels onto the weather grid
# ---------------------------------------------------------------------------

def label_panel(panel: pd.DataFrame) -> pd.DataFrame:
    fires = pd.read_parquet(FIRES_DAILY)

    # Left join: POWER already has every (cell, date), so rows with no
    # matching fire simply get NaN here -- that NaN becomes our 0 label.
    panel = panel.merge(fires, on=["lat_cell", "lon_cell", "date"], how="left")
    panel["burned"] = panel["n_detections"].notna().astype(int)
    panel[["n_detections", "total_frp"]] = panel[["n_detections", "total_frp"]].fillna(0)

    rate = panel["burned"].mean()
    print(f"  labeled panel: {len(panel):,} rows, {rate:.2%} positive class")
    return panel


# ---------------------------------------------------------------------------
# STEP 4 -- rolling dryness features
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every feature here only looks BACKWARDS in time. Never derive a
    feature from a future date, or evaluation results become invalid.
    """
    df = df.sort_values(["lat_cell", "lon_cell", "date"])
    g = df.groupby(["lat_cell", "lon_cell"], sort=False)

    for window in (7, 14, 30):
        df[f"rain_{window}d"] = (
            g["PRECTOTCORR"].rolling(window, min_periods=1).sum()
             .reset_index(level=[0, 1], drop=True)
        )
        df[f"tmax_{window}d"] = (
            g["T2M_MAX"].rolling(window, min_periods=1).mean()
             .reset_index(level=[0, 1], drop=True)
        )
        df[f"rhmin_{window}d"] = (
            g["RH2M"].rolling(window, min_periods=1).min()
             .reset_index(level=[0, 1], drop=True)
        )

    # Consecutive dry days: counter resets to 0 every day it rains >1mm.
    wet = df["PRECTOTCORR"] > 1.0
    streak_id = wet.groupby([df["lat_cell"], df["lon_cell"]]).cumsum()
    df["dry_days"] = df.groupby(["lat_cell", "lon_cell", streak_id]).cumcount()

    # Seasonality as two smooth cyclic features instead of raw day-of-year.
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def main():
    print("Loading and merging POWER weather grid ...")
    panel = load_power()

    print("Joining fire labels ...")
    panel = label_panel(panel)

    print("Adding rolling dryness features ...")
    panel = add_features(panel)

    panel.to_parquet(FIRE_PANEL, index=False)
    print(f"Saved {FIRE_PANEL} ({len(panel):,} rows, {panel.shape[1]} columns)")


if __name__ == "__main__":
    main()
