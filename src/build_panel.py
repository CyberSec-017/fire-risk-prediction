
"""
STEP 3 -- Merge NASA POWER weather data with FIRMS fire labels.
STEP 4 -- Feature engineering.

Input:
    data/raw/power_cache/*.csv
    data/interim/fires_daily.parquet

Output:
    data/processed/fire_panel.parquet

The target 'burned' represents a FIRMS detection on the SAME day.
This is a same-day classification dataset, not a next-day forecast.
"""

import datetime
from io import StringIO

import numpy as np
import pandas as pd

from config import (
    PARAMETERS,
    START_YEAR,
    END_YEAR,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    POWER_CACHE,
    FIRES_DAILY,
    FIRE_PANEL,
    generate_tiles,
)

TILES = generate_tiles()
CURRENT_YEAR = datetime.date.today().year
KEYS = ["lat_cell", "lon_cell", "date"]


def _parse_power_csv(path, param):
    """Read a NASA POWER CSV and normalize its columns."""

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()

    header_end = next(
        (i for i, line in enumerate(lines) if "-END HEADER-" in line),
        None,
    )

    if header_end is None:
        raise ValueError(f"NASA POWER header not found: {path.name}")

    df = pd.read_csv(StringIO("\n".join(lines[header_end + 1:])))
    df.columns = [str(c).strip().upper() for c in df.columns]

    if not {"LAT", "LON"}.issubset(df.columns):
        raise ValueError(f"LAT/LON missing in {path.name}")

    if {"YEAR", "MO", "DY"}.issubset(df.columns):
        date_parts = pd.DataFrame({
            "year": pd.to_numeric(df["YEAR"], errors="coerce"),
            "month": pd.to_numeric(df["MO"], errors="coerce"),
            "day": pd.to_numeric(df["DY"], errors="coerce"),
        })
        df["date"] = pd.to_datetime(date_parts, errors="coerce")

    elif {"YEAR", "DOY"}.issubset(df.columns):
        year = pd.to_numeric(df["YEAR"], errors="coerce")
        doy = pd.to_numeric(df["DOY"], errors="coerce")
        df["date"] = (
            pd.to_datetime(year, format="%Y", errors="coerce")
            + pd.to_timedelta(doy - 1, unit="D")
        )
    else:
        raise ValueError(f"Unsupported date columns in {path.name}")

    excluded = {"LAT", "LON", "YEAR", "MO", "DY", "DOY", "DATE"}
    value_columns = [c for c in df.columns if c not in excluded]

    if param in value_columns:
        value_col = param
    elif len(value_columns) == 1:
        value_col = value_columns[0]
    else:
        raise ValueError(
            f"Cannot identify {param} in {path.name}: {value_columns}"
        )

    result = df[["LAT", "LON", "date", value_col]].copy()
    result.columns = KEYS + [param]

    result["lat_cell"] = pd.to_numeric(
        result["lat_cell"], errors="coerce"
    ).round(3)

    result["lon_cell"] = pd.to_numeric(
        result["lon_cell"], errors="coerce"
    ).round(3)

    result[param] = pd.to_numeric(result[param], errors="coerce")
    result[param] = result[param].replace(-999, np.nan)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    return result.dropna(subset=KEYS)


def _deduplicate_weather(frame, param):
    """Remove duplicated grid/date records at tile boundaries."""

    return frame.groupby(
        KEYS, as_index=False, dropna=False
    )[param].mean()


def load_power():
    """Load all POWER files and merge weather parameters."""

    merged = None

    for param in PARAMETERS:
        pieces = []

        for year in range(START_YEAR, END_YEAR + 1):
            suffix = "_partial" if year == CURRENT_YEAR else ""

            for tile_idx in range(len(TILES)):
                path = POWER_CACHE / (
                    f"{param}_{year}_tile{tile_idx}{suffix}.csv"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing POWER file: {path}"
                    )

                if suffix:
                    print(
                        f"Note: {year} data are partial; "
                        "the final date depends on NASA availability."
                    )

                pieces.append(_parse_power_csv(path, param))

        combined = pd.concat(pieces, ignore_index=True)
        combined = _deduplicate_weather(combined, param)

        if merged is None:
            merged = combined
        else:
            merged = merged.merge(
                combined,
                on=KEYS,
                how="outer",
                validate="one_to_one",
            )

        print(f"Loaded {param}: {len(merged):,} rows")

    if merged is None or merged.empty:
        raise ValueError("No NASA POWER data were loaded.")

    return merged.sort_values(KEYS).reset_index(drop=True)


def label_panel(panel):
    """Join daily FIRMS fire labels to the weather grid."""

    if not FIRES_DAILY.exists():
        raise FileNotFoundError(
            f"{FIRES_DAILY} is missing. Run clean_firms.py first."
        )

    fires = pd.read_parquet(FIRES_DAILY)

    required = set(KEYS + ["n_detections", "total_frp"])
    missing = required - set(fires.columns)

    if missing:
        raise ValueError(
            f"FIRMS columns missing: {sorted(missing)}"
        )

    fires = fires.copy()
    fires["date"] = pd.to_datetime(fires["date"], errors="coerce")
    fires = fires.dropna(subset=KEYS)

    if fires.duplicated(KEYS).any():
        raise ValueError("Duplicate FIRMS cell/day keys detected.")

    panel = panel.merge(
        fires[KEYS + ["n_detections", "total_frp"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    panel["burned"] = panel["n_detections"].notna().astype("int8")

    panel[["n_detections", "total_frp"]] = (
        panel[["n_detections", "total_frp"]].fillna(0)
    )

    print(
        f"Labeled panel: {len(panel):,} rows, "
        f"{panel['burned'].mean():.2%} positive class"
    )

    return panel


def _dry_days_for_group(rain):
    """Count consecutive observed days with precipitation <= 1 mm."""

    output = []
    counter = 0

    for value in rain:
        if pd.isna(value):
            output.append(np.nan)
            counter = 0
        elif value > 1.0:
            counter = 0
            output.append(0)
        else:
            counter += 1
            output.append(counter)

    return pd.Series(output, index=rain.index, dtype="float64")


def add_features(df):
    """Add backward-looking rolling weather and seasonal features."""

    df = df.sort_values(KEYS).copy()
    groups = df.groupby(["lat_cell", "lon_cell"], sort=False)

    for window in (7, 14, 30):
        df[f"rain_{window}d"] = groups["PRECTOTCORR"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )

        df[f"tmax_{window}d"] = groups["T2M_MAX"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

        df[f"rhmin_{window}d"] = groups["RH2M"].transform(
            lambda s: s.rolling(window, min_periods=1).min()
        )

    df["dry_days"] = groups["PRECTOTCORR"].transform(
        _dry_days_for_group
    )

    day_of_year = df["date"].dt.dayofyear

    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df


def validate_panel(df):
    """Check keys, coordinates and target values."""

    print("\n===== PANEL QUALITY CHECKS =====")
    print("Dimensions:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())

    duplicates = int(df.duplicated(KEYS).sum())
    print("Duplicate cell/day keys:", duplicates)

    outside = (
        (df["lat_cell"] < LAT_MIN)
        | (df["lat_cell"] > LAT_MAX)
        | (df["lon_cell"] < LON_MIN)
        | (df["lon_cell"] > LON_MAX)
    )

    outside_count = int(outside.sum())
    print("Coordinates outside study area:", outside_count)

    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

    if duplicates:
        raise ValueError("Duplicate cell/day keys found.")

    if outside_count:
        raise ValueError("Coordinates outside study area.")

    if not df["burned"].isin([0, 1]).all():
        raise ValueError("Target burned must contain only 0 and 1.")


def main():
    print("Loading NASA POWER data...")
    panel = load_power()

    print("\nJoining FIRMS fire labels...")
    panel = label_panel(panel)

    print("\nAdding features...")
    panel = add_features(panel)

    validate_panel(panel)

    FIRE_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FIRE_PANEL, index=False)

    print(
        f"\nSaved {FIRE_PANEL} "
        f"({len(panel):,} rows, {panel.shape[1]} columns)"
    )

    print(
        "\nReminder: burned is a same-day target. "
        "Do not use n_detections or total_frp as model predictors."
    )


if __name__ == "__main__":
    main()
