"""
STEP 2 — FIRMS Data Preprocessing

This script prepares NASA FIRMS fire detections for integration
with NASA POWER meteorological data.

Processing steps:
    1. Load the raw FIRMS CSV file.
    2. Validate required columns and data types.
    3. Remove invalid observations.
    4. Keep vegetation-fire detections.
    5. Apply the confidence filter.
    6. Keep detections inside the study area.
    7. Snap detections to the nearest NASA POWER grid cell.
    8. Aggregate detections by grid cell and acquisition date.
    9. Run quality checks on the final dataset.
    10. Save the result as a Parquet file.

Input:
    data/raw/firms_raw.csv

Output:
    data/interim/fires_daily.parquet

Usage:
    python src/clean_firms.py
"""

import pandas as pd

from config import FIRMS_RAW, FIRES_DAILY, LAT_STEP, LON_STEP


# Geographic limits of the study area.
LAT_MIN = 32.0
LAT_MAX = 37.5
LON_MIN = -2.0
LON_MAX = 9.0


def load_raw() -> pd.DataFrame:
    """
    Load the raw FIRMS CSV file and standardize column names.

    The column names are converted to lowercase and surrounding
    spaces are removed to facilitate subsequent processing.
    """
    if not FIRMS_RAW.exists():
        raise FileNotFoundError(
            f"{FIRMS_RAW} not found.\n"
            "Download the FIRMS data from "
            "https://firms.modaps.eosdis.nasa.gov/download/ "
            "and save the CSV file at this location."
        )

    df = pd.read_csv(FIRMS_RAW, low_memory=False)
    df.columns = [column.strip().lower() for column in df.columns]

    # Columns required by the preprocessing pipeline.
    required_columns = {
        "latitude",
        "longitude",
        "acq_date",
        "frp",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required FIRMS columns: {sorted(missing_columns)}"
        )

    print(f"  Raw records loaded: {len(df):,}")

    return df


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters to FIRMS observations.

    Filters:
        - Remove invalid coordinates.
        - Remove invalid acquisition dates.
        - Keep type 0 (presumed vegetation fires).
        - Keep nominal/high confidence observations.
        - Keep detections inside the study area.
        - Convert FRP to a numeric value.
    """
    df = df.copy()
    n0 = len(df)

    # Convert geographic coordinates to numeric values.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Convert acquisition dates. Invalid dates become NaT.
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    # Remove observations with invalid coordinates or dates.
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    # Keep only detections inside the geographic study area.
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    ]

    print(f"  After coordinate/date validation: {len(df):,}")

    # FIRMS type 0 represents presumed vegetation fires.
    # Other types include volcanoes and other thermal sources.
    if "type" in df.columns:
        df["type"] = pd.to_numeric(df["type"], errors="coerce")
        df = df[df["type"] == 0]
    else:
        print("  WARNING: 'type' column not found; type filter skipped.")

    n1 = len(df)

    # VIIRS confidence values can be l/n/h.
    # Keep nominal and high confidence detections.
    # For numeric confidence values, use a threshold of 50.
    if "confidence" in df.columns:
        confidence_text = (
            df["confidence"].astype(str).str.strip().str.lower()
        )
        confidence_numeric = pd.to_numeric(
            confidence_text, errors="coerce"
        )

        keep_confidence = (
            confidence_numeric.ge(50)
            | confidence_text.isin(
                ["n", "h", "nominal", "high"]
            )
        )

        df = df[keep_confidence]
    else:
        print(
            "  WARNING: 'confidence' column not found; "
            "confidence filter skipped."
        )

    n2 = len(df)

    # FRP (Fire Radiative Power) must be numeric for aggregation.
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

    # Remove observations with missing or negative FRP.
    df = df.dropna(subset=["frp"])
    df = df[df["frp"] >= 0]

    print(f"  Raw detections                  : {n0:,}")
    print(f"  After type filter               : {n1:,}")
    print(f"  After confidence filter         : {n2:,}")
    print(f"  After FRP validation             : {len(df):,}")

    if df.empty:
        raise ValueError(
            "No FIRMS observations remain after quality filtering."
        )

    return df


def snap_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each fire detection to the nearest NASA POWER grid cell.

    LAT_STEP and LON_STEP define the spatial resolution of the grid.
    """
    if LAT_STEP <= 0 or LON_STEP <= 0:
        raise ValueError("Grid steps must be strictly positive.")

    df = df.copy()

    # Round coordinates to the nearest grid-cell center.
    df["lat_cell"] = (
        df["latitude"] / LAT_STEP
    ).round() * LAT_STEP

    df["lon_cell"] = (
        df["longitude"] / LON_STEP
    ).round() * LON_STEP

    # Use the acquisition date as the daily time key.
    df["date"] = df["acq_date"].dt.normalize()

    return df


def collapse_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate detections belonging to the same grid cell and day.

    For each (lat_cell, lon_cell, date):
        - n_detections: number of fire detections.
        - total_frp: sum of Fire Radiative Power.

    Persistent hotspot cells are excluded using the existing
    project rule: active on more than half of the observed days.
    """
    daily = (
        df.groupby(["lat_cell", "lon_cell", "date"])
        .agg(
            n_detections=("latitude", "size"),
            total_frp=("frp", "sum"),
        )
        .reset_index()
    )

    # Identify cells that appear active on more than half
    # of the calendar days covered by the dataset.
    if not daily.empty:
        days_span = (
            daily["date"].max() - daily["date"].min()
        ).days + 1

        per_cell = daily.groupby(
            ["lat_cell", "lon_cell"]
        ).size()

        persistent = per_cell[
            per_cell > 0.5 * days_span
        ].index

        if len(persistent):
            print(
                f"  Dropping {len(persistent)} "
                "persistent hotspot cell(s)."
            )

            daily_index = pd.MultiIndex.from_frame(
                daily[["lat_cell", "lon_cell"]]
            )

            daily = daily[
                ~daily_index.isin(persistent)
            ]

    return daily


def validate_output(daily: pd.DataFrame) -> None:
    """
    Perform quality checks on the final FIRMS daily dataset.

    These checks verify:
        - The dataset is not empty.
        - Required output columns are present.
        - No missing values exist in key columns.
        - Coordinates and dates are valid.
        - There are no duplicate cell/day combinations.
        - Detection counts and FRP values are non-negative.
    """
    required_columns = [
        "lat_cell",
        "lon_cell",
        "date",
        "n_detections",
        "total_frp",
    ]

    missing_columns = set(required_columns) - set(daily.columns)

    if missing_columns:
        raise ValueError(
            f"Missing output columns: {sorted(missing_columns)}"
        )

    if daily.empty:
        raise ValueError("The final FIRMS dataset is empty.")

    if daily[required_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in required output columns."
        )

    if daily.duplicated(
        subset=["lat_cell", "lon_cell", "date"]
    ).any():
        raise ValueError("Duplicate cell/day combinations detected.")

    if (daily["n_detections"] <= 0).any():
        raise ValueError("Invalid detection counts detected.")

    if (daily["total_frp"] < 0).any():
        raise ValueError("Negative FRP values detected.")

    if not daily["date"].map(
        lambda value: isinstance(value, pd.Timestamp)
    ).all():
        raise ValueError("Invalid date values in final dataset.")

    print("\nFIRMS output quality checks:")
    print("  Empty dataset                 : NO")
    print("  Missing values in key columns : NO")
    print("  Duplicate cell/day rows       : NO")
    print("  Invalid detection counts      : NO")
    print("  Negative FRP values           : NO")
    print("  Quality checks                : PASSED")


def main():
    """Execute the complete FIRMS preprocessing pipeline."""

    print("STEP 1 — Loading raw FIRMS data ...")
    df = load_raw()

    print("\nSTEP 2 — Applying quality filters ...")
    df = filter_quality(df)

    print("\nSTEP 3 — Snapping detections to POWER grid cells ...")
    df = snap_to_grid(df)

    print("\nSTEP 4 — Aggregating detections by cell and day ...")
    daily = collapse_to_daily(df)

    print("\nSTEP 5 — Validating the final dataset ...")
    validate_output(daily)

    # Create the output directory if it does not exist.
    FIRES_DAILY.parent.mkdir(parents=True, exist_ok=True)

    # Save the cleaned daily dataset.
    daily.to_parquet(FIRES_DAILY, index=False)

    print(f"\nSaved: {FIRES_DAILY}")
    print(f"Final number of fire-day rows: {len(daily):,}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")


if __name__ == "__main__":
    main()