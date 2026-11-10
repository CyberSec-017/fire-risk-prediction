"""
FIREWATCH AI — Data loading with Streamlit caching.

All heavy I/O is wrapped in @st.cache_data so that parquet files
are read only once per session (or until the file changes).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import (
    FIRE_PANEL_PATH,
    FIRES_DAILY_PATH,
)


@st.cache_data(show_spinner="Loading fire panel data …")
def load_fire_panel() -> pd.DataFrame | None:
    """Load the main fire_panel.parquet dataset.

    Returns None if the file does not exist yet.
    """
    if not FIRE_PANEL_PATH.exists():
        return None

    df = pd.read_parquet(FIRE_PANEL_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Ensure burned is int
    if "burned" in df.columns:
        df["burned"] = df["burned"].astype(int)

    return df


@st.cache_data(show_spinner="Loading fire detections …")
def load_fires_daily() -> pd.DataFrame | None:
    """Load the cleaned daily fire detections from fires_daily.parquet.

    Returns None if the file does not exist yet.
    """
    if not FIRES_DAILY_PATH.exists():
        return None

    df = pd.read_parquet(FIRES_DAILY_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_date_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) from the 'date' column."""
    return df["date"].min(), df["date"].max()


def filter_by_date(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Filter dataframe to rows within [start, end]."""
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask]


def get_unique_years(df: pd.DataFrame) -> list[int]:
    """Return sorted list of unique years in the dataset."""
    return sorted(df["date"].dt.year.dropna().unique().tolist())

