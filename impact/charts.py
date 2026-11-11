"""Plotly views for the Wildfire Impact Assessment module.

The charts reuse the dashboard's scientific design system by importing
``dashboard.charts`` rather than duplicating its theme.  They only consume
already-prepared impact frames; no I/O, Streamlit or model calls happen here.

Every chart distinguishes the three evidence layers it actually uses:
predicted risk, potential exposure and observed detections.  Confirmed burned
area is deliberately absent because no burned-area product is ingested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard import charts as dashboard_charts
from dashboard.config import COLORS

from impact import config
from impact.severity import normalize_risk_band


_LANDCOVER_COLORS = {
    "tree": "#166534",
    "shrub": "#4D7C0F",
    "grass": "#CA8A04",
    "crop": "#D97706",
    "bare": "#A8A29E",
    "built": "#475569",
    "water": "#0369A1",
    "other": "#CBD5E1",
}


def _empty_figure(title: str, subtitle: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color=COLORS["text_muted"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return dashboard_charts._apply_theme(fig, title=title, subtitle=subtitle)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _finite_frame(frame: pd.DataFrame | None, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=list(columns))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=list(columns))
    out = frame.copy()
    mask = pd.Series(True, index=out.index)
    for column in columns:
        values = _numeric(out, column)
        out[column] = values
        mask &= values.notna() & np.isfinite(values)
    return out.loc[mask].copy()


# ---------------------------------------------------------------------------
# Risk vs exposure
# ---------------------------------------------------------------------------
def risk_exposure_scatter(
    assessed: pd.DataFrame | None,
    card: dict | None = None,
) -> go.Figure:
    """Scatter predicted risk against potential exposure for each cell.

    The x axis is the model probability that a thermal detection occurs in the
    cell within three days.  The y axis is the static Impact Exposure Index.
    Neither axis is a damage or burned-area estimate.
    """
    frame = _finite_frame(assessed, ("probability", "iei"))
    if frame.empty:
        return _empty_figure(
            "Predicted risk vs. potential exposure",
            "Model probability against static cell exposure",
            "Not available: predicted risk and exposure must both be present.",
        )

    frame["risk_band"] = (
        frame["risk"].map(normalize_risk_band)
        if "risk" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    band_colors = {
        "LOW": COLORS["risk_low"],
        "MODERATE": COLORS["risk_moderate"],
        "HIGH": COLORS["risk_high"],
        "VERY HIGH": COLORS["risk_very_high"],
    }

    fig = go.Figure()
    for band in ("LOW", "MODERATE", "HIGH", "VERY HIGH"):
        subset = frame[frame["risk_band"] == band]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=subset["probability"] * 100.0,
                y=subset["iei"],
                mode="markers",
                name=band.title(),
                marker=dict(
                    size=9,
                    color=band_colors[band],
                    opacity=0.8,
                    line=dict(color="#FFFFFF", width=0.8),
                ),
                customdata=np.column_stack(
                    [
                        subset.get("lat_cell", pd.Series([None] * len(subset))),
                        subset.get("lon_cell", pd.Series([None] * len(subset))),
                        subset.get("exposure_tier", pd.Series([None] * len(subset))),
                    ]
                ),
                hovertemplate=(
                    "<b>Predicted risk:</b> %{x:.2f}%<br>"
                    "<b>Potential exposure (IEI):</b> %{y:.3f}<br>"
                    "<b>Cell:</b> %{customdata[0]:.2f}°N, "
                    "%{customdata[1]:.2f}°E<br>"
                    "<b>Exposure tier:</b> %{customdata[2]}<extra></extra>"
                ),
            )
        )

    unknown = frame[frame["risk_band"].isna()]
    if not unknown.empty:
        fig.add_trace(
            go.Scattergl(
                x=unknown["probability"] * 100.0,
                y=unknown["iei"],
                mode="markers",
                name="Outside model area",
                marker=dict(size=8, color=COLORS["text_muted"], opacity=0.55),
                hovertemplate=(
                    "Predicted risk: not scored<br>"
                    "Potential exposure (IEI): %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title_text="Predicted fire risk — model probability (%)")
    fig.update_yaxes(title_text="Potential exposure — IEI (0–1)")
    return dashboard_charts._apply_theme(
        fig,
        title="Predicted risk vs. potential exposure",
        subtitle=(
            "Model probability (3-day thermal-detection target) against static "
            "exposure; neither axis is damage or burned area"
        ),
    )


# ---------------------------------------------------------------------------
# Land-cover composition
# ---------------------------------------------------------------------------
def landcover_composition(assessed: pd.DataFrame | None) -> go.Figure:
    """Stacked land-cover composition of the assessed cells.

    Values are present-day ESA WorldCover areas.  They describe potential
    exposure, not the vegetation that burned in a historical event.
    """
    columns = [
        (key, column)
        for key, column in config.LC_AREA_COLUMNS.items()
        if assessed is not None
        and isinstance(assessed, pd.DataFrame)
        and column in assessed.columns
    ]
    if not columns:
        return _empty_figure(
            "Land-cover composition",
            "Potential exposure by ESA WorldCover class",
            "Not available: no land-cover area columns were supplied.",
        )

    totals: list[tuple[str, str, float]] = []
    for key, column in columns:
        values = pd.to_numeric(assessed[column], errors="coerce")
        total = float(values.sum(min_count=1)) if values.notna().any() else np.nan
        if not np.isfinite(total) or total <= 0:
            continue
        label = next(
            (
                name
                for class_key, name in config.WORLDCOVER_CLASSES.values()
                if class_key == key
            ),
            key.title(),
        )
        totals.append((key, label, total))

    if not totals:
        return _empty_figure(
            "Land-cover composition",
            "Potential exposure by ESA WorldCover class",
            "Not available: land-cover areas are zero or missing.",
        )

    fig = go.Figure()
    for key, label, value in totals:
        fig.add_trace(
            go.Bar(
                y=["Assessed cells"],
                x=[value],
                orientation="h",
                name=label,
                marker=dict(
                    color=_LANDCOVER_COLORS.get(key, _LANDCOVER_COLORS["other"]),
                    line=dict(color="#FFFFFF", width=0.6),
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>Area: %{{x:,.1f}} km²<extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Land-cover area in assessed cells (km²)")
    fig.update_yaxes(title_text="")
    return dashboard_charts._apply_theme(
        fig,
        title="Land-cover composition",
        subtitle=(
            "Potential exposure — present-day ESA WorldCover classes, not "
            "burned area"
        ),
    )


# ---------------------------------------------------------------------------
# Regional comparison
# ---------------------------------------------------------------------------
def _latitude_band(latitude: float) -> str:
    lower = int(np.floor(float(latitude)))
    return f"{lower}.0–{lower + 1}.0°N"


def regional_comparison(
    assessed: pd.DataFrame | None,
    region_column: str = "admin_name",
) -> go.Figure:
    """Compare mean predicted risk and mean potential exposure by region.

    Uses ``admin_name`` when the table provides it, otherwise a one-degree
    latitude band.  The two measures keep separate axes because they are not
    on the same scale.
    """
    if (
        assessed is None
        or not isinstance(assessed, pd.DataFrame)
        or assessed.empty
        or "lat_cell" not in assessed.columns
    ):
        return _empty_figure(
            "Regional comparison",
            "Predicted risk and potential exposure by region",
            "Not available: no assessed cells were supplied.",
        )

    frame = assessed.copy()
    if region_column in frame.columns:
        frame["region"] = frame[region_column].astype("string")
        frame["region"] = frame["region"].where(
            frame["region"].notna() & frame["region"].str.len().gt(0),
            "Unassigned",
        )
        region_label = "Administrative region"
    else:
        lat = pd.to_numeric(frame["lat_cell"], errors="coerce")
        frame["region"] = lat.map(
            lambda value: _latitude_band(value) if pd.notna(value) else "Unknown"
        )
        region_label = "Latitude band"

    numeric_columns = [column for column in ("probability", "iei", "pop_cell") if column in frame.columns]
    if not numeric_columns:
        return _empty_figure(
            "Regional comparison",
            "Predicted risk and potential exposure by region",
            "Not available: no comparable risk or exposure columns were supplied.",
        )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = frame.groupby("region", dropna=False)
    rows = []
    for region, subset in grouped:
        probability = (
            float(subset["probability"].mean())
            if "probability" in subset.columns
            and subset["probability"].notna().any()
            else np.nan
        )
        iei = (
            float(subset["iei"].mean())
            if "iei" in subset.columns and subset["iei"].notna().any()
            else np.nan
        )
        population = (
            float(subset["pop_cell"].sum())
            if "pop_cell" in subset.columns and subset["pop_cell"].notna().any()
            else np.nan
        )
        rows.append(
            {
                "region": str(region),
                "probability": probability,
                "iei": iei,
                "population": population,
                "cells": int(len(subset)),
            }
        )

    if not rows:
        return _empty_figure(
            "Regional comparison",
            "Predicted risk and potential exposure by region",
            "Not available: no regions could be compared.",
        )

    comparison = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.16,
        subplot_titles=(
            "Predicted fire risk (mean probability, %)",
            "Potential exposure (mean IEI, 0–1)",
        ),
    )

    fig.add_trace(
        go.Bar(
            y=comparison["region"],
            x=(comparison["probability"] * 100.0).round(3),
            orientation="h",
            name="Predicted risk",
            marker_color=COLORS["risk_high"],
            customdata=np.column_stack(
                [
                    comparison["cells"],
                    comparison["population"].round(0),
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Mean probability: %{x:.2f}%<br>"
                "Cells: %{customdata[0]}<br>"
                "Potential population: %{customdata[1]:,.0f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=comparison["region"],
            x=comparison["iei"].round(4),
            orientation="h",
            name="Potential exposure",
            marker_color=COLORS["brand_primary"],
            customdata=np.column_stack(
                [
                    comparison["cells"],
                    comparison["population"].round(0),
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Mean IEI: %{x:.3f}<br>"
                "Cells: %{customdata[0]}<br>"
                "Potential population: %{customdata[1]:,.0f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Mean probability (%)", row=1, col=1)
    fig.update_xaxes(title_text="Mean IEI", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=1)
    return dashboard_charts._apply_theme(
        fig,
        title="Regional comparison",
        subtitle=(
            f"{region_label}: predicted risk is date-specific; exposure is a "
            "present-day static snapshot"
        ),
        height=440,
    )


__all__ = [
    "risk_exposure_scatter",
    "landcover_composition",
    "regional_comparison",
]
