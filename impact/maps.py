"""Folium maps for the Wildfire Impact Assessment module.

The maps reuse the dashboard's basemap, risk palette and legend helpers by
importing ``dashboard.maps`` rather than re-implementing them.  They only
consume already-prepared impact frames; no I/O, Streamlit or model calls happen
here.

Two views are provided:

* :func:`exposure_choropleth` colours grid cells by a potential-exposure
  metric (IEI by default);
* :func:`risk_exposure_map` combines the model's predicted risk band with the
  static exposure tier of the same cell.

Neither map shows confirmed burned area; that layer is unavailable until an
independent burned-area product is ingested in phase 2.
"""

from __future__ import annotations

import math
from typing import Any

import folium
import numpy as np
import pandas as pd

from dashboard import maps as dashboard_maps
from dashboard.config import COLORS

from impact.exposure import cell_bounds
from impact.severity import normalize_risk_band


_EXPOSURE_PALETTE = (
    COLORS["brand_subtle"],
    COLORS["risk_low"],
    COLORS["risk_moderate"],
    COLORS["risk_high"],
    COLORS["risk_very_high"],
)

_EXPOSURE_DISPLAY = {
    "Very Low": "Very Low",
    "Low": "Low",
    "Moderate": "Moderate",
    "High": "High",
    "Very High": "Very High",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _value_text(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(number):
        return "Not available"
    return f"{number:,.{digits}f}{suffix}"


def _is_true(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _popup(title: str, rows: list[tuple[str, str]]) -> folium.Popup:
    body = "".join(
        f"<tr><td style='color:{COLORS['text_muted']}; padding:2px 0;'>{label}</td>"
        f"<td style='font-weight:600; text-align:right;'>{value}</td></tr>"
        for label, value in rows
    )
    content = f"""
    <div style="font-family:Inter,sans-serif; font-size:11px;
                color:{COLORS['text_primary']}; min-width:210px;">
        <div style="font-weight:600; border-bottom:1px solid #E2E8F0;
                    padding-bottom:4px; margin-bottom:6px;">{title}</div>
        <table style="width:100%; border-collapse:collapse;">{body}</table>
    </div>
    """
    return folium.Popup(content, max_width=280)


def _empty_map(message: str, title: str) -> folium.Map:
    m = dashboard_maps.create_base_map()
    dashboard_maps._inject_clean_legend(
        m,
        title=title,
        entries=[(COLORS["text_muted"], message)],
    )
    return m


def _bounded_frame(assessed: pd.DataFrame | None, metric: str | None) -> pd.DataFrame:
    if (
        assessed is None
        or not isinstance(assessed, pd.DataFrame)
        or assessed.empty
        or "lat_cell" not in assessed.columns
        or "lon_cell" not in assessed.columns
    ):
        return pd.DataFrame()
    frame = assessed.copy()
    frame["lat_cell"] = pd.to_numeric(frame["lat_cell"], errors="coerce")
    frame["lon_cell"] = pd.to_numeric(frame["lon_cell"], errors="coerce")
    frame = frame[frame["lat_cell"].notna() & frame["lon_cell"].notna()]
    if metric is not None:
        if metric not in frame.columns:
            return pd.DataFrame()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame[frame[metric].notna() & np.isfinite(frame[metric])]
    return frame


# ---------------------------------------------------------------------------
# Exposure choropleth
# ---------------------------------------------------------------------------
def exposure_choropleth(
    assessed: pd.DataFrame | None,
    metric: str = "iei",
    metric_label: str = "Impact Exposure Index",
    tiles: str = "Light",
    max_cells: int = 1500,
) -> folium.Map:
    """Colour each grid cell by a potential-exposure metric.

    Exposure is a present-day static snapshot.  A dark cell means more mapped
    assets or people inside it, not that a fire occurred there.
    """
    title = f"Potential exposure — {metric_label}"
    frame = _bounded_frame(assessed, metric)
    if frame.empty:
        return _empty_map("Not available: no exposure values were supplied.", title)

    if len(frame) > max_cells:
        frame = frame.nlargest(max_cells, metric)

    values = frame[metric].to_numpy(dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return _empty_map("Not available: exposure values are not finite.", title)

    if vmax <= vmin:
        edges = np.array([vmin, vmax], dtype=float)
    else:
        quantiles = np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges = np.unique(quantiles)
        if edges.size < 2:
            edges = np.array([vmin, vmax], dtype=float)

    def bucket(value: float) -> int:
        if edges.size <= 1:
            return 0
        index = int(np.searchsorted(edges, value, side="right") - 1)
        return max(0, min(index, len(_EXPOSURE_PALETTE) - 1))

    m = dashboard_maps.create_base_map(tiles=tiles)
    for _, row in frame.iterrows():
        value = float(row[metric])
        color = _EXPOSURE_PALETTE[bucket(value)]
        lat_min, lat_max, lon_min, lon_max = cell_bounds(
            float(row["lat_cell"]), float(row["lon_cell"])
        )
        tier = str(row.get("exposure_tier", "Not available"))
        probability = row.get("probability")
        rows = [
            (f"{metric_label}", _value_text(value, digits=3)),
            ("Exposure tier", _EXPOSURE_DISPLAY.get(tier, tier)),
            ("Predicted fire risk", _value_text(probability, digits=4)),
        ]
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color=COLORS["surface_white"],
            weight=0.6,
            fill=True,
            fill_color=color,
            fill_opacity=0.72,
            popup=_popup(
                f"Cell {float(row['lat_cell']):.2f}°N, "
                f"{float(row['lon_cell']):.2f}°E",
                rows,
            ),
            tooltip=(
                f"{float(row['lat_cell']):.2f}°N, {float(row['lon_cell']):.2f}°E"
                f" • {metric_label}: {_value_text(value, digits=3)}"
            ),
        ).add_to(m)

    entries = []
    for index, color in enumerate(_EXPOSURE_PALETTE):
        if index < edges.size - 1:
            label = f"{edges[index]:,.3f} – {edges[index + 1]:,.3f}"
        elif index == 0:
            label = _value_text(edges[0], digits=3)
        else:
            break
        entries.append((color, label))
    dashboard_maps._inject_clean_legend(m, title=title, entries=entries)
    return m


# ---------------------------------------------------------------------------
# Risk vs exposure
# ---------------------------------------------------------------------------
def risk_exposure_map(
    assessed: pd.DataFrame | None,
    card: dict | None = None,
    tiles: str = "Light",
    max_cells: int = 1500,
) -> folium.Map:
    """Show predicted risk by cell, shaded by potential exposure.

    The cell colour is the model risk band.  Fill opacity rises with the
    static exposure tier so that high-risk, high-exposure cells stand out.
    This is a prioritisation view, not a damage forecast.
    """
    title = "Predicted fire risk vs. potential exposure"
    frame = _bounded_frame(assessed, None)
    if frame.empty:
        return _empty_map("Not available: no risk scores were supplied.", title)

    if len(frame) > max_cells:
        sort_column = "iei" if "iei" in frame.columns else "probability"
        if sort_column in frame.columns:
            frame[sort_column] = pd.to_numeric(frame[sort_column], errors="coerce")
            frame = frame.nlargest(max_cells, sort_column)
        else:
            frame = frame.head(max_cells)

    m = dashboard_maps.create_base_map(tiles=tiles)
    seen_bands: set[str] = set()
    outside_seen = False

    for _, row in frame.iterrows():
        band = normalize_risk_band(row.get("risk"))
        probability = row.get("probability")
        in_area = _is_true(row.get("in_model_area"))
        outside = band is None or not in_area

        if outside:
            color = dashboard_maps._OUTSIDE_COLOR
            opacity = 0.28
            outside_seen = True
            legend_band = "Outside"
        else:
            color = dashboard_maps._RISK_COLORS.get(band, COLORS["brand_primary"])
            tier = str(row.get("exposure_tier", ""))
            tier_rank = {
                "Very Low": 0,
                "Low": 1,
                "Moderate": 2,
                "High": 3,
                "Very High": 4,
            }.get(tier, 1)
            opacity = 0.38 + 0.12 * tier_rank
            opacity = min(opacity, 0.88)
            seen_bands.add(band)
            legend_band = band

        lat_min, lat_max, lon_min, lon_max = cell_bounds(
            float(row["lat_cell"]), float(row["lon_cell"])
        )
        rows = [
            (
                "Predicted fire risk",
                _value_text(probability, digits=4) if probability is not None else "Not scored",
            ),
            ("Risk band", dashboard_maps._RISK_DISPLAY.get(legend_band, legend_band.title())),
            ("Potential exposure (IEI)", _value_text(row.get("iei"), digits=3)),
            ("Exposure tier", str(row.get("exposure_tier", "Not available"))),
            ("Impact priority", str(row.get("impact_priority_label", "Not applicable"))),
            (
                "Observed detection",
                "Yes (not burned area)"
                if _value_text(row.get("burned"), digits=0) == "1"
                else "No",
            ),
        ]
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color=COLORS["surface_white"],
            weight=0.6,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            popup=_popup(
                f"Cell {float(row['lat_cell']):.2f}°N, "
                f"{float(row['lon_cell']):.2f}°E",
                rows,
            ),
            tooltip=(
                f"{float(row['lat_cell']):.2f}°N, {float(row['lon_cell']):.2f}°E"
                f" • predicted risk: {_value_text(probability, digits=4)}"
                f" • potential exposure IEI: {_value_text(row.get('iei'), digits=3)}"
            ),
        ).add_to(m)

    entries = [
        (
            dashboard_maps._RISK_COLORS.get(band, COLORS["brand_primary"]),
            f"{dashboard_maps._RISK_DISPLAY.get(band, band.title())} predicted risk",
        )
        for band in ("LOW", "MODERATE", "HIGH", "VERY HIGH")
        if band in seen_bands
    ]
    if outside_seen:
        entries.append(
            (dashboard_maps._OUTSIDE_COLOR, dashboard_maps._OUTSIDE_LEGEND)
        )
    if not entries:
        entries = [(COLORS["text_muted"], "Not available")]
    dashboard_maps._inject_clean_legend(m, title=title, entries=entries)
    return m


__all__ = [
    "exposure_choropleth",
    "risk_exposure_map",
]
