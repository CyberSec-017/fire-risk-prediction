"""
FIREWATCH AI — Cartographic and Geospatial Map Builders (Folium).

Generates high-performance interactive maps for wildfire monitoring
with custom scientific styling, legends, and inspection popups.
"""

from __future__ import annotations

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from dashboard.config import (
    COLORS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_ZOOM,
)


def _get_fire_color(days: int) -> str:
    """Return scientific risk color based on frequency of active fire days."""
    if days <= 4:
        return COLORS["risk_low"]
    elif days <= 12:
        return COLORS["risk_moderate"]
    elif days <= 25:
        return COLORS["risk_high"]
    else:
        return COLORS["risk_very_high"]


def create_base_map(
    zoom: int = MAP_ZOOM,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a clean base map centered over northern Algeria."""
    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=zoom,
        tiles=tiles,
        control_scale=True,
        prefer_canvas=True,
    )

    # Draw study area bounding box
    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        color=COLORS["brand_primary"],
        weight=1.5,
        fill=False,
        dash_array="6 4",
        popup="Study Domain: 32.0°–37.5°N, -2.0°–9.0°E",
        tooltip="Project Geographic Extent (Northern Algeria)",
    ).add_to(m)

    return m


def render_grid_detection_map(
    df: pd.DataFrame,
    max_cells: int = 1500,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Render aggregated fire occurrences per NASA POWER grid cell."""
    m = create_base_map(tiles=tiles)

    if df.empty:
        return m

    # Aggregate by spatial coordinates
    cell_agg = (
        df.groupby(["lat_cell", "lon_cell"])
        .agg(
            fire_days=("date", "nunique"),
            total_detections=("n_detections", "sum") if "n_detections" in df.columns else ("date", "size"),
            total_frp=("total_frp", "sum") if "total_frp" in df.columns else ("date", "size"),
            first_seen=("date", "min"),
            last_seen=("date", "max"),
        )
        .reset_index()
    )

    if len(cell_agg) > max_cells:
        cell_agg = cell_agg.nlargest(max_cells, "fire_days")

    for _, r in cell_agg.iterrows():
        days = int(r["fire_days"])
        color = _get_fire_color(days)
        radius = max(3.5, min(days * 0.7 + 3, 16))

        first_str = r["first_seen"].strftime("%Y-%m-%d") if hasattr(r["first_seen"], "strftime") else str(r["first_seen"])
        last_str = r["last_seen"].strftime("%Y-%m-%d") if hasattr(r["last_seen"], "strftime") else str(r["last_seen"])
        detections = int(r["total_detections"])
        frp_str = f"{r['total_frp']:,.1f} MW" if r["total_frp"] > 0 else "N/A"

        popup_content = f"""
        <div style="font-family:Inter,sans-serif; font-size:12px; color:{COLORS['text_primary']}; min-width:190px;">
            <div style="font-weight:600; color:{color}; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                Grid Cell ({r['lat_cell']:.2f}°N, {r['lon_cell']:.2f}°E)
            </div>
            <table style="width:100%; border-collapse:collapse; font-size:11px;">
                <tr><td style="color:{COLORS['text_muted']}; padding:2px 0;">Fire Days:</td><td style="font-weight:600; text-align:right;">{days}</td></tr>
                <tr><td style="color:{COLORS['text_muted']}; padding:2px 0;">FIRMS Hits:</td><td style="font-weight:600; text-align:right;">{detections:,}</td></tr>
                <tr><td style="color:{COLORS['text_muted']}; padding:2px 0;">Total FRP:</td><td style="font-weight:600; text-align:right;">{frp_str}</td></tr>
                <tr><td style="color:{COLORS['text_muted']}; padding:2px 0;">First Active:</td><td style="text-align:right;">{first_str}</td></tr>
                <tr><td style="color:{COLORS['text_muted']}; padding:2px 0;">Last Active:</td><td style="text-align:right;">{last_str}</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[r["lat_cell"], r["lon_cell"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            weight=1,
            popup=folium.Popup(popup_content, max_width=240),
            tooltip=f"{r['lat_cell']:.2f}°N, {r['lon_cell']:.2f}°E • {days} fire-days",
        ).add_to(m)

    _inject_clean_legend(m)
    return m


def render_density_heatmap(
    df: pd.DataFrame,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Render a kernel density heatmap of cumulative fire activity."""
    m = create_base_map(tiles=tiles)

    if df.empty:
        return m

    fire_data = df[df["burned"] == 1] if "burned" in df.columns else df
    counts = fire_data.groupby(["lat_cell", "lon_cell"]).size().reset_index(name="count")

    if counts.empty:
        return m

    points = counts[["lat_cell", "lon_cell", "count"]].values.tolist()

    HeatMap(
        points,
        min_opacity=0.25,
        max_zoom=9,
        radius=16,
        blur=14,
        gradient={
            "0.2": COLORS["brand_accent"],
            "0.5": COLORS["risk_moderate"],
            "0.75": COLORS["risk_high"],
            "1.0": COLORS["risk_very_high"],
        },
    ).add_to(m)

    _inject_clean_legend(m)
    return m


def render_cluster_map(
    df: pd.DataFrame,
    max_samples: int = 2500,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Render incident markers grouped dynamically via MarkerCluster."""
    m = create_base_map(tiles=tiles)

    if df.empty:
        return m

    fire_data = df[df["burned"] == 1] if "burned" in df.columns else df
    if len(fire_data) > max_samples:
        fire_data = fire_data.sample(max_samples, random_state=42)

    cluster = MarkerCluster(name="Fire Incidents").add_to(m)

    for _, row in fire_data.iterrows():
        d_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        folium.CircleMarker(
            location=[row["lat_cell"], row["lon_cell"]],
            radius=5,
            color=COLORS["risk_very_high"],
            fill=True,
            fill_color=COLORS["risk_very_high"],
            fill_opacity=0.75,
            weight=1,
            tooltip=f"{d_str} ({row['lat_cell']:.2f}°N, {row['lon_cell']:.2f}°E)",
        ).add_to(cluster)

    return m


def _inject_clean_legend(m: folium.Map) -> None:
    """Add a professional floating legend to the Folium map."""
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 1000;
        background-color: {COLORS['surface_white']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 6px;
        padding: 10px 14px;
        font-family: Inter, sans-serif;
        font-size: 11px;
        color: {COLORS['text_secondary']};
        box-shadow: 0 1px 4px rgba(15,23,42,0.08);
    ">
        <div style="font-weight: 600; color:{COLORS['text_primary']}; margin-bottom: 6px;">
            Fire Activity Density
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
            <div style="width:10px; height:10px; border-radius:50%; background-color:{COLORS['risk_low']};"></div>
            <span>1 – 4 fire-days (Low)</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
            <div style="width:10px; height:10px; border-radius:50%; background-color:{COLORS['risk_moderate']};"></div>
            <span>5 – 12 fire-days (Moderate)</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
            <div style="width:10px; height:10px; border-radius:50%; background-color:{COLORS['risk_high']};"></div>
            <span>13 – 25 fire-days (High)</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
            <div style="width:10px; height:10px; border-radius:50%; background-color:{COLORS['risk_very_high']};"></div>
            <span>> 25 fire-days (Very High)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
