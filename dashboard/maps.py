"""
FIREWATCH AI — Cartographic and Geospatial Map Builders (Folium).

Generates high-performance interactive maps for wildfire monitoring
with custom scientific styling, legends, and inspection popups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

_DEFAULT_RISK_BINS = [0.0, 0.01, 0.05, 0.15, 1.0]
_DEFAULT_RISK_LABELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
_RISK_COLORS = {
    "LOW": COLORS["risk_low"],
    "MODERATE": COLORS["risk_moderate"],
    "HIGH": COLORS["risk_high"],
    "VERY HIGH": COLORS["risk_very_high"],
}
_RISK_DISPLAY = {
    "LOW": "Low",
    "MODERATE": "Moderate",
    "HIGH": "High",
    "VERY HIGH": "Very High",
}

# Keyless basemaps.  The tile names are the user-facing selectbox labels.
_ESRI_ATTR = "Tiles © Esri"
_ESRI_LIGHT = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_DARK = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
_TILE_LAYERS = {
    "Light": (_ESRI_LIGHT, _ESRI_ATTR, "Light"),
    "Dark": (_ESRI_DARK, _ESRI_ATTR, "Dark"),
    "OpenStreetMap": ("OpenStreetMap", None, "OpenStreetMap"),
}
_DEFAULT_TILE = "Light"

_OUTSIDE_RISK = "OUTSIDE"
_OUTSIDE_COLOR = COLORS["text_muted"]
_OUTSIDE_TOOLTIP = "Outside the model's study area (no prediction)"
_OUTSIDE_LEGEND = "Outside the model's study area (no prediction)"


def _build_tile_layer(tiles: str) -> folium.TileLayer:
    """Return a keyless tile layer for a user-facing basemap name."""
    template, attr, name = _TILE_LAYERS.get(
        tiles, _TILE_LAYERS[_DEFAULT_TILE]
    )
    return folium.TileLayer(
        tiles=template,
        attr=attr,
        name=name,
        no_wrap=True,
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
    tiles: str = "Light",
) -> folium.Map:
    """Create a clean base map centered over northern Algeria."""
    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    _build_tile_layer(tiles).add_to(m)

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


def _selection_marker_html(color: str) -> str:
    """Return a self-contained SVG map pin (no external icon font needed)."""
    return (
        '<div style="line-height:0;">'
        f'<svg width="30" height="40" viewBox="0 0 30 40" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M15 1C7.3 1 1 7.3 1 15c0 10.4 14 24 14 24s14-13.6 14-24'
        f'C29 7.3 22.7 1 15 1z" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>'
        '<circle cx="15" cy="15" r="5.2" fill="#FFFFFF"/>'
        "</svg></div>"
    )


def _selection_marker(
    selected: tuple[float, float],
    *,
    in_area: bool = True,
) -> folium.Marker:
    """Build the single selected-location marker."""
    lat, lon = float(selected[0]), float(selected[1])
    color = COLORS["risk_very_high"] if in_area else COLORS["text_muted"]
    suffix = "" if in_area else " • outside the study area"
    return folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=_selection_marker_html(color),
            icon_size=(30, 40),
            icon_anchor=(15, 40),
            class_name="firewatch-selection-pin",
        ),
        tooltip=f"Selected: {lat:.4f}°N, {lon:.4f}°E{suffix}",
    )


def selection_marker_group(
    selected: tuple[float, float] | None = None,
    *,
    in_area: bool = True,
) -> folium.FeatureGroup | None:
    """Return the selected-location marker as a dynamic feature group.

    streamlit-folium derives the component identity from a hash of the base
    map's Leaflet JavaScript.  Keeping the marker out of that script (and in a
    feature group added on the client) means selecting a new point updates only
    the marker instead of remounting the map at its default centre and zoom.
    """
    if selected is None:
        return None
    group = folium.FeatureGroup(name="Selected location")
    _selection_marker(selected, in_area=in_area).add_to(group)
    return group


def viewport_from_map_data(
    map_data: Mapping[str, object] | None,
) -> dict | None:
    """Normalise the viewport returned by ``st_folium`` for session state.

    The installed streamlit-folium version returns ``center``, ``zoom`` and
    ``bounds`` when they are requested through ``returned_objects``.  Values are
    converted to plain JSON-friendly types so they survive Streamlit reruns.
    """
    if not map_data:
        return None

    center = map_data.get("center")
    zoom = map_data.get("zoom")
    if not isinstance(center, Mapping):
        return None

    lat = center.get("lat")
    lng = center.get("lng", center.get("lon"))
    if lat is None or lng is None or zoom is None:
        return None

    viewport: dict = {
        "center": {"lat": float(lat), "lng": float(lng)},
        "zoom": int(round(float(zoom))),
    }
    bounds = map_data.get("bounds")
    if isinstance(bounds, Mapping):
        viewport["bounds"] = dict(bounds)
    return viewport


def render_selection_map(
    selected: tuple[float, float] | None = None,
    *,
    tiles: str = "Light",
    in_area: bool = True,
    polygon: Sequence[tuple[float, float]] | None = None,
) -> folium.Map:
    """Render the click-to-select study-area map.

    The base map keeps the existing northern-Algeria centre, zoom and model
    bounding rectangle.  The Algeria boundary is drawn as the selectable area,
    and at most one marker is placed at the selected point.  No risk circles
    are drawn here.
    """
    m = create_base_map(tiles=tiles)

    if polygon:
        folium.Polygon(
            locations=[(float(lat), float(lon)) for lon, lat in polygon],
            color=COLORS["brand_accent"],
            weight=1.4,
            fill=True,
            fill_color=COLORS["brand_accent"],
            fill_opacity=0.05,
            tooltip="Selectable study area: Algerian land inside the model domain",
        ).add_to(m)

    if selected is not None:
        _selection_marker(selected, in_area=in_area).add_to(m)

    return m


def render_grid_detection_map(
    df: pd.DataFrame,
    max_cells: int = 1500,
    tiles: str = "Light",
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
    tiles: str = "Light",
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
    tiles: str = "Light",
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


def _risk_label(probability: float, card: dict | None) -> str:
    """Return the card's risk label for one probability."""
    source = card or {}
    bins = [float(edge) for edge in source.get("risk_bins", _DEFAULT_RISK_BINS)]
    labels = [
        str(label).upper()
        for label in source.get("risk_labels", _DEFAULT_RISK_LABELS)
    ]
    index = int(np.searchsorted(np.asarray(bins), float(probability), side="left") - 1)
    index = max(0, min(index, len(labels) - 1))
    return labels[index]


def _risk_legend_entries(card: dict | None) -> list[tuple[str, str]]:
    """Build legend entries from the model card's bins and labels."""
    source = card or {}
    bins = [float(edge) for edge in source.get("risk_bins", _DEFAULT_RISK_BINS)]
    labels = [
        str(label).upper()
        for label in source.get("risk_labels", _DEFAULT_RISK_LABELS)
    ]

    entries: list[tuple[str, str]] = []
    for index, label in enumerate(labels):
        low = bins[index] if index < len(bins) else 0.0
        high = bins[index + 1] if index + 1 < len(bins) else 1.0
        if index == len(labels) - 1:
            range_text = f"{low * 100:.1f}%+"
        else:
            range_text = f"{low * 100:.1f}% – {high * 100:.1f}%"
        entries.append((
            _RISK_COLORS.get(label, COLORS["brand_primary"]),
            f"{_RISK_DISPLAY.get(label, label.title())} ({range_text})",
        ))
    entries.append((_OUTSIDE_COLOR, _OUTSIDE_LEGEND))
    return entries


def render_risk_map(
    scores: pd.DataFrame,
    card: dict | None = None,
    tiles: str = "Light",
    show_observed: bool = False,
) -> folium.Map:
    """Render per-cell predicted fire risk using the model-card bands."""
    m = create_base_map(tiles=tiles)
    legend_entries = _risk_legend_entries(card)

    if scores is None or scores.empty:
        _inject_clean_legend(
            m, title="Predicted Fire Risk", entries=legend_entries
        )
        return m

    required = {"lat_cell", "lon_cell", "probability"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Risk scores are missing column(s): {missing}.")

    frame = scores.copy()
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    frame = frame.dropna(subset=["lat_cell", "lon_cell"])

    # Cells that are not on the model's training grid are never scored or
    # coloured by risk; they are drawn neutral and excluded from the legend's
    # risk bands.
    outside_flags = frame["probability"].isna()
    if "in_model_area" in frame.columns:
        outside_flags = outside_flags | ~frame["in_model_area"].fillna(False).astype(bool)
    if "risk" in frame.columns:
        outside_flags = outside_flags | (
            frame["risk"].astype(str).str.upper() == _OUTSIDE_RISK
        )

    for is_outside, (_, row) in zip(outside_flags.to_numpy(), frame.iterrows()):
        lat_cell = float(row["lat_cell"])
        lon_cell = float(row["lon_cell"])

        if is_outside:
            folium.CircleMarker(
                location=[lat_cell, lon_cell],
                radius=6,
                color=_OUTSIDE_COLOR,
                fill=True,
                fill_color=_OUTSIDE_COLOR,
                fill_opacity=0.55,
                weight=1,
                tooltip=_OUTSIDE_TOOLTIP,
            ).add_to(m)
            continue

        probability = float(row["probability"])
        if "risk" in frame.columns and pd.notna(row["risk"]):
            band = str(row["risk"]).upper()
        else:
            band = _risk_label(probability, card)
        color = _RISK_COLORS.get(band, COLORS["brand_primary"])
        display = _RISK_DISPLAY.get(band, band.title())

        folium.CircleMarker(
            location=[lat_cell, lon_cell],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.68,
            weight=1,
            tooltip=(
                f"{lat_cell:.2f}°N, {lon_cell:.2f}°E • "
                f"{probability * 100:.1f}% • {display}"
            ),
        ).add_to(m)

    if show_observed and "burned" in frame.columns:
        burned = pd.to_numeric(frame["burned"], errors="coerce").fillna(0).astype(int)
        for _, row in frame[burned == 1].iterrows():
            lat_cell = float(row["lat_cell"])
            lon_cell = float(row["lon_cell"])
            folium.CircleMarker(
                location=[lat_cell, lon_cell],
                radius=7,
                color="#000000",
                fill=False,
                weight=2,
                tooltip=f"Observed fire • {lat_cell:.2f}°N, {lon_cell:.2f}°E",
            ).add_to(m)

    _inject_clean_legend(
        m, title="Predicted Fire Risk", entries=legend_entries
    )
    return m


def _inject_clean_legend(
    m: folium.Map,
    title: str = "Fire Activity Density",
    entries: list[tuple[str, str]] | None = None,
) -> None:
    """Add a professional floating legend to the Folium map."""
    if entries is None:
        entries = [
            (COLORS["risk_low"], "1 – 4 fire-days (Low)"),
            (COLORS["risk_moderate"], "5 – 12 fire-days (Moderate)"),
            (COLORS["risk_high"], "13 – 25 fire-days (High)"),
            (COLORS["risk_very_high"], "> 25 fire-days (Very High)"),
        ]

    rows_html = "\n".join(
        f"""
        <div style="display:flex; align-items:center; gap:6px; margin:3px 0;">
            <div style="width:10px; height:10px; border-radius:50%; background-color:{color};"></div>
            <span>{text}</span>
        </div>
        """
        for color, text in entries
    )

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
            {title}
        </div>
        {rows_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
