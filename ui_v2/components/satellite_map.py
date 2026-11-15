"""Interactive Satellite Map builder using Folium and Esri World Imagery.

Features:
  - High-resolution 2D satellite basemap from Esri World Imagery.
  - Transparent overlay layer for administrative boundaries, roads, and cities.
  - OpenStreetMap street map toggle via Folium LayerControl.
  - Bounding rectangle for the model's official training envelope (32.0–37.5°N, -2.0–9.0°E).
  - Visualization of the 209 active NASA POWER grid cell polygons.
  - Interactive risk-colored markers for evaluated points in the current session.
"""

from __future__ import annotations

from typing import Any, Sequence
import folium
from folium import plugins

import app_core
from ui_v2.components.point_context import (
    get_active_209_cells,
    get_cell_bounding_box,
)

# Geographic defaults (Northern Algeria)
MAP_CENTER_LAT = 35.5
MAP_CENTER_LON = 3.5
MAP_DEFAULT_ZOOM = 7

# Esri tile specifications
ESRI_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_IMAGERY_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, "
    "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
)

ESRI_BOUNDARIES_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)
ESRI_BOUNDARIES_ATTR = "Esri Boundaries & Places"


def build_satellite_map(
    predicted_points: Sequence[dict[str, Any]] | None = None,
    selected_point: tuple[float, float] | None = None,
    show_cells_grid: bool = True,
    zoom_start: int = MAP_DEFAULT_ZOOM,
    active_basemap: str = "satellite",
) -> folium.Map:
    """Build the interactive 2D satellite map of Algeria.

    Parameters
    ----------
    predicted_points : list of dicts, optional
        Historical predictions to display as markers.
    selected_point : tuple of (lat, lon), optional
        Currently targeted coordinates for risk evaluation.
    show_cells_grid : bool, default True
        Whether to render the 209 active NASA POWER grid polygons.
    zoom_start : int, default 7
        Initial zoom level.
    active_basemap : str, default "satellite"
        Persistent basemap tile layer ("satellite" for Esri World Imagery, "osm" for OpenStreetMap).
    """
    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=False,
        prefer_canvas=True,
        attributionControl=False,
    )

    # Clean map styling: hide technical attributions and scale bars from the canvas
    m.get_root().header.add_child(
        folium.Element(
            """
            <style>
            .leaflet-control-attribution,
            .leaflet-control-scale,
            .leaflet-bottom.leaflet-right,
            .leaflet-bottom.leaflet-left .leaflet-control-scale {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0 !important;
                pointer-events: none !important;
            }
            </style>
            """
        )
    )

    is_satellite = (active_basemap.lower() == "satellite")

    # 1. Base Layer: Esri World Imagery (Satellite)
    folium.TileLayer(
        tiles=ESRI_IMAGERY_URL,
        attr=ESRI_IMAGERY_ATTR,
        name="🛰️ Satellite (Esri World Imagery)",
        overlay=False,
        control=True,
        show=is_satellite,
        max_zoom=18,
    ).add_to(m)

    # 2. Alternative Base Layer: OpenStreetMap
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ Carte routière (OpenStreetMap)",
        overlay=False,
        control=True,
        show=(not is_satellite),
    ).add_to(m)

    # 3. Transparent Reference Overlay: Boundaries, roads, cities
    folium.TileLayer(
        tiles=ESRI_BOUNDARIES_URL,
        attr=ESRI_BOUNDARIES_ATTR,
        name="🏷️ Villes, Routes & Frontières (Esri)",
        overlay=True,
        control=True,
    ).add_to(m)

    # 4. Model Training Envelope (Northern Algeria: 32.0–37.5°N, -2.0–9.0°E)
    training_bounds = [
        [app_core.MODEL_LAT_MIN, app_core.MODEL_LON_MIN],
        [app_core.MODEL_LAT_MAX, app_core.MODEL_LON_MAX],
    ]
    folium.Rectangle(
        bounds=training_bounds,
        color="#EF4444",
        weight=2.0,
        dash_array="6, 6",
        fill=False,
        tooltip="Zone d'entraînement du modèle (Northern Algeria: 32.0°–37.5°N, -2.0°–9.0°E)",
    ).add_to(m)

    # 5. The 209 Active NASA POWER Grid Cell Polygons
    if show_cells_grid:
        cells_group = folium.FeatureGroup(name="🌐 Maillage NASA POWER (209 cellules)", show=True)
        active_cells = get_active_209_cells()
        for lat_cell, lon_cell in sorted(active_cells):
            bbox = get_cell_bounding_box(lat_cell, lon_cell)
            folium.Rectangle(
                bounds=bbox,
                color="#0284C7",
                weight=1.0,
                fill=True,
                fill_color="#38BDF8",
                fill_opacity=0.04,
                dash_array="4, 4",
                tooltip=f"Cellule NASA POWER : ({lat_cell:.3f}°N, {lon_cell:.3f}°E)",
            ).add_to(cells_group)
        cells_group.add_to(m)

    # 6. Persistent Markers for Predicted Points
    if predicted_points:
        markers_group = folium.FeatureGroup(name="📍 Prédictions de la session", show=True)
        for pt in predicted_points:
            p_lat = pt.get("lat")
            p_lon = pt.get("lon")
            if p_lat is None or p_lon is None:
                continue

            color = pt.get("risk_color", "#EA580C")
            prob_str = pt.get("probability_formatted", "—")
            risk_disp = pt.get("risk_display", "—")
            date_str = pt.get("date", "—")

            popup_html = f"""
            <div style="font-family:Inter,sans-serif; font-size:12px; min-width:180px;">
                <div style="font-weight:700; color:{color}; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                    {risk_disp.upper()} RISK ({prob_str})
                </div>
                <div><b>Date :</b> {date_str}</div>
                <div><b>Position :</b> {p_lat:.4f}°N, {p_lon:.4f}°E</div>
                <div><b>Cellule :</b> {pt.get('lat_cell', '—')}°N, {pt.get('lon_cell', '—')}°E</div>
            </div>
            """

            folium.CircleMarker(
                location=[p_lat, p_lon],
                radius=8,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=f"Risque {risk_disp} : {prob_str} ({date_str})",
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(markers_group)
        markers_group.add_to(m)

    # 7. Highlight Marker for Currently Selected Point (if any)
    if selected_point is not None:
        s_lat, s_lon = selected_point
        folium.CircleMarker(
            location=[s_lat, s_lon],
            radius=11,
            color="#FFFFFF",
            weight=3,
            fill=True,
            fill_color="#DC2626",
            fill_opacity=0.9,
            tooltip=f"Position ciblée : {s_lat:.4f}°N, {s_lon:.4f}°E",
        ).add_to(m)

    # Layer control for basemap switching
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    return m
