"""
FIREWATCH AI — Geospatial Wildfire Monitoring (Module 2).

Interactive GIS mapping workspace for inspecting spatial distribution,
intensity hotspots, and cluster dynamics across northern Algeria.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Geospatial Monitoring | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
)

from dashboard.components import (
    inject_global_css,
    render_header,
    render_sidebar,
    render_kpi_card,
    render_section_header,
    render_callout,
    render_data_unavailable,
)
from dashboard.data_loader import load_fires_daily, load_fire_panel, get_date_range
from dashboard.maps import (
    render_grid_detection_map,
    render_density_heatmap,
    render_cluster_map,
)
from dashboard.config import (
    COLORS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
)

inject_global_css()
render_sidebar()
render_header(
    title="Geospatial Fire Monitoring",
    subtitle="Interactive spatial analysis and thermal hotspot mapping (32.0°–37.5°N, -2.0°–9.0°E)",
    badge="GIS Engine Active",
)

# Load fire records
fires_df = load_fires_daily()
if fires_df is None or fires_df.empty:
    panel_df = load_fire_panel()
    if panel_df is not None and not panel_df.empty:
        fires_df = panel_df[panel_df["burned"] == 1].copy()

if fires_df is None or fires_df.empty:
    render_data_unavailable("fires_daily.parquet or fire_panel.parquet")
    st.stop()

min_d, max_d = get_date_range(fires_df)

# Top Filter Toolbar
st.markdown(
    f"""
    <div style="
        background-color: {COLORS['surface_white']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 18px;
    ">
        <div style="font-size: 0.8rem; font-weight: 600; color: {COLORS['text_primary']}; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 10px;">
            Filter Controls & Map Configurations
        </div>
    """,
    unsafe_allow_html=True,
)

col_f1, col_f2, col_f3 = st.columns([1.5, 1.2, 1.3])

with col_f1:
    date_filter = st.date_input(
        "Observation Time Window",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
        help="Filter fire detections by satellite acquisition date",
    )

with col_f2:
    map_layer = st.selectbox(
        "Cartographic Layer",
        options=["Grid Frequency Markers", "Density HeatMap Surface", "Clustered Incidents"],
        index=0,
        help="Select spatial visualization technique",
    )

with col_f3:
    map_base_tile = st.selectbox(
        "Base Map Cartography",
        options=["CartoDB positron", "CartoDB dark_matter", "OpenStreetMap"],
        index=0,
        help="Switch underlying basemap tile provider",
    )

st.markdown("</div>", unsafe_allow_html=True)

# Parse date range
start_date, end_date = None, None
if isinstance(date_filter, (tuple, list)):
    if len(date_filter) == 2:
        start_date, end_date = date_filter
    elif len(date_filter) == 1:
        start_date = end_date = date_filter[0]
else:
    start_date = end_date = date_filter

# Geographic Sub-filter expander
with st.expander("Geographic Sub-Region Bounding Filter", expanded=False):
    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        lat_sub = st.slider(
            "Latitude Range (°N)",
            min_value=float(LAT_MIN),
            max_value=float(LAT_MAX),
            value=(float(LAT_MIN), float(LAT_MAX)),
            step=0.25,
        )
    with col_geo2:
        lon_sub = st.slider(
            "Longitude Range (°E)",
            min_value=float(LON_MIN),
            max_value=float(LON_MAX),
            value=(float(LON_MIN), float(LON_MAX)),
            step=0.25,
        )

# Apply filters
filtered_fires = fires_df.copy()
if start_date and end_date:
    filtered_fires = filtered_fires[
        (filtered_fires["date"].dt.date >= start_date)
        & (filtered_fires["date"].dt.date <= end_date)
    ]

filtered_fires = filtered_fires[
    filtered_fires["lat_cell"].between(lat_sub[0], lat_sub[1])
    & filtered_fires["lon_sub"].between(lon_sub[0], lon_sub[1]) if "lon_sub" in filtered_fires.columns
    else filtered_fires["lon_cell"].between(lon_sub[0], lon_sub[1])
]

# Summary KPI strip
total_fire_records = len(filtered_fires)
unique_grid_cells = filtered_fires.groupby(["lat_cell", "lon_cell"]).ngroups
total_satellite_hits = int(filtered_fires["n_detections"].sum()) if "n_detections" in filtered_fires.columns else total_fire_records
total_frp_mw = float(filtered_fires["total_frp"].sum()) if "total_frp" in filtered_fires.columns else 0.0

col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    render_kpi_card(
        title="Filtered Fire Cell-Days",
        value=f"{total_fire_records:,}",
        subtitle="Matching active filters",
        status="danger",
    )
with col_k2:
    render_kpi_card(
        title="Active Spatial Coordinates",
        value=f"{unique_grid_cells:,}",
        subtitle="Distinct grid centers with fire",
        status="neutral",
    )
with col_k3:
    render_kpi_card(
        title="Raw VIIRS Detections",
        value=f"{total_satellite_hits:,}",
        subtitle="Spaceborne thermal anomalies",
        status="high",
    )
with col_k4:
    render_kpi_card(
        title="Cumulative FRP",
        value=f"{total_frp_mw:,.0f} MW" if total_frp_mw > 0 else "N/A",
        subtitle="Fire Radiative Power",
        status="moderate",
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# Render Map Container
if filtered_fires.empty:
    render_callout(
        title="No Fire Records Found",
        message="No fire detections match the selected date window and geographic sub-bounds. Try broadening the observation filters.",
        variant="warning",
    )
else:
    # Build map instance
    if map_layer == "Density HeatMap Surface":
        folium_map = render_density_heatmap(filtered_fires, tiles=map_base_tile)
    elif map_layer == "Clustered Incidents":
        folium_map = render_cluster_map(filtered_fires, tiles=map_base_tile)
    else:
        folium_map = render_grid_detection_map(filtered_fires, tiles=map_base_tile)

    # Display map with Streamlit-Folium
    try:
        from streamlit_folium import st_folium
        st_folium(folium_map, width=None, height=620, use_container_width=True)
    except Exception:
        st.components.v1.html(folium_map._repr_html_(), height=630)

# Data Table & CSV Export
with st.expander("Inspect & Export Filtered Geospatial Dataset", expanded=False):
    st.write(f"Displaying top 500 records from the currently filtered {total_fire_records:,} fire-day observations.")
    st.dataframe(
        filtered_fires.head(500).sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    csv_payload = filtered_fires.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Filtered Geospatial Records (CSV)",
        data=csv_payload,
        file_name="firewatch_geospatial_export.csv",
        mime="text/csv",
    )
