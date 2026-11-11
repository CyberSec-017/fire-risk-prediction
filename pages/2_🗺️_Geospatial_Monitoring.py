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

import app_core
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
from dashboard.model_loader import load_model, score_panel_date
from dashboard.maps import (
    render_grid_detection_map,
    render_density_heatmap,
    render_cluster_map,
    render_risk_map,
)
from dashboard.config import (
    COLORS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
)

RISK_LAYER = "Predicted Fire Risk (model)"


@st.cache_data(show_spinner="Scoring predicted fire risk …")
def _load_risk_scores(date_value: str) -> pd.DataFrame:
    """Cache model scores per panel date."""
    bundle = load_model()
    panel_df = load_fire_panel()
    if bundle.status != "REAL" or panel_df is None or panel_df.empty:
        return pd.DataFrame(
            columns=[
                "lat_cell",
                "lon_cell",
                "probability",
                "risk",
                "burned",
                "in_model_area",
            ]
        )
    return score_panel_date(bundle, panel_df, date_value)


inject_global_css()
render_sidebar()
render_header(
    title="Geospatial Fire Monitoring",
    subtitle="Interactive spatial analysis and thermal hotspot mapping (32.0°–37.5°N, -2.0°–9.0°E)",
    badge="GIS Engine Active",
)

# Load fire records and the full panel (the risk layer needs all cells)
fires_df = load_fires_daily()
panel_df = load_fire_panel()
if fires_df is None or fires_df.empty:
    if panel_df is not None and not panel_df.empty:
        fires_df = panel_df[panel_df["burned"] == 1].copy()

if (fires_df is None or fires_df.empty) and (panel_df is None or panel_df.empty):
    render_data_unavailable("fires_daily.parquet or fire_panel.parquet")
    st.stop()

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

with col_f2:
    map_layer = st.selectbox(
        "Cartographic Layer",
        options=[
            "Grid Frequency Markers",
            "Density HeatMap Surface",
            "Clustered Incidents",
            RISK_LAYER,
        ],
        index=0,
        help="Select spatial visualization technique",
    )

is_risk_layer = map_layer == RISK_LAYER

with col_f1:
    if is_risk_layer:
        if panel_df is None or panel_df.empty:
            render_callout(
                title="Panel Required",
                message="The predicted-risk layer needs data/processed/fire_panel.parquet.",
                variant="warning",
            )
            st.stop()

        panel_dates = pd.to_datetime(panel_df["date"], errors="coerce").dropna()
        first_panel_date = panel_dates.min()
        earliest_risk_date = first_panel_date + pd.Timedelta(days=30)
        available_risk_dates = sorted(
            panel_dates[panel_dates >= earliest_risk_date].dt.normalize().unique()
        )

        if available_risk_dates:
            risk_date = st.selectbox(
                "Prediction Date",
                options=available_risk_dates,
                index=len(available_risk_dates) - 1,
                format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
                help=(
                    "Dates start 30 days after the panel begins because the "
                    "7/14/30-day rolling features are incomplete before that."
                ),
                key="risk_prediction_date",
            )
            period_note = app_core.data_period_note(risk_date)
            note_palette = {
                "training period": (COLORS["brand_subtle"], COLORS["brand_accent"]),
                "calibration year": (COLORS["risk_moderate_bg"], COLORS["risk_moderate"]),
                "original test period": (COLORS["risk_low_bg"], COLORS["risk_low"]),
                "new data, never used for this model": (
                    COLORS["risk_high_bg"],
                    COLORS["risk_high"],
                ),
            }
            badge_bg, badge_fg = note_palette.get(
                period_note,
                (COLORS["surface_subtle"], COLORS["text_secondary"]),
            )
            st.markdown(
                f"""
                <div style="
                    display: inline-block;
                    margin-top: 6px;
                    padding: 3px 9px;
                    border-radius: 999px;
                    background-color: {badge_bg};
                    color: {badge_fg};
                    font-size: 0.72rem;
                    font-weight: 600;
                    letter-spacing: 0.01em;
                ">
                    Model period: {period_note}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            risk_date = None
            st.info("No panel dates are at least 30 days after the panel start.")
    else:
        if fires_df is None or fires_df.empty:
            render_callout(
                title="No Fire Records Found",
                message="This map layer needs fires_daily.parquet or labeled panel rows.",
                variant="warning",
            )
            st.stop()
        min_d, max_d = get_date_range(fires_df)
        date_filter = st.date_input(
            "Observation Time Window",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
            help="Filter fire detections by satellite acquisition date",
        )

with col_f3:
    map_base_tile = st.selectbox(
        "Base Map Cartography",
        options=["Light", "Dark", "OpenStreetMap"],
        index=0,
        help="Switch underlying basemap tile provider",
    )

st.markdown("</div>", unsafe_allow_html=True)

# Parse date range (only used by the three non-model layers)
start_date, end_date = None, None
if not is_risk_layer:
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

# Apply filters (the model layer scores its own cell frame below)
if not is_risk_layer:
    filtered_fires = fires_df.copy()
    if start_date and end_date:
        filtered_fires = filtered_fires[
            (filtered_fires["date"].dt.date >= start_date)
            & (filtered_fires["date"].dt.date <= end_date)
        ]

    lon_col = "lon_sub" if "lon_sub" in filtered_fires.columns else "lon_cell"
    filtered_fires = filtered_fires[
        filtered_fires["lat_cell"].between(lat_sub[0], lat_sub[1])
        & filtered_fires[lon_col].between(lon_sub[0], lon_sub[1])
    ]

if is_risk_layer:
    bundle = load_model()
    if bundle.status != "REAL":
        message = (
            f"The model loader returned {bundle.status}. Predictions are disabled "
            "until a REAL model/card pair is available."
        )
        if bundle.error:
            message += f" Details: {bundle.error}"
        render_callout(
            title="Predicted Fire Risk Unavailable",
            message=message,
            variant="danger" if bundle.status == "ERROR" else "warning",
        )
    elif risk_date is None:
        render_callout(
            title="No Prediction Date Available",
            message="The panel has no date at least 30 days after its start.",
            variant="warning",
        )
    else:
        selected_date_str = pd.Timestamp(risk_date).strftime("%Y-%m-%d")
        risk_scores = _load_risk_scores(selected_date_str)

        if risk_scores.empty:
            render_callout(
                title="No Cells Found",
                message=f"No panel cells are available for {selected_date_str}.",
                variant="warning",
            )
        else:
            risk_scores = risk_scores[
                risk_scores["lat_cell"].between(lat_sub[0], lat_sub[1])
                & risk_scores["lon_cell"].between(lon_sub[0], lon_sub[1])
            ].copy()

            if risk_scores.empty:
                render_callout(
                    title="No Cells Match the Geographic Filter",
                    message="Broaden the latitude/longitude sub-region and try again.",
                    variant="warning",
                )
            else:
                # Off-grid cells are drawn neutral and excluded from all KPIs.
                in_area_scores = risk_scores[risk_scores["in_model_area"]]
                highest_value = (
                    f"{float(in_area_scores['probability'].max()) * 100:.1f}%"
                    if not in_area_scores.empty
                    else "N/A"
                )
                high_count = int(
                    in_area_scores["risk"]
                    .astype(str)
                    .str.upper()
                    .isin(["HIGH", "VERY HIGH"])
                    .sum()
                )
                observed_count = (
                    int(in_area_scores["burned"].sum())
                    if "burned" in in_area_scores.columns
                    else 0
                )

                col_k1, col_k2, col_k3 = st.columns(3)

                with col_k1:
                    render_kpi_card(
                        title="Highest Predicted Probability",
                        value=highest_value,
                        subtitle=f"Selected date: {selected_date_str}",
                        status="danger",
                    )
                with col_k2:
                    render_kpi_card(
                        title="High / Very High Cells",
                        value=f"{high_count:,}",
                        subtitle="Model-card risk bands",
                        status="high",
                    )
                with col_k3:
                    render_kpi_card(
                        title="Observed Fires",
                        value=f"{observed_count:,}",
                        subtitle="Cells with burned = 1 on that date",
                        status="danger",
                    )

                st.markdown(
                    "<div style='height: 14px;'></div>",
                    unsafe_allow_html=True,
                )

                show_observed = st.checkbox(
                    "Overlay observed fires (burned = 1)",
                    value=False,
                    key="risk_overlay_observed",
                )

                risk_map = render_risk_map(
                    risk_scores,
                    bundle.card,
                    tiles=map_base_tile,
                    show_observed=show_observed,
                )

                try:
                    from streamlit_folium import st_folium
                    st_folium(
                        risk_map,
                        width=None,
                        height=620,
                        use_container_width=True,
                        returned_objects=[],
                        key=f"risk_map_{selected_date_str}",
                    )
                except Exception:
                    st.components.v1.html(
                        risk_map._repr_html_(), height=630
                    )

else:
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
