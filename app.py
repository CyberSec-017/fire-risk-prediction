"""
FIREWATCH AI — Executive Environmental Intelligence Platform.

Main application landing page presenting platform status, study area
parameters, module gateways, and system provenance.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="FIREWATCH AI | Wildfire Intelligence Platform",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "FIREWATCH AI — Wildfire Monitoring & Risk Assessment Platform\n"
            "Academic Data Science Project (2024–2026)\n"
            "Data Sources: NASA FIRMS (VIIRS SV-C2) & NASA POWER"
        ),
    },
)

from dashboard.components import (
    inject_global_css,
    render_header,
    render_sidebar,
    render_kpi_card,
    render_section_header,
)
from dashboard.data_loader import load_fire_panel, get_date_range
from dashboard.model_loader import load_model
from dashboard.config import COLORS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, START_YEAR, END_YEAR

# Inject theme styling
inject_global_css()
render_sidebar()
render_header(
    title="FIREWATCH AI",
    subtitle="Intelligent Wildfire Monitoring & Environmental Risk Assessment",
    badge="Operational System • Northern Algeria",
)

# Load dataset for live status check
df = load_fire_panel()
panel_available = df is not None and not df.empty
ml_model = load_model()
model_status_text = "Production Model Integrated" if ml_model is not None else "Model Staging Ready (Pending Member 4)"

# Executive Briefing Card
st.markdown(
    f"""
    <div style="
        background-color: {COLORS['surface_white']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div style="max-width: 780px;">
                <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {COLORS['brand_primary']}; margin-bottom: 6px;">
                    Executive Overview
                </div>
                <div style="font-size: 1.15rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 8px; letter-spacing: -0.01em;">
                    Satellite-Derived Environmental Intelligence for Wildfire Hazard Assessment
                </div>
                <p style="font-size: 0.88rem; color: {COLORS['text_secondary']}; line-height: 1.6; margin: 0;">
                    FIREWATCH AI fuses spaceborne active thermal anomalies from <b>NASA FIRMS (Suomi-NPP VIIRS)</b> with surface meteorological observations from <b>NASA POWER</b>. 
                    Targeting <b>northern Algeria</b> ({LAT_MIN}°–{LAT_MAX}°N, {LON_MIN}°–{LON_MAX}°E), the platform monitors historical fire occurrences, analyzes atmospheric ignition drivers, and provides a standardized inference pipeline for machine learning wildfire risk classification.
                </p>
            </div>
            <div style="
                background-color: {COLORS['surface_subtle']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 6px;
                padding: 12px 16px;
                min-width: 220px;
                font-size: 0.78rem;
            ">
                <div style="font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 4px;">System Health</div>
                <div style="color: {COLORS['text_secondary']};">Data Engine: <span style="color:{COLORS['risk_low']}; font-weight:600;">● Online</span></div>
                <div style="color: {COLORS['text_secondary']};">Panel Records: <b>{f'{len(df):,}' if panel_available else 'Pending'}</b></div>
                <div style="color: {COLORS['text_secondary']};">ML Model: <span style="color:{COLORS['risk_low'] if ml_model else COLORS['risk_moderate']}; font-weight:600;">● {model_status_text.split(' ')[0]}</span></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Key Dataset KPIs
if panel_available:
    min_d, max_d = get_date_range(df)
    total_obs = len(df)
    total_fires = int(df["burned"].sum())
    cells_count = df.groupby(["lat_cell", "lon_cell"]).ngroups
    fire_pct = (total_fires / total_obs) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Total Monitored Cell-Days",
            value=f"{total_obs:,}",
            subtitle=f"{min_d.strftime('%b %Y')} → {max_d.strftime('%b %Y')}",
            status="neutral",
        )
    with col2:
        render_kpi_card(
            title="Confirmed Fire Occurrences",
            value=f"{total_fires:,}",
            subtitle=f"{fire_pct:.2f}% rare-event base rate",
            status="danger",
        )
    with col3:
        render_kpi_card(
            title="Spatial Grid Resolution",
            value=f"{cells_count:,} Cells",
            subtitle="0.5° × 0.625° NASA grid",
            status="neutral",
        )
    with col4:
        render_kpi_card(
            title="Meteorological Parameters",
            value="7 Variables",
            subtitle="19 Total engineered predictors",
            status="low",
        )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Platform Workspace Modules
render_section_header(
    title="Platform Analytical Workspaces",
    subtitle="Navigate to specialized monitoring, diagnostic, and predictive modules using the sidebar or the modules below",
)

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['text_primary']};">
                    1. Overview Dashboard
                </div>
                <span style="font-size: 0.72rem; background-color: {COLORS['surface_subtle']}; padding: 2px 8px; border-radius: 4px; color: {COLORS['text_muted']};">Module 1</span>
            </div>
            <p style="font-size: 0.84rem; color: {COLORS['text_secondary']}; line-height: 1.5; margin: 0 0 14px 0;">
                High-level operational metrics, aggregated daily and weekly fire occurrence timelines, multi-year comparisons, and environmental condition sparklines.
            </p>
            <div style="font-size: 0.76rem; color: {COLORS['text_muted']}; border-top: 1px solid {COLORS['border_light']}; padding-top: 8px;">
                <b>Key Capabilities:</b> Temporal trends • Monthly seasonality • Macro indicators
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['text_primary']};">
                    2. Geospatial Fire Monitoring
                </div>
                <span style="font-size: 0.72rem; background-color: {COLORS['surface_subtle']}; padding: 2px 8px; border-radius: 4px; color: {COLORS['text_muted']};">Module 2</span>
            </div>
            <p style="font-size: 0.84rem; color: {COLORS['text_secondary']}; line-height: 1.5; margin: 0 0 14px 0;">
                Interactive GIS map of northern Algeria with variable layer controls: individual cell activity markers, heat density surfaces, and incident clusters with inspection popups.
            </p>
            <div style="font-size: 0.76rem; color: {COLORS['text_muted']}; border-top: 1px solid {COLORS['border_light']}; padding-top: 8px;">
                <b>Key Capabilities:</b> Folium mapping • Spatial bounding filters • CSV geo-export
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['text_primary']};">
                    3. Data Exploration & Analytics
                </div>
                <span style="font-size: 0.72rem; background-color: {COLORS['surface_subtle']}; padding: 2px 8px; border-radius: 4px; color: {COLORS['text_muted']};">Module 3</span>
            </div>
            <p style="font-size: 0.84rem; color: {COLORS['text_secondary']}; line-height: 1.5; margin: 0 0 14px 0;">
                Statistical workbench exploring relationships between atmospheric variables and fire occurrence. Includes Pearson correlation matrices, boxplot separations, and bivariate distributions.
            </p>
            <div style="font-size: 0.76rem; color: {COLORS['text_muted']}; border-top: 1px solid {COLORS['border_light']}; padding-top: 8px;">
                <b>Key Capabilities:</b> Hypothesis testing • Weather factor separation • Multi-variable filters
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 8px;
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="font-size: 0.95rem; font-weight: 600; color: {COLORS['text_primary']};">
                    4. Fire Risk Prediction Engine
                </div>
                <span style="font-size: 0.72rem; background-color: {COLORS['surface_subtle']}; padding: 2px 8px; border-radius: 4px; color: {COLORS['text_muted']};">Module 4</span>
            </div>
            <p style="font-size: 0.84rem; color: {COLORS['text_secondary']}; line-height: 1.5; margin: 0 0 14px 0;">
                Operational inference cockpit accepting weather parameters, drought history, and spatial harmonics to predict fire ignition risk levels and calibrated probability scores.
            </p>
            <div style="font-size: 0.76rem; color: {COLORS['text_muted']}; border-top: 1px solid {COLORS['border_light']}; padding-top: 8px;">
                <b>Key Capabilities:</b> Real-time inference • Model staging verification • Scenario presets
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Study Area Parameters Table
with st.expander("Geographic and Sensor Specifications", expanded=False):
    spec_data = {
        "Specification Dimension": [
            "Spatial Extent (Northern Algeria)",
            "Grid Center Reference",
            "Temporal Coverage",
            "Satellite Fire Sensor",
            "Detection Filtering Thresholds",
            "Meteorological Model Engine",
            "Prediction Unit Definition",
        ],
        "Configuration Details": [
            f"Latitude: {LAT_MIN}° to {LAT_MAX}°N | Longitude: {LON_MIN}° to {LON_MAX}°E",
            "34.75°N, 3.50°E (Tell Atlas & Steppe Transition)",
            f"{START_YEAR} to {END_YEAR} (Continuous daily time series)",
            "Suomi-NPP VIIRS (375m spatial resolution, Collection 2)",
            "Type = 0 (Presumed vegetation fire), Confidence ≥ Nominal",
            "NASA POWER Daily Regional Model (0.5° × 0.625° resolution)",
            "Binary same-day ignition indicator (burned = 1 if ≥1 active fire detected)",
        ],
    }
    st.dataframe(pd.DataFrame(spec_data), use_container_width=True, hide_index=True)
