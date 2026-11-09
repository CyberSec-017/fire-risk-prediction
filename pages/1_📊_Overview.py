"""
FIREWATCH AI — Overview Dashboard (Module 1).

Presents executive KPI indicators, temporal fire activity patterns,
seasonal concentrations, and meteorological summaries from the processed panel.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Overview Dashboard | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
)

from dashboard.components import (
    inject_global_css,
    render_header,
    render_sidebar,
    render_kpi_card,
    render_section_header,
    render_data_unavailable,
)
from dashboard.data_loader import load_fire_panel, get_date_range
from dashboard.charts import (
    fire_detections_timeline,
    monthly_fire_distribution,
    yearly_fire_breakdown,
)
from dashboard.config import COLORS, WEATHER_VARIABLES

inject_global_css()
render_sidebar()
render_header(
    title="Overview Dashboard",
    subtitle="Macro wildfire metrics, temporal dynamics, and environmental indicators",
    badge="Dataset Status: Synchronized",
)

# Load data
df = load_fire_panel()
if df is None or df.empty:
    render_data_unavailable("fire_panel.parquet")
    st.stop()

# Compute summary statistics
min_d, max_d = get_date_range(df)
total_records = len(df)
fire_cell_days = int(df["burned"].sum())
unique_cells = df.groupby(["lat_cell", "lon_cell"]).ngroups
fire_rate = (fire_cell_days / total_records) * 100
total_days = (max_d - min_d).days + 1
weather_param_count = sum(1 for c in WEATHER_VARIABLES if c in df.columns)

# Top KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi_card(
        title="Active Fire Cell-Days",
        value=f"{fire_cell_days:,}",
        subtitle=f"{fire_rate:.2f}% observed ignition frequency",
        status="danger",
    )

with col2:
    render_kpi_card(
        title="Monitored Grid Cells",
        value=f"{unique_cells:,}",
        subtitle="0.5° × 0.625° geographic cells",
        status="neutral",
    )

with col3:
    render_kpi_card(
        title="Observation Period",
        value=f"{total_days:,} Days",
        subtitle=f"{min_d.strftime('%Y-%m-%d')} → {max_d.strftime('%Y-%m-%d')}",
        status="low",
    )

with col4:
    render_kpi_card(
        title="Total Observations",
        value=f"{total_records:,}",
        subtitle=f"{weather_param_count} Atmospheric variables tracked",
        status="neutral",
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Temporal Dynamics Section
render_section_header(
    title="Fire Occurrence Timelines",
    subtitle="Temporal distribution of active fire detections across the study domain",
)

tab_weekly, tab_daily, tab_monthly = st.tabs(["Weekly Aggregation", "Daily Resolution", "Monthly Volume"])

with tab_weekly:
    fig_w = fire_detections_timeline(df, freq="W")
    st.plotly_chart(fig_w, use_container_width=True, key="overview_weekly")

with tab_daily:
    fig_d = fire_detections_timeline(df, freq="D")
    st.plotly_chart(fig_d, use_container_width=True, key="overview_daily")

with tab_monthly:
    fig_m = fire_detections_timeline(df, freq="ME")
    st.plotly_chart(fig_m, use_container_width=True, key="overview_monthly")

# Seasonal & Multi-Year Breakdowns
col_s1, col_s2 = st.columns(2)

with col_s1:
    render_section_header(
        title="Seasonal Concentration",
        subtitle="Cumulative fire cell counts by calendar month",
    )
    fig_dist = monthly_fire_distribution(df)
    st.plotly_chart(fig_dist, use_container_width=True, key="overview_season")

with col_s2:
    render_section_header(
        title="Year-over-Year Dynamics",
        subtitle="Comparison of fire intensity between 2024, 2025, and 2026",
    )
    fig_yearly = yearly_fire_breakdown(df)
    st.plotly_chart(fig_yearly, use_container_width=True, key="overview_yearly")

# Environmental Baseline Table
render_section_header(
    title="Meteorological Parameters Climatology",
    subtitle="Summary statistics of NASA POWER environmental variables across all monitored cell-days",
)

stats_rows = []
for key, meta in WEATHER_VARIABLES.items():
    if key in df.columns:
        s = df[key].dropna()
        stats_rows.append({
            "Variable Code": key,
            "Meteorological Parameter": meta["name"],
            "Unit": meta["unit"],
            "Mean": round(float(s.mean()), 2),
            "Std Dev": round(float(s.std()), 2),
            "10th Pct": round(float(s.quantile(0.10)), 2),
            "Median (50th)": round(float(s.median()), 2),
            "90th Pct": round(float(s.quantile(0.90)), 2),
            "Max Recorded": round(float(s.max()), 2),
        })

stats_df = pd.DataFrame(stats_rows)
st.dataframe(stats_df, use_container_width=True, hide_index=True)
