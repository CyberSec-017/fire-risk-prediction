"""
FIREWATCH AI — Data Exploration & Analytics Workspace (Module 3).

Statistical analysis of environmental fire drivers, meteorological separations,
and spatial climatology profiles across northern Algeria.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Analytics | FIREWATCH AI",
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
from dashboard.data_loader import load_fire_panel, get_date_range, get_unique_years
from dashboard.charts import (
    weather_distribution_comparison,
    weather_time_series_trend,
    weather_correlation_heatmap,
    bivariate_weather_scatter,
    fire_by_latitude_profile,
    fire_by_longitude_profile,
)
from dashboard.config import (
    COLORS,
    WEATHER_VARIABLES,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
)

inject_global_css()
render_sidebar()
render_header(
    title="Data Exploration & Analytics",
    subtitle="Multivariate investigation of meteorological drivers and historical wildfire occurrences",
    badge="Analytical Workspace",
)

df = load_fire_panel()
if df is None or df.empty:
    render_data_unavailable("fire_panel.parquet")
    st.stop()

# Sidebar Filters
st.sidebar.markdown("### Analytical Filters")

available_years = get_unique_years(df)
chosen_years = st.sidebar.multiselect(
    "Observation Year(s)",
    options=available_years,
    default=available_years,
    help="Restrict dataset to specific annual periods",
)

min_d, max_d = get_date_range(df)
date_bounds = st.sidebar.date_input(
    "Date Window",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)

start_date, end_date = None, None
if isinstance(date_bounds, (tuple, list)):
    if len(date_bounds) == 2:
        start_date, end_date = date_bounds
    elif len(date_bounds) == 1:
        start_date = end_date = date_bounds[0]
else:
    start_date = end_date = date_bounds

with st.sidebar.expander("Geographic Sub-Bounds", expanded=False):
    lat_sel = st.slider("Latitude (°N)", float(LAT_MIN), float(LAT_MAX), (float(LAT_MIN), float(LAT_MAX)), 0.25)
    lon_sel = st.slider("Longitude (°E)", float(LON_MIN), float(LON_MAX), (float(LON_MIN), float(LON_MAX)), 0.25)

# Filter dataframe
filtered_df = df.copy()
if chosen_years:
    filtered_df = filtered_df[filtered_df["date"].dt.year.isin(chosen_years)]
if start_date and end_date:
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date)
        & (filtered_df["date"].dt.date <= end_date)
    ]
filtered_df = filtered_df[
    filtered_df["lat_cell"].between(lat_sel[0], lat_sel[1])
    & filtered_df["lon_cell"].between(lon_sel[0], lon_sel[1])
]

# Metrics strip
n_subset = len(filtered_df)
n_fires = int(filtered_df["burned"].sum())
subset_rate = (n_fires / n_subset * 100) if n_subset > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card(
        title="Filtered Observations",
        value=f"{n_subset:,}",
        subtitle=f"{(n_subset / len(df) * 100):.1f}% of total panel records",
        status="neutral",
    )
with col2:
    render_kpi_card(
        title="Active Fire Cell-Days",
        value=f"{n_fires:,}",
        subtitle=f"Ignition rate: {subset_rate:.2f}%",
        status="danger",
    )
with col3:
    render_kpi_card(
        title="Time Span",
        value=f"{(end_date - start_date).days if (start_date and end_date) else 0} Days",
        subtitle=f"{start_date} → {end_date}",
        status="low",
    )
with col4:
    render_kpi_card(
        title="Spatial Footprint",
        value=f"{filtered_df.groupby(['lat_cell', 'lon_cell']).ngroups:,} Cells",
        subtitle="Selected coordinate cells",
        status="neutral",
    )

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Analytical Investigations in Tabs
tab_drivers, tab_spatial, tab_corr = st.tabs([
    "Atmospheric & Fuel Moisture Drivers",
    "Spatial Climatology Profiles",
    "Multivariate Correlations & Clustering",
])

# Tab 1: Atmospheric Drivers
with tab_drivers:
    render_section_header(
        title="Meteorological Variable Fire/No-Fire Separation",
        subtitle="Evaluate how environmental conditions diverge between active fire incidents and unburned days",
    )

    col_v, _ = st.columns([1.5, 2.5])
    with col_v:
        target_var = st.selectbox(
            "Select Environmental Parameter",
            options=list(WEATHER_VARIABLES.keys()),
            format_func=lambda x: f"{x} — {WEATHER_VARIABLES[x]['name']} ({WEATHER_VARIABLES[x]['unit']})",
            index=1,
            help="Choose variable for distribution and time-series inspection",
        )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(
            weather_distribution_comparison(filtered_df, var_key=target_var),
            use_container_width=True,
            key="analytics_dist_comp",
        )
    with col_g2:
        st.plotly_chart(
            weather_time_series_trend(filtered_df, var_key=target_var, freq="W"),
            use_container_width=True,
            key="analytics_ts_trend",
        )

# Tab 2: Spatial Profiles
with tab_spatial:
    render_section_header(
        title="Geographic Climatology Gradients",
        subtitle="Examine fire incidence frequency along latitude (zonal) and longitude (meridional) axes",
    )

    col_sp1, col_sp2 = st.columns(2)
    with col_sp1:
        st.plotly_chart(
            fire_by_latitude_profile(filtered_df),
            use_container_width=True,
            key="analytics_lat_prof",
        )
    with col_sp2:
        st.plotly_chart(
            fire_by_longitude_profile(filtered_df),
            use_container_width=True,
            key="analytics_lon_prof",
        )

# Tab 3: Correlation & Clustering
with tab_corr:
    render_section_header(
        title="Inter-Variable Correlation & Feature Space Clustering",
        subtitle="Pearson correlation matrix and bivariate separation in environmental feature space",
    )

    col_c1, col_c2 = st.columns([1.1, 1.0])
    with col_c1:
        st.plotly_chart(
            weather_correlation_heatmap(filtered_df),
            use_container_width=True,
            key="analytics_corr_mat",
        )

    with col_c2:
        st.markdown(
            f"""
            <div style="font-size:0.86rem; font-weight:600; color:{COLORS['text_primary']}; margin-bottom:8px;">
                Bivariate Scatter Exploration
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_bx, col_by = st.columns(2)
        with col_bx:
            scatter_x = st.selectbox("X-Axis Variable", list(WEATHER_VARIABLES.keys()), index=1, key="scat_x")
        with col_by:
            scatter_y = st.selectbox("Y-Axis Variable", list(WEATHER_VARIABLES.keys()), index=2, key="scat_y")

        st.plotly_chart(
            bivariate_weather_scatter(filtered_df, var_x=scatter_x, var_y=scatter_y),
            use_container_width=True,
            key="analytics_bivariate",
        )

# Filtered Subset Export
with st.expander("Export Filtered Analytics Subset (CSV)", expanded=False):
    st.write(f"Download the current filtered subset containing {len(filtered_df):,} records for offline analysis.")
    csv_bytes = filtered_df.head(5000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data (Top 5,000 Rows)",
        data=csv_bytes,
        file_name="firewatch_analytics_subset.csv",
        mime="text/csv",
    )
