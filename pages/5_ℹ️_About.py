"""
FIREWATCH AI — Scientific Methodology & Architecture Whitepaper (Module 5).

Documentation of Earth Observation data products, spatial grid snapping,
antecedent feature engineering, and evaluation protocols for rare-event fire modeling.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Methodology & Data | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
)

from dashboard.components import (
    inject_global_css,
    render_header,
    render_sidebar,
    render_section_header,
)
from dashboard.config import (
    COLORS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    START_YEAR,
    END_YEAR,
    WEATHER_VARIABLES,
)

inject_global_css()
render_sidebar()
render_header(
    title="Scientific Methodology & Data Architecture",
    subtitle="Data provenance, spatial-temporal integration, and evaluation framework",
    badge="Scientific Whitepaper",
)

# Fundamental Conceptual Distinction
st.markdown(
    f"""
    <div style="
        background-color: {COLORS['surface_white']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    ">
        <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {COLORS['brand_primary']}; margin-bottom: 4px;">
            Foundational Premise
        </div>
        <div style="font-size: 1.1rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 14px;">
            Conceptual Boundary: Retrospective Detection vs. Predictive Risk Assessment
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px;">
            <div style="
                background-color: {COLORS['surface_subtle']};
                border: 1px solid {COLORS['border_light']};
                border-left: 3px solid {COLORS['brand_primary']};
                border-radius: 6px;
                padding: 14px 16px;
            ">
                <div style="font-size: 0.88rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 4px;">
                    Satellite Fire Detection (NASA FIRMS)
                </div>
                <p style="font-size: 0.82rem; color: {COLORS['text_secondary']}; line-height: 1.5; margin: 0;">
                    <b>Retrospective physical observation:</b> VIIRS sensors measure mid-infrared radiative emissions to confirm thermal anomalies at the exact instant of satellite overpass. It reports where active combustion has already occurred.
                </p>
            </div>
            <div style="
                background-color: {COLORS['surface_subtle']};
                border: 1px solid {COLORS['border_light']};
                border-left: 3px solid {COLORS['risk_high']};
                border-radius: 6px;
                padding: 14px 16px;
            ">
                <div style="font-size: 0.88rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 4px;">
                    Predictive Wildfire Risk Assessment (ML Model)
                </div>
                <p style="font-size: 0.82rem; color: {COLORS['text_secondary']}; line-height: 1.5; margin: 0;">
                    <b>Environmental susceptibility estimation:</b> Machine learning models evaluate fuel moisture deficits, vapor pressure deficit, drought history, and thermal forcing to estimate the likelihood of fire ignition, independent of active flame detection.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabbed Documentation Sections
tab_sources, tab_pipeline, tab_ml, tab_governance = st.tabs([
    "1. Earth Observation Data Sources",
    "2. ETL & Feature Engineering Pipeline",
    "3. Modeling Strategy & Imbalance Handling",
    "4. Methodological Safeguards & Data Leakage",
])

# Tab 1: Data Sources
with tab_sources:
    render_section_header(
        title="Primary Data Products & Acquisition Parameters",
        subtitle="Detailed specifications of NASA satellite products ingested into the platform",
    )

    st.markdown(
        f"""
        #### 1. NASA FIRMS (Fire Information for Resource Management System)
        - **Satellite & Instrument:** Suomi-NPP VIIRS (Visible Infrared Imaging Radiometer Suite)
        - **Data Collection:** VIIRS Collection 2 (SV-C2 Archive & NRT)
        - **Nominal Resolution:** 375 meters at nadir
        - **Filtering Protocol:** Presumed vegetation fires only (`type = 0`), discarding urban heat islands, volcanic activity, and offshore flares. Non-nominal/low confidence hits are removed.
        - **Total Records Processed:** 86,884 raw detections aggregated into 5,230 cell-day ignition events.

        #### 2. NASA POWER (Prediction of Worldwide Energy Resources)
        - **Data Product:** Daily Regional Surface Meteorology (Community AG)
        - **Spatial Resolution:** 0.5° latitude × 0.625° longitude regular grid
        - **Temporal Domain:** {START_YEAR} to {END_YEAR} (6 regional download tiles covering northern Algeria)
        - **Parameters Ingested:**
        """
    )

    var_table_rows = []
    for k, v in WEATHER_VARIABLES.items():
        var_table_rows.append({
            "Variable Code": k,
            "Formal Description": v["name"],
            "Measurement Unit": v["unit"],
            "Physical Relevance to Wildfire Behavior": v["description"],
        })
    st.dataframe(pd.DataFrame(var_table_rows), use_container_width=True, hide_index=True)

# Tab 2: ETL Pipeline
with tab_pipeline:
    render_section_header(
        title="Data Fusion & Feature Construction Pipeline",
        subtitle="End-to-end data transformation architecture from raw CSVs to final panel",
    )

    st.markdown(
        """
        The project enforces a strict, reproducible four-stage data preparation pipeline:

        ```text
        Raw FIRMS VIIRS Detections                         NASA POWER Meteorological API
                   │                                                     │
         [ src/clean_firms.py ]                                [ src/fetch_power.py ]
         • Filter type = 0 (vegetation)                        • Ingest 7 variables across 6 tiles
         • Filter nominal/high confidence                      • Deduplicate tile seam overlaps
         • Snap coordinates to 0.5° x 0.625° grid              • Impute NASA sentinel values (-999 -> NaN)
         • Aggregate daily by (lat_cell, lon_cell, date)                 │
                   │                                                     │
         data/interim/fires_daily.parquet                      data/raw/power_cache/*.csv
                   └──────────────────────────┬──────────────────────────┘
                                              │
                                     [ src/build_panel.py ]
                                     • Full Cartesian panel expansion
                                     • Left-join fire occurrence labels
                                     • Binary target generation: burned in {0, 1}
                                     • Backward rolling precipitation: rain_7d, 14d, 30d
                                     • Backward rolling max temperature: tmax_7d, 14d, 30d
                                     • Backward rolling min humidity: rhmin_7d, 14d, 30d
                                     • Consecutive dry spell counter: dry_days (rain <= 1mm)
                                     • Seasonal harmonic indicators: doy_sin, doy_cos
                                              │
                                     data/processed/fire_panel.parquet
                                     (276,480 rows x 25 columns)
        ```
        """
    )

# Tab 3: ML Strategy
with tab_ml:
    render_section_header(
        title="Machine Learning Evaluation & Rare-Event Protocols",
        subtitle="Methodological guidelines implemented for training the wildfire risk model",
    )

    st.markdown(
        """
        #### 1. Severe Class Imbalance Handling
        In northern Algeria, wildfire occurrence is an episodic phenomenon: active fires are detected in less than 2% of total cell-day observations.
        - **Standard Accuracy is Meaningless:** A trivial baseline predicting constant zero achieves >98% accuracy while failing completely as an early warning system.
        - **Required Metric Standards:**
          - **PR-AUC (Precision-Recall Area Under Curve):** Primary performance benchmark; sensitive to rare positive class ranking.
          - **Brier Score Loss:** Assesses calibration of output probability estimates.
        - **Shipped Estimator:** `HistGradientBoostingClassifier` (200 iterations, depth 5, learning rate 0.05) wrapped in `CalibratedClassifierCV` with isotonic calibration fitted on the 2021 validation year.

        #### 2. Temporal Splitting Protocol
        Due to severe temporal auto-correlation between contiguous days (heatwaves lasting 5–10 days), standard randomized K-Fold cross-validation suffers from temporal data leakage.
        - **Chronological Split:** The shipped model trains on 2018–2020, calibrates on 2021, and holds out 2022–2024. The dashboard panel (2024–2026) is newer than the model's test period and is scored as new data, not as a fresh evaluation set.
        - **Prediction Horizon:** The target is at least one fire in the grid cell within the next 3 days (`horizon_days = 3`). Calibrated probabilities refer to that 3-day window, not to a single calendar day.
        """
    )

# Tab 4: Governance & Leakage
with tab_governance:
    render_section_header(
        title="Methodological Guardrails & Data Leakage Elimination",
        subtitle="Strict protocols prohibiting circular predictive logic",
    )

    st.markdown(
        """
        #### Strict Feature Exclusion Rules
        To ensure scientific validity and real-world applicability:
        1. **Fire-Derived Columns Excluded:** Columns such as `n_detections` and `total_frp` are strictly banned from feature sets during training and inference. Using them would represent catastrophic data leakage (predicting fire presence using fire measurements).
        2. **Spatial Coordinates Used Deliberately:** `lat_cell` and `lon_cell` are model inputs (`use_coords = true`) so the classifier can learn regional fire climatology. They are not derived from fire observations and therefore do not leak the target.
        3. **Target Horizon vs. Observed Layer:** The panel's `burned` column is a same-day detection label. The shipped model's target is at least one fire within the next 3 days (`horizon_days = 3`), so its probabilities refer to that window. The map's observed overlay remains same-day and is deliberately not treated as the model's target.
        """
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; font-size: 0.78rem; color: {COLORS['text_muted']};">
        <b>FIREWATCH AI Platform</b> • Academic Data Science & Mining Project (2024–2026)<br>
        Developed with Python, Streamlit, Plotly, Folium, and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
