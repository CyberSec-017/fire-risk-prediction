"""
FIREWATCH AI — Fire Risk Prediction Cockpit (Module 4).

Operational decision-support interface accepting real-time environmental inputs,
calculating seasonal harmonics, and scoring them with the calibrated 20-feature
wildfire model.
"""

from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fire Risk Prediction | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
)

import app_core
from dashboard.components import (
    inject_global_css,
    render_header,
    render_sidebar,
    render_section_header,
    render_callout,
)
from dashboard.model_loader import (
    load_model,
    predict_fire_risk,
    MODEL_PATH,
    MODEL_FEATURES,
)
from dashboard.config import (
    COLORS,
    PROJECT_ROOT,
    RISK_LEVELS,
    WEATHER_VARIABLES,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
)

inject_global_css()
render_sidebar()
render_header(
    title="Fire Risk Prediction Engine",
    subtitle="Decision-support inference using atmospheric parameters, fuel aridity, and seasonal harmonics to estimate fire risk over the next 3 days",
    badge="Inference Pipeline",
)

# Inspect model status and contract
model = load_model()
model_state = model.status
model_card = model.card or {}
model_version = model_card.get("model_version", "unknown")
feature_count = len(MODEL_FEATURES)

try:
    model_file_label = MODEL_PATH.relative_to(PROJECT_ROOT).as_posix()
except ValueError:
    model_file_label = MODEL_PATH.name

state_palette = {
    "REAL": (COLORS["risk_low_bg"], COLORS["risk_low"]),
    "DEMO": (COLORS["risk_moderate_bg"], COLORS["risk_moderate"]),
    "ERROR": (COLORS["risk_very_high_bg"], COLORS["risk_very_high"]),
}
state_bg, state_fg = state_palette.get(
    model_state,
    (COLORS["surface_subtle"], COLORS["text_secondary"]),
)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 6px 0 4px 0;">
        <span style="
            padding: 3px 10px;
            border-radius: 999px;
            background-color: {state_bg};
            color: {state_fg};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        ">MODEL STATE: {model_state}</span>
        <span style="font-size: 0.8rem; color: {COLORS['text_secondary']};">
            Model v{model_version} • {feature_count} features • <code>{model_file_label}</code>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_state == "ERROR":
    render_callout(
        title="Model Load Error",
        message=model.error or "The trained model artifact could not be loaded.",
        variant="danger",
    )

# Scenario Presets
render_section_header(
    title="Quick Calibration Presets",
    subtitle="Load scientifically standardized weather scenarios into the inference engine",
)

col_p1, col_p2, col_p3 = st.columns(3)
preset_data = None

with col_p1:
    if st.button("Summer Heatwave Scenario (High Threat)", use_container_width=True):
        preset_data = {
            "T2M": 38.0, "T2M_MAX": 43.5, "RH2M": 15.0, "WS2M": 8.5,
            "PRECTOTCORR": 0.0, "ALLSKY_SFC_SW_DWN": 29.5, "GWETROOT": 0.12,
            "rain_7d": 0.0, "rain_14d": 0.0, "rain_30d": 1.0,
            "tmax_7d": 41.5, "tmax_14d": 39.8, "tmax_30d": 37.5,
            "rhmin_7d": 12.0, "rhmin_14d": 13.0, "rhmin_30d": 14.5,
            "dry_days": 42.0,
        }

with col_p2:
    if st.button("Spring Intermediate Scenario (Moderate)", use_container_width=True):
        preset_data = {
            "T2M": 21.5, "T2M_MAX": 26.8, "RH2M": 46.0, "WS2M": 4.2,
            "PRECTOTCORR": 0.6, "ALLSKY_SFC_SW_DWN": 20.5, "GWETROOT": 0.44,
            "rain_7d": 5.0, "rain_14d": 14.0, "rain_30d": 32.0,
            "tmax_7d": 25.5, "tmax_14d": 24.2, "tmax_30d": 23.0,
            "rhmin_7d": 36.0, "rhmin_14d": 33.0, "rhmin_30d": 30.0,
            "dry_days": 4.0,
        }

with col_p3:
    if st.button("Winter Wet Scenario (Low Threat)", use_container_width=True):
        preset_data = {
            "T2M": 10.5, "T2M_MAX": 14.0, "RH2M": 84.0, "WS2M": 5.0,
            "PRECTOTCORR": 16.5, "ALLSKY_SFC_SW_DWN": 10.0, "GWETROOT": 0.86,
            "rain_7d": 42.0, "rain_14d": 78.0, "rain_30d": 135.0,
            "tmax_7d": 14.5, "tmax_14d": 15.2, "tmax_30d": 16.0,
            "rhmin_7d": 64.0, "rhmin_14d": 60.0, "rhmin_30d": 58.0,
            "dry_days": 0.0,
        }

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Form Layout: 3 Organized Fieldsets
render_section_header(
    title="Environmental Feature Inputs",
    subtitle="Configure the 20 predictors expected by the trained wildfire model",
)

# Row 1: Spatial & Temporal Coordinates
st.markdown(
    f"""
    <div style="font-size:0.84rem; font-weight:600; color:{COLORS['text_primary']}; margin-bottom:8px;">
        1. Spatial Location & Observation Date
    </div>
    """,
    unsafe_allow_html=True,
)
col_c1, col_c2, col_c3 = st.columns(3)
with col_c1:
    in_lat = st.number_input(
        "Target Latitude (°N)",
        min_value=float(LAT_MIN),
        max_value=float(LAT_MAX),
        value=36.5,
        step=0.5,
        help="Grid latitude within northern Algeria (32.0°–37.5°N)",
    )
with col_c2:
    in_lon = st.number_input(
        "Target Longitude (°E)",
        min_value=float(LON_MIN),
        max_value=float(LON_MAX),
        value=3.0,
        step=0.625,
        help="Grid longitude within northern Algeria (-2.0°–9.0°E)",
    )
with col_c3:
    in_date = st.date_input(
        "Observation Calendar Date",
        value=datetime.date.today(),
        help="Used to compute day-of-year seasonal harmonics (doy_sin, doy_cos)",
    )

lat_cell, lon_cell = app_core.snap_to_grid(in_lat, in_lon)
st.markdown(
    f"""
    <div style="font-size:0.8rem; color:{COLORS['text_secondary']}; margin-top:-2px;">
        Grid cell used: lat {lat_cell:.3f}, lon {lon_cell:.3f}
    </div>
    """,
    unsafe_allow_html=True,
)

doy = in_date.timetuple().tm_yday
doy_sin = float(np.sin(2 * np.pi * doy / 365.25))
doy_cos = float(np.cos(2 * np.pi * doy / 365.25))

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Row 2: Same-Day Atmospheric Conditions
st.markdown(
    f"""
    <div style="font-size:0.84rem; font-weight:600; color:{COLORS['text_primary']}; margin-bottom:8px;">
        2. Daily Atmospheric Conditions (NASA POWER)
    </div>
    """,
    unsafe_allow_html=True,
)
col_a1, col_a2, col_a3, col_a4 = st.columns(4)

with col_a1:
    v_t2m = st.number_input(
        "Mean Air Temp: T2M (°C)",
        value=preset_data["T2M"] if preset_data else 26.0,
        step=0.5,
    )
    v_t2m_max = st.number_input(
        "Max Air Temp: T2M_MAX (°C)",
        value=preset_data["T2M_MAX"] if preset_data else 32.5,
        step=0.5,
    )

with col_a2:
    v_rh2m = st.number_input(
        "Relative Humidity: RH2M (%)",
        min_value=0.0,
        max_value=100.0,
        value=preset_data["RH2M"] if preset_data else 32.0,
        step=1.0,
    )
    v_ws2m = st.number_input(
        "Wind Speed: WS2M (m/s)",
        min_value=0.0,
        value=preset_data["WS2M"] if preset_data else 5.0,
        step=0.5,
    )

with col_a3:
    v_prec = st.number_input(
        "Precipitation: PRECTOTCORR (mm)",
        min_value=0.0,
        value=preset_data["PRECTOTCORR"] if preset_data else 0.0,
        step=0.2,
    )
    v_rad = st.number_input(
        "Solar Radiation: ALLSKY (MJ/m²) — not used by the model",
        min_value=0.0,
        value=preset_data["ALLSKY_SFC_SW_DWN"] if preset_data else 25.0,
        step=1.0,
        help="Display only. Solar radiation is not one of the 20 model features.",
    )

with col_a4:
    v_gwet = st.number_input(
        "Root Soil Wetness: GWETROOT (0–1)",
        min_value=0.0,
        max_value=1.0,
        value=preset_data["GWETROOT"] if preset_data else 0.28,
        step=0.05,
    )
    v_dry = st.number_input(
        "Consecutive Dry Days (≤1mm)",
        min_value=0.0,
        value=preset_data["dry_days"] if preset_data else 12.0,
        step=1.0,
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Row 3: Antecedent Moisture & Temperature Indicators
with st.expander("3. Antecedent Drought & Fuel Indicators (7-day, 14-day, 30-day Lags)", expanded=True):
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#475569;'>Cumulative Rainfall (mm)</div>", unsafe_allow_html=True)
        v_r7 = st.number_input("rain_7d", value=preset_data["rain_7d"] if preset_data else 0.0, step=1.0)
        v_r14 = st.number_input("rain_14d", value=preset_data["rain_14d"] if preset_data else 2.0, step=1.0)
        v_r30 = st.number_input("rain_30d", value=preset_data["rain_30d"] if preset_data else 10.0, step=1.0)
    with col_l2:
        st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#475569;'>Rolling Mean Max Temp (°C)</div>", unsafe_allow_html=True)
        v_t7 = st.number_input("tmax_7d", value=preset_data["tmax_7d"] if preset_data else 31.0, step=0.5)
        v_t14 = st.number_input("tmax_14d", value=preset_data["tmax_14d"] if preset_data else 30.0, step=0.5)
        v_t30 = st.number_input("tmax_30d", value=preset_data["tmax_30d"] if preset_data else 28.5, step=0.5)
    with col_l3:
        st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#475569;'>Rolling Min Humidity (%)</div>", unsafe_allow_html=True)
        v_rh7 = st.number_input("rhmin_7d", value=preset_data["rhmin_7d"] if preset_data else 22.0, step=1.0)
        v_rh14 = st.number_input("rhmin_14d", value=preset_data["rhmin_14d"] if preset_data else 20.0, step=1.0)
        v_rh30 = st.number_input("rhmin_30d", value=preset_data["rhmin_30d"] if preset_data else 18.0, step=1.0)

# Build feature values (solar is display-only and deliberately excluded).
feature_values = {
    "lat_cell": lat_cell,
    "lon_cell": lon_cell,
    "T2M": v_t2m,
    "T2M_MAX": v_t2m_max,
    "RH2M": v_rh2m,
    "WS2M": v_ws2m,
    "PRECTOTCORR": v_prec,
    "GWETROOT": v_gwet,
    "rain_7d": v_r7,
    "rain_14d": v_r14,
    "rain_30d": v_r30,
    "tmax_7d": v_t7,
    "tmax_14d": v_t14,
    "tmax_30d": v_t30,
    "rhmin_7d": v_rh7,
    "rhmin_14d": v_rh14,
    "rhmin_30d": v_rh30,
    "dry_days": v_dry,
    "doy_sin": doy_sin,
    "doy_cos": doy_cos,
}

# Model-card order, not widget order.
feature_order = list(model_card.get("feature_names", MODEL_FEATURES))
input_feature_map = {name: feature_values[name] for name in feature_order}

# Non-blocking warnings: study-area geometry and model-period provenance.
area_info = app_core.check_training_area(in_lat, in_lon, in_date)
period_note = app_core.data_period_note(in_date)

if not area_info["in_box"]:
    render_callout(
        title="Outside the Model's Study Area",
        message=(
            area_info["warnings"][0]
            if area_info["warnings"]
            else "The snapped grid cell is outside the model's study area."
        ),
        variant="warning",
    )

note_palette = {
    "training period": (COLORS["brand_subtle"], COLORS["brand_accent"]),
    "calibration year": (COLORS["risk_moderate_bg"], COLORS["risk_moderate"]),
    "original test period": (COLORS["risk_low_bg"], COLORS["risk_low"]),
    "new data, never used for this model": (
        COLORS["risk_high_bg"],
        COLORS["risk_high"],
    ),
}
note_bg, note_fg = note_palette.get(
    period_note,
    (COLORS["surface_subtle"], COLORS["text_secondary"]),
)
st.markdown(
    f"""
    <div style="
        display: inline-block;
        margin-top: 4px;
        padding: 3px 9px;
        border-radius: 999px;
        background-color: {note_bg};
        color: {note_fg};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    ">
        Model period: {period_note}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
run_predict = st.button(
    "Evaluate Wildfire Risk",
    use_container_width=True,
    disabled=model_state == "ERROR",
)

# Prediction Result or Waiting Staging State
render_section_header(
    title="Inference Results & Diagnostics",
    subtitle="Model evaluation output with contributing factors and advisory classification",
)

if run_predict and model_state != "ERROR":
    try:
        inference = predict_fire_risk(model, input_feature_map)
        risk_cat = inference["risk_level"]
        prob_score = inference["probability"]
        cat_cfg = RISK_LEVELS.get(risk_cat, RISK_LEVELS["Moderate"])

        if model_state == "REAL":
            probability_line = (
                "<div style='font-size: 1.05rem; font-weight: 600; "
                f"color: {COLORS['text_primary']};'>Calibrated 3-day probability: "
                f"{prob_score * 100:.1f}%</div>"
            )
        else:
            probability_line = (
                "<div style='font-size: 1.0rem; font-weight: 600; "
                f"color: {COLORS['text_primary']};'>Demo heuristic, not a trained "
                f"model ({prob_score * 100:.1f}%)</div>"
            )

        col_r1, col_r2 = st.columns([1.1, 1.9])

        with col_r1:
            st.markdown(
                f"""
                <div style="
                    background-color: {cat_cfg['bg']};
                    border: 1px solid {cat_cfg['color']}50;
                    border-top: 4px solid {cat_cfg['color']};
                    border-radius: 8px;
                    padding: 22px;
                    text-align: center;
                ">
                    <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: {cat_cfg['color']};">
                        Risk Assessment
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {cat_cfg['color']}; margin: 6px 0;">
                        {risk_cat} Risk
                    </div>
                    {probability_line}
                    <div style="font-size: 0.78rem; color: {COLORS['text_secondary']}; margin-top: 8px; line-height: 1.4;">
                        {cat_cfg['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r2:
            st.markdown(
                f"""
                <div style="
                    background-color: {COLORS['surface_white']};
                    border: 1px solid {COLORS['border_light']};
                    border-radius: 8px;
                    padding: 20px 24px;
                ">
                    <div style="font-size: 0.9rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 8px;">
                        Environmental Drivers & Condition Assessment
                    </div>
                    <ul style="font-size: 0.84rem; color: {COLORS['text_secondary']}; line-height: 1.7; margin: 0; padding-left: 18px;">
                        <li><b>Prediction Target:</b> at least one fire in this grid cell within the next 3 days</li>
                        <li><b>Spatial Domain:</b> {in_lat:.3f}°N, {in_lon:.3f}°E on {in_date.strftime('%Y-%m-%d')} (Day of year {doy}) — grid cell {lat_cell:.3f}, {lon_cell:.3f}</li>
                        <li><b>Thermal Forcing:</b> Max Temperature at {v_t2m_max:.1f}°C (7d moving avg: {v_t7:.1f}°C)</li>
                        <li><b>Vapor Pressure & Aridity:</b> Relative Humidity at {v_rh2m:.1f}%, 30-day min humidity at {v_rh30:.1f}%</li>
                        <li><b>Antecedent Moisture Deficit:</b> {v_dry:.0f} dry days, 14d rainfall total at {v_r14:.1f} mm, root soil moisture at {v_gwet:.2f}</li>
                        <li><b>Display-only Variable:</b> Solar radiation at {v_rad:.1f} MJ/m² (not used by the model)</li>
                        <li><b>Model Artifact:</b> v{model_version} • {feature_count} features • <code>{model_file_label}</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.error(f"Inference pipeline execution error: {exc}")
elif model_state == "ERROR":
    st.info("Inference is disabled until the model artifact loads successfully.")
else:
    st.info("Select or adjust environmental parameters above and click 'Evaluate Wildfire Risk' to trigger inference.")
