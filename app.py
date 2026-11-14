"""
FIREWATCH AI — Fire Risk Predictor (landing page).

The first screen is deliberately prediction-first: pick a place and a date,
describe today's fire weather in six inputs, press one button, and get a
calibrated fire-risk probability with the drivers behind it.  Analytical
modules remain one click away through the sidebar and the links at the bottom.

The heavy lifting lives in ``dashboard.predictor`` (pure, tested) and
``app_core`` (model contract, feature building, scoring).
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Fire Risk Predictor | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "FIREWATCH AI — Wildfire Risk Prediction & Impact Assessment\n"
            "Academic Data Science Project (2024–2026)\n"
            "Data: NASA FIRMS (VIIRS) + NASA POWER"
        ),
    },
)

import app_core
from dashboard.components import (
    inject_global_css,
    render_sidebar,
    render_callout,
)
from dashboard.config import (
    COLORS,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    PROJECT_ROOT,
    RISK_LEVELS,
)
from dashboard.model_loader import load_model
from dashboard.predictor import (
    PRESET_ORDER,
    PRESETS,
    baseline_comparison,
    build_feature_row,
    changed_features,
    compare_presets,
    condition_summary,
    feature_label,
    feature_sensitivity,
    format_pp,
    guidance_for,
    metric_drift,
    prediction_payload,
    read_evaluation_metrics,
    read_share_params,
    risk_narrative,
    score_conditions,
    share_query,
    share_url,
)

# ---------------------------------------------------------------------------
# Fire-forward landing-page styling
# ---------------------------------------------------------------------------
# Kept local to the landing page so the rest of the platform keeps its calm
# green environmental-intelligence palette.
_FIRE_CSS = """
<style>
.fwx-hero {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    padding: 38px 42px;
    margin: 4px 0 26px 0;
    background:
        radial-gradient(circle at 88% 12%, rgba(249, 115, 22, 0.42), transparent 42%),
        radial-gradient(circle at 6% 96%, rgba(220, 38, 38, 0.28), transparent 40%),
        linear-gradient(120deg, #160A05 0%, #431407 48%, #7C2D12 100%);
    color: #FFF7ED;
    box-shadow: 0 18px 40px -22px rgba(124, 45, 18, 0.55);
}
.fwx-hero-grid {
    display: flex;
    gap: 28px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
}
.fwx-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #FDBA74;
}
.fwx-hero h1 {
    color: #FFF7ED !important;
    font-size: 2.55rem !important;
    line-height: 1.05 !important;
    margin: 10px 0 10px 0 !important;
    letter-spacing: -0.035em !important;
}
.fwx-hero p {
    color: #FED7AA !important;
    font-size: 0.98rem;
    line-height: 1.65;
    max-width: 620px;
    margin: 0;
}
.fwx-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}
.fwx-chip {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 5px 11px;
    border-radius: 999px;
    border: 1px solid rgba(253, 186, 116, 0.38);
    background: rgba(255, 247, 237, 0.07);
    color: #FFEDD5;
    letter-spacing: 0.01em;
}
.fwx-hero-card {
    min-width: 236px;
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid rgba(253, 186, 116, 0.28);
    background: rgba(15, 8, 4, 0.45);
    backdrop-filter: blur(4px);
}
.fwx-hero-card .cap {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #FDBA74;
    font-weight: 700;
}
.fwx-hero-card .big {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFF7ED;
    line-height: 1.1;
    margin: 4px 0 2px 0;
    letter-spacing: -0.03em;
}
.fwx-hero-card .small {
    font-size: 0.76rem;
    color: #FDBA74;
    line-height: 1.55;
}
.fwx-step {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2px;
}
.fwx-step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #C2410C;
    color: #FFF7ED;
    font-size: 0.78rem;
    font-weight: 800;
    flex: 0 0 26px;
}
.fwx-step-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
}
.fwx-step-sub {
    font-size: 0.79rem;
    color: #64748B;
    margin: 2px 0 12px 36px;
}
.fwx-gauge {
    width: 176px;
    height: 176px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 6px auto 14px auto;
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
}
.fwx-gauge-inner {
    width: 138px;
    height: 138px;
    border-radius: 50%;
    background: #FFFFFF;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
}
.fwx-gauge-num {
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
}
.fwx-gauge-cap {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    color: #64748B;
    margin-top: 5px;
}
.fwx-band {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.fwx-prob {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 10px 0 2px 0;
}
.fwx-prob-cap {
    font-size: 0.74rem;
    color: #64748B;
    line-height: 1.4;
}
.fwx-src {
    display: inline-block;
    margin-top: 12px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 4px;
    background: #F1F5F9;
    color: #475569;
}
.fwx-how {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 16px 14px 16px;
    height: 100%;
}
.fwx-how .n {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: #C2410C;
    text-transform: uppercase;
}
.fwx-how .t {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0F172A;
    margin: 3px 0 4px 0;
}
.fwx-how .d {
    font-size: 0.76rem;
    color: #64748B;
    line-height: 1.5;
}
.fwx-disclaimer {
    font-size: 0.74rem;
    color: #64748B;
    line-height: 1.55;
    border-left: 3px solid #FDBA74;
    padding: 2px 0 2px 12px;
    margin-top: 8px;
}
/* Fire-orange primary call to action */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #EA580C, #C2410C) !important;
    border: 1px solid #9A3412 !important;
    color: #FFF7ED !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
    padding: 13px 22px !important;
    box-shadow: 0 10px 22px -12px rgba(194, 65, 12, 0.75) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #F97316, #EA580C) !important;
    border-color: #7C2D12 !important;
}
</style>
"""

inject_global_css()
st.markdown(_FIRE_CSS, unsafe_allow_html=True)
render_sidebar()

# ---------------------------------------------------------------------------
# Shareable state (?lat=&lon=&date=&scenario=&run=1)
# ---------------------------------------------------------------------------
_shared = read_share_params(
    st.query_params.to_dict(), default_date=datetime.date.today()
)
_default_scenario = (
    PRESET_ORDER.index(_shared["scenario"])
    if _shared["scenario"] in PRESET_ORDER
    else 0
)

# ---------------------------------------------------------------------------
# Model + reference facts
# ---------------------------------------------------------------------------
bundle = load_model()
card = bundle.card or {}
metrics = card.get("metrics", {})
base_rate = float(metrics.get("base_rate", 0.0237))
pr_auc = metrics.get("pr_auc")
top_share = metrics.get("top_decile_fire_share")
model_version = card.get("model_version", "unknown")
model_ready = bundle.status in {"REAL", "DEMO"}
_evaluation_metrics = read_evaluation_metrics(
    PROJECT_ROOT / "report" / "evaluation_summary.json"
)

_accuracy_line = (
    f"PR-AUC {pr_auc:.3f} · {top_share:.0%} of fires in the top decile"
    if pr_auc is not None and top_share is not None
    else "Trained on 2018–2021 · calibrated on 2021"
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="fwx-hero">
        <div class="fwx-hero-grid">
            <div>
                <span class="fwx-eyebrow">🔥 Wildfire risk prediction · Northern Algeria</span>
                <h1>Will a fire start here?</h1>
                <p>
                    Choose a place and a date, describe today's fire weather, and FIREWATCH
                    estimates the probability of an observed fire in the next
                    <b>3 days</b>. One calibrated score, the conditions that drive it,
                    and clear guidance — no dashboard clutter.
                </p>
                <div class="fwx-chips">
                    <span class="fwx-chip">NASA POWER weather</span>
                    <span class="fwx-chip">NASA FIRMS fire labels</span>
                    <span class="fwx-chip">Calibrated gradient boosting</span>
                    <span class="fwx-chip">20 engineered features</span>
                </div>
            </div>
            <div class="fwx-hero-card">
                <div class="cap">Deployed model</div>
                <div class="big">v{model_version}</div>
                <div class="small">
                    {_accuracy_line}<br>
                    Base fire rate <b>{base_rate * 100:.2f}%</b> of cell-days<br>
                    Horizon <b>H = 3 days</b>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_ready:
    render_callout(
        title="Model artifact unavailable",
        message=bundle.error or "The trained model could not be loaded.",
        variant="danger",
    )

# ---------------------------------------------------------------------------
# Step 1 — location & date
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        """
        <div class="fwx-step">
            <span class="fwx-step-num">1</span>
            <span class="fwx-step-title">Where and when?</span>
        </div>
        <div class="fwx-step-sub">The model scores a 0.5° × 0.625° NASA POWER grid cell for one date.</div>
        """,
        unsafe_allow_html=True,
    )
    loc1, loc2, loc3 = st.columns([1, 1, 1])
    with loc1:
        in_lat = st.number_input(
            "Latitude (°N)",
            min_value=float(LAT_MIN),
            max_value=float(LAT_MAX),
            value=float(_shared["lat"]),
            step=0.5,
            key="in_lat",
            help="Study area: 32.0°N–37.5°N (northern Algeria).",
        )
    with loc2:
        in_lon = st.number_input(
            "Longitude (°E)",
            min_value=float(LON_MIN),
            max_value=float(LON_MAX),
            value=float(_shared["lon"]),
            step=0.625,
            key="in_lon",
            help="Study area: -2.0°E–9.0°E.",
        )
    with loc3:
        in_date = st.date_input(
            "Date",
            value=_shared["date"],
            key="in_date",
            help="Controls the seasonal features (day-of-year sine and cosine).",
        )

    lat_cell, lon_cell = app_core.snap_to_grid(in_lat, in_lon)
    area = app_core.check_training_area(in_lat, in_lon, in_date)
    area_note = (
        f"Grid cell <b>{lat_cell:.3f}°N, {lon_cell:.3f}°E</b> · day {in_date.timetuple().tm_yday} of the year"
        if area["in_box"]
        else f"⚠️ Grid cell <b>{lat_cell:.3f}°N, {lon_cell:.3f}°E</b> is outside the model's training area."
    )
    st.markdown(
        f"<div style='font-size:0.78rem;color:{COLORS['text_muted']};margin-top:2px;'>{area_note}</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Step 2 — fire weather
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        """
        <div class="fwx-step">
            <span class="fwx-step-num">2</span>
            <span class="fwx-step-title">What is the fire weather?</span>
        </div>
        <div class="fwx-step-sub">Pick a realistic scenario, then tune the six drivers that matter most.</div>
        """,
        unsafe_allow_html=True,
    )

    scenario = st.radio(
        "Scenario",
        PRESET_ORDER,
        index=_default_scenario,
        horizontal=True,
        key="scenario",
        label_visibility="collapsed",
    )
    preset = PRESETS[scenario]
    st.markdown(
        f"<div style='font-size:0.78rem;color:{COLORS['text_muted']};margin:-4px 0 12px 0;'>"
        f"{preset['description']}</div>",
        unsafe_allow_html=True,
    )

    slug = scenario.lower().replace(" ", "_").replace("-", "_")
    w1, w2, w3 = st.columns(3)
    with w1:
        t2m_max = st.number_input(
            "Daily max temperature (°C)",
            min_value=-5.0,
            max_value=55.0,
            value=float(preset["T2M_MAX"]),
            step=0.5,
            key=f"tmax_{slug}",
        )
        rh2m = st.number_input(
            "Relative humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(preset["RH2M"]),
            step=1.0,
            key=f"rh_{slug}",
        )
    with w2:
        ws2m = st.number_input(
            "Wind speed (m/s)",
            min_value=0.0,
            max_value=40.0,
            value=float(preset["WS2M"]),
            step=0.5,
            key=f"wind_{slug}",
        )
        precip = st.number_input(
            "Rain today (mm)",
            min_value=0.0,
            max_value=200.0,
            value=float(preset["PRECTOTCORR"]),
            step=0.5,
            key=f"rain_{slug}",
        )
    with w3:
        gwetroot = st.number_input(
            "Root-zone soil wetness (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=float(preset["GWETROOT"]),
            step=0.05,
            key=f"soil_{slug}",
            help="0 = bone dry, 1 = saturated. NASA POWER GWETROOT.",
        )
        dry_days = st.number_input(
            "Consecutive dry days",
            min_value=0.0,
            max_value=365.0,
            value=float(preset["dry_days"]),
            step=1.0,
            key=f"dry_{slug}",
            help="Days since more than 1 mm of rain.",
        )

    antecedent: dict[str, float] = {}
    with st.expander("Advanced: override the 7 / 14 / 30-day history", expanded=False):
        st.caption(
            "By default the model assumes today's weather is representative of the last "
            "30 days (a transparent persistence assumption). Override the windows below "
            "if you know the real antecedent conditions."
        )
        use_history = st.checkbox("Override antecedent history", key="adv_on")
        if use_history:
            h1, h2, h3 = st.columns(3)
            with h1:
                antecedent["rain_7d"] = st.number_input(
                    "Rain last 7 days (mm)",
                    min_value=0.0,
                    value=float(preset["PRECTOTCORR"]) * 7.0,
                    step=1.0,
                    key=f"r7_{slug}",
                )
                antecedent["rain_30d"] = st.number_input(
                    "Rain last 30 days (mm)",
                    min_value=0.0,
                    value=float(preset["PRECTOTCORR"]) * 30.0,
                    step=5.0,
                    key=f"r30_{slug}",
                )
            with h2:
                antecedent["tmax_7d"] = st.number_input(
                    "Mean max temp last 7 days (°C)",
                    value=float(preset["T2M_MAX"]),
                    step=0.5,
                    key=f"t7_{slug}",
                )
                antecedent["tmax_30d"] = st.number_input(
                    "Mean max temp last 30 days (°C)",
                    value=float(preset["T2M_MAX"]),
                    step=0.5,
                    key=f"t30_{slug}",
                )
            with h3:
                antecedent["rhmin_7d"] = st.number_input(
                    "Min humidity last 7 days (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(preset["RH2M"]),
                    step=1.0,
                    key=f"h7_{slug}",
                )
                antecedent["rhmin_30d"] = st.number_input(
                    "Min humidity last 30 days (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(preset["RH2M"]),
                    step=1.0,
                    key=f"h30_{slug}",
                )

    with st.expander("Advanced: pull real weather from NASA POWER", expanded=False):
        st.caption(
            "Fetch the last ~4 months of daily weather for this grid cell and let the "
            "pipeline compute the rolling features exactly as it did in training. "
            "Requires internet access; the manual inputs above are ignored while a "
            "live row is loaded."
        )
        live_key = (
            round(float(in_lat), 4),
            round(float(in_lon), 4),
            in_date.isoformat(),
        )
        live_state = st.session_state.get("live_weather")
        lc1, lc2 = st.columns([1.4, 1])
        with lc1:
            fetch_live = st.button(
                "Fetch NASA POWER history",
                key="fetch_live",
                use_container_width=True,
            )
        with lc2:
            clear_live = st.button(
                "Clear live data",
                key="clear_live",
                use_container_width=True,
                disabled=live_state is None,
            )

        if clear_live:
            st.session_state.pop("live_weather", None)
            live_state = None
            st.success("Live weather cleared — manual inputs are active again.")

        if fetch_live:
            try:
                with st.spinner("Contacting NASA POWER …"):
                    history = app_core.fetch_power_history(
                        in_lat, in_lon, in_date, days=120
                    )
                    live_row = app_core.build_live_features(
                        history,
                        in_lat,
                        in_lon,
                        in_date,
                        card=bundle.card,
                    )
                st.session_state["live_weather"] = {
                    "key": live_key,
                    "features": {
                        name: float(value) for name, value in live_row.iloc[0].items()
                    },
                    "days": len(history),
                }
                live_state = st.session_state["live_weather"]
                st.success(f"Loaded {len(history)} days of NASA POWER weather.")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                st.error(f"NASA POWER fetch failed: {exc}")

        if live_state is not None:
            if live_state.get("key") == live_key:
                st.info(
                    f"Live NASA POWER row loaded ({live_state.get('days', 0)} days of history). "
                    "Predictions below use it instead of the manual inputs."
                )
            else:
                st.warning(
                    "Live data was fetched for a different location or date. "
                    "Fetch again or clear it to use the manual inputs."
                )

# Build the feature contract used for scoring
live_match = (
    st.session_state.get("live_weather")
    if st.session_state.get("live_weather", {}).get("key")
    == (round(float(in_lat), 4), round(float(in_lon), 4), in_date.isoformat())
    else None
)
using_live = live_match is not None

if using_live:
    features = dict(live_match["features"])
else:
    features = build_feature_row(
        in_lat,
        in_lon,
        in_date,
        t2m_max=float(t2m_max),
        rh2m=float(rh2m),
        ws2m=float(ws2m),
        precip=float(precip),
        gwetroot=float(gwetroot),
        dry_days=float(dry_days),
        card=bundle.card,
        antecedent=antecedent,
    )

# ---------------------------------------------------------------------------
# Predict action
# ---------------------------------------------------------------------------
predict_col, note_col = st.columns([1.25, 2])
with predict_col:
    run_prediction = st.button(
        "🔥 Predict fire risk",
        type="primary",
        use_container_width=True,
        disabled=not model_ready,
        key="predict_button",
    )

# A shared link with run=1 scores once on arrival, then behaves like a normal
# session so later edits still show the "inputs changed" warning.
if _shared["run"] and not st.session_state.get("share_consumed"):
    st.session_state["share_consumed"] = True
    run_prediction = run_prediction or model_ready
with note_col:
    st.markdown(
        f"<div style='font-size:0.76rem;color:{COLORS['text_muted']};padding-top:14px;line-height:1.5;'>"
        "Every prediction is a calibrated probability for the <b>next 3 days</b>, "
        "not a guarantee. Re-run after changing any input.</div>",
        unsafe_allow_html=True,
    )

if run_prediction:
    st.session_state["has_prediction"] = True
    st.session_state["share_consumed"] = True
    st.session_state["prediction_snapshot"] = {
        "features": dict(features),
        "lat": float(in_lat),
        "lon": float(in_lon),
        "lat_cell": float(lat_cell),
        "lon_cell": float(lon_cell),
        "date": in_date,
        "scenario": scenario,
        "using_live": using_live,
    }
    st.query_params.update(
        share_query(
            lat=in_lat,
            lon=in_lon,
            when=in_date,
            scenario=scenario,
            run=True,
        )
    )

# ---------------------------------------------------------------------------
# Step 3 — result
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        """
        <div class="fwx-step">
            <span class="fwx-step-num">3</span>
            <span class="fwx-step-title">Prediction</span>
        </div>
        <div class="fwx-step-sub">Calibrated probability, the drivers behind it, and what to do next.</div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("has_prediction") or not st.session_state.get(
        "prediction_snapshot"
    ):
        st.info(
            "Set the location and fire weather above, then press **Predict fire risk**. "
            "The score, its drivers and field guidance will appear here."
        )
    elif not model_ready:
        st.warning("Prediction is disabled until the model artifact loads successfully.")
    else:
        snapshot = st.session_state["prediction_snapshot"]
        scoring_features = snapshot["features"]
        try:
            prediction = score_conditions(bundle, scoring_features)
        except Exception as exc:  # noqa: BLE001 - keep the page alive
            st.error(f"Inference failed: {exc}")
        else:
            stale = changed_features(features, scoring_features)
            if stale:
                shown = ", ".join(feature_label(name) for name in stale[:3])
                remainder = f" and {len(stale) - 3} more" if len(stale) > 3 else ""
                st.warning(
                    f"Inputs changed since this prediction ({shown}{remainder}). "
                    "The score below is kept stable on purpose — press "
                    "**Predict fire risk** to refresh it."
                )
            cfg = RISK_LEVELS.get(prediction.risk_level, RISK_LEVELS["Moderate"])
            index = max(0.0, min(100.0, float(prediction.risk_index)))
            comparison = baseline_comparison(prediction.probability, base_rate)
            multiple = comparison["multiple"]
            multiple_text = (
                f"{multiple:.1f}× the historical base rate"
                if np.isfinite(multiple)
                else "vs. historical base rate"
            )
            source_text = (
                f"Calibrated model v{model_version}"
                if prediction.source == "REAL"
                else "Demo heuristic (model missing)"
            )
            summary = condition_summary(scoring_features)

            left, right = st.columns([1, 2.05])

            with left:
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <div class="fwx-gauge" style="background: conic-gradient({cfg['color']} {index * 3.6:.1f}deg, #E2E8F0 0deg);">
                            <div class="fwx-gauge-inner">
                                <span class="fwx-gauge-num" style="color:{cfg['color']};">{index:.0f}</span>
                                <span class="fwx-gauge-cap">RISK INDEX</span>
                            </div>
                        </div>
                        <div class="fwx-band" style="background:{cfg['bg']};color:{cfg['color']};">{prediction.risk_level} risk</div>
                        <div class="fwx-prob" style="color:{cfg['color']};">{prediction.percent:.2f}%</div>
                        <div class="fwx-prob-cap">probability of an observed fire<br>within the next 3 days</div>
                        <div class="fwx-src">{source_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div style="margin-top:14px;padding-top:12px;border-top:1px solid {COLORS['border_light']};
                                font-size:0.76rem;color:{COLORS['text_secondary']};line-height:1.6;">
                        <b>{multiple_text}</b><br>
                        Historical base rate: {base_rate * 100:.2f}% of cell-days<br>
                        Grid cell: {snapshot['lat_cell']:.3f}°N, {snapshot['lon_cell']:.3f}°E · {snapshot['date']:%Y-%m-%d}
                        {(' · live NASA POWER' if snapshot['using_live'] else '')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    f"""
                    <div style="font-size:1.02rem;font-weight:700;color:{COLORS['text_primary']};margin-bottom:4px;">
                        {risk_narrative(prediction.risk_label)}
                    </div>
                    <div style="font-size:0.8rem;color:{COLORS['text_muted']};margin-bottom:14px;">
                        Max {summary['max_temp']:.0f} °C · {summary['humidity']:.0f}% humidity ·
                        {summary['wind']:.1f} m/s wind · {summary['rain_7d']:.0f} mm rain (7 d) ·
                        {summary['dry_days']:.0f} dry days · soil {summary['soil_wetness']:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                drivers = (
                    feature_sensitivity(bundle, scoring_features, top_n=7)
                    if prediction.source == "REAL"
                    else pd.DataFrame()
                )
                if drivers.empty:
                    st.caption("No single driver dominates these conditions.")
                else:
                    plot = drivers.sort_values("delta_pp", ascending=True)
                    colors = [
                        COLORS["risk_high"] if value > 0 else COLORS["risk_low"]
                        for value in plot["delta_pp"]
                    ]
                    fig = go.Figure(
                        go.Bar(
                            x=plot["delta_pp"],
                            y=plot["label"],
                            orientation="h",
                            marker_color=colors,
                            text=[f"{value:+.2f} pp" for value in plot["delta_pp"]],
                            textposition="outside",
                            cliponaxis=False,
                            customdata=plot[["value", "reference"]].to_numpy(),
                            hovertemplate=(
                                "%{y}<br>current: %{customdata[0]:.2f}"
                                " → fire-suppressing: %{customdata[1]:.2f}"
                                "<br>score change: %{x:+.3f} pp<extra></extra>"
                            ),
                        )
                    )
                    fig.update_layout(
                        template="plotly_white",
                        title=dict(
                            text=(
                                "<b>What moves the score</b><br>"
                                "<span style='font-size:10px;color:#64748B;'>"
                                "Change in probability when one driver is replaced by a fire-suppressing value"
                                "</span>"
                            ),
                            x=0,
                            xanchor="left",
                            font=dict(size=12, color=COLORS["text_primary"]),
                        ),
                        height=max(230, 34 * len(plot) + 110),
                        margin=dict(l=10, r=34, t=64, b=34),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, Segoe UI, sans-serif", size=11),
                        xaxis=dict(
                            title="Δ probability (percentage points)",
                            gridcolor="#F1F5F9",
                            zeroline=True,
                            zerolinecolor="#CBD5E1",
                            tickfont=dict(size=10),
                        ),
                        yaxis=dict(tickfont=dict(size=10), title=None),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                guidance = guidance_for(prediction.risk_label)
                items = "".join(f"<li>{item}</li>" for item in guidance)
                st.markdown(
                    f"""
                    <div style="background:{cfg['bg']};border:1px solid {cfg['color']}33;
                                border-radius:10px;padding:14px 18px;margin-top:2px;">
                        <div style="font-size:0.74rem;font-weight:800;letter-spacing:0.07em;
                                    text-transform:uppercase;color:{cfg['color']};margin-bottom:6px;">
                            Recommended actions
                        </div>
                        <ul style="margin:0;padding-left:18px;font-size:0.82rem;
                                   color:{COLORS['text_secondary']};line-height:1.7;">
                            {items}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="fwx-disclaimer">
                    <b>How to read this:</b> the score is a calibrated probability estimated from
                    2018–2021 weather and satellite fire detections, with a 3-day horizon. It is a
                    decision-support signal, not an official emergency forecast. Local terrain,
                    fuel load and human activity can change the real outcome. In an emergency,
                    contact civil protection.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(
                "Share, bookmark or export this prediction", expanded=False
            ):
                st.caption(
                    "Append this query string to the app URL to reopen the page with the same "
                    "location, date and scenario. `run=1` re-scores automatically on arrival."
                )
                st.code(
                    share_url(
                        "",
                        share_query(
                            lat=snapshot["lat"],
                            lon=snapshot["lon"],
                            when=snapshot["date"],
                            scenario=snapshot["scenario"],
                        ),
                    ),
                    language="text",
                )
                st.caption(
                    "The JSON record below captures the inputs, the score and the drivers "
                    "so a prediction can be archived or cited in a report."
                )
                payload = prediction_payload(
                    prediction=prediction,
                    features=scoring_features,
                    lat=snapshot["lat"],
                    lon=snapshot["lon"],
                    when=snapshot["date"],
                    scenario=snapshot["scenario"],
                    model_version=model_version,
                    base_rate=base_rate,
                    using_live=snapshot["using_live"],
                    sensitivity=drivers,
                )
                st.download_button(
                    "⬇️ Download prediction as JSON",
                    data=json.dumps(payload, indent=2, ensure_ascii=False),
                    file_name=(
                        "firewatch_prediction_"
                        f"{snapshot['date']:%Y-%m-%d}_"
                        f"{snapshot['lat']:.2f}_{snapshot['lon']:.2f}.json"
                    ),
                    mime="application/json",
                    key="download_prediction_json",
                )

# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
with st.expander("Compare the four scenarios at this location", expanded=False):
    st.caption(
        "Same grid cell and date, four standard fire-weather setups. This isolates how "
        "the calibrated model responds to weather alone — the manual values in step 2 "
        "are intentionally ignored here."
    )
    if not model_ready:
        st.info("The comparison becomes available once the model artifact loads.")
    else:
        comparison = compare_presets(
            bundle, in_lat, in_lon, in_date, card=bundle.card
        )
        display = comparison[
            ["scenario", "description", "percent", "risk_level", "multiple"]
        ].copy()
        display.columns = [
            "Scenario",
            "Description",
            "Probability (%)",
            "Band",
            "× base rate",
        ]
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Probability (%)": st.column_config.ProgressColumn(
                    "Probability (%)",
                    help="Calibrated probability of an observed fire in the next 3 days.",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.2f%%",
                ),
                "× base rate": st.column_config.NumberColumn(
                    "× base rate",
                    help="How many times the historical base rate this score represents.",
                    format="%.1f×",
                ),
            },
        )

        ordered = comparison.sort_values("percent")
        fig = go.Figure(
            go.Bar(
                x=ordered["percent"],
                y=ordered["scenario"],
                orientation="h",
                marker_color=[
                    RISK_LEVELS.get(level, RISK_LEVELS["Moderate"])["color"]
                    for level in ordered["risk_level"]
                ],
                text=[f"{value:.2f}%" for value in ordered["percent"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.2f}% probability<extra></extra>",
            )
        )
        fig.add_vline(
            x=base_rate * 100,
            line_dash="dot",
            line_color=COLORS["text_muted"],
            annotation_text=f"base rate {base_rate * 100:.2f}%",
            annotation_position="top",
            annotation_font_size=9,
        )
        fig.update_layout(
            template="plotly_white",
            height=260,
            margin=dict(l=10, r=54, t=28, b=34),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Segoe UI, sans-serif", size=11),
            xaxis=dict(
                title="Probability of an observed fire in 3 days (%)",
                gridcolor="#F1F5F9",
                tickfont=dict(size=10),
            ),
            yaxis=dict(tickfont=dict(size=10), title=None),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Presets only prefill the form; every value above remains editable in step 2."
        )

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
how_cols = st.columns(4)
how_steps = [
    ("01", "Weather", "NASA POWER supplies daily temperature, humidity, wind, rain and soil wetness for every grid cell."),
    ("02", "Fuel memory", "Rolling 7, 14 and 30-day windows turn today's weather into a drought and fuel-aridity signal."),
    ("03", "Calibrated model", "A gradient-boosting classifier, isotonic-calibrated on a held-out year, outputs a probability."),
    ("04", "Risk + action", "The probability maps to a risk band, the drivers behind it, and practical field guidance."),
]
for column, (number, title, description) in zip(how_cols, how_steps):
    with column:
        st.markdown(
            f"""
            <div class="fwx-how">
                <div class="n">Step {number}</div>
                <div class="t">{title}</div>
                <div class="d">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Model card + navigation
# ---------------------------------------------------------------------------
with st.expander("Model card, metrics and data provenance", expanded=False):
    st.markdown(
        f"""
        <div style="font-size:0.82rem;color:{COLORS['text_secondary']};line-height:1.7;">
            <b>Artifact:</b> <code>models/fire_model.joblib</code> · version <b>{model_version}</b><br>
            <b>Training:</b> 2018–2020 · <b>calibration:</b> 2021 · <b>test:</b> 2022–2024 (chronological split)<br>
            <b>Target:</b> observed fire in the next 3 days (binary, highly imbalanced)<br>
            <b>Features:</b> 20 — location, 6 daily weather variables, 9 rolling-window features,
            dry-spell length and two seasonal harmonics.
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_rows = [
        ("Precision-Recall AUC", f"{pr_auc:.4f}" if pr_auc is not None else "—"),
        ("Base fire rate", f"{base_rate * 100:.2f}%"),
        (
            "Brier score",
            f"{metrics.get('brier'):.5f}" if metrics.get("brier") is not None else "—",
        ),
        (
            "Brier skill score",
            f"{metrics.get('brier_skill'):.4f}"
            if metrics.get("brier_skill") is not None
            else "—",
        ),
        (
            "Fires captured in top decile",
            f"{top_share:.1%}" if top_share is not None else "—",
        ),
        ("Test rows", f"{metrics.get('test_rows', 0):,}"),
    ]
    st.dataframe(
        pd.DataFrame(metric_rows, columns=["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "PR-AUC must be read against the base rate: a rare-event classifier at "
        f"{base_rate * 100:.2f}% positives is not judged with accuracy."
    )

    drift = metric_drift(metrics, _evaluation_metrics)
    if not drift.empty and drift["changed"].any():
        st.markdown("**Metric provenance**")
        st.dataframe(
            drift.rename(
                columns={
                    "metric": "Metric",
                    "model_card": "Model card snapshot",
                    "reproducible": "Reproducible evaluation",
                }
            )[["Metric", "Model card snapshot", "Reproducible evaluation"]],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "The deployed model is unchanged. The model card records the metrics "
            "measured when it was trained; `report/evaluation_summary.json` is "
            "regenerated by `notebook.py` on the current panel. Differences come "
            "from the evaluation data, not from a different model."
        )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='font-size:0.95rem;font-weight:700;color:{COLORS['text_primary']};margin-bottom:4px;'>"
    "Explore the rest of the platform</div>"
    f"<div style='font-size:0.79rem;color:{COLORS['text_muted']};margin-bottom:10px;'>"
    "The predictor is the front door; these modules hold the full analysis.</div>",
    unsafe_allow_html=True,
)
nav_cols = st.columns(5)
links = [
    ("pages/1_📊_Overview.py", "📊 Overview", "Macro fire metrics"),
    ("pages/2_🗺️_Geospatial_Monitoring.py", "🗺️ Geospatial", "Fire map & hotspots"),
    ("pages/3_📈_Data_Analytics.py", "📈 Analytics", "Weather vs. fire"),
    ("pages/4_🎯_Risk_Prediction.py", "🎯 Cockpit", "Full 20-feature mode"),
    ("pages/6_Impact_Assessment.py", "🧭 Impact", "Population exposure"),
]
for column, (path, label, caption) in zip(nav_cols, links):
    with column:
        st.page_link(path, label=label, use_container_width=True)
        st.caption(caption)
