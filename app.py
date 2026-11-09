"""FireWatch — single-page Streamlit dashboard for the wildfire-risk model.

Run with:
    streamlit run app.py

The UI is deliberately a thin layer over ``app_core.py``.  All model loading,
feature construction, scoring and sensitivity logic live in the core module so
they can be tested without Streamlit.

The page is intentionally focused: the map, weather inputs and risk result are
always on screen.  The sidebar carries only app metadata and the two controls
that matter here — the prediction mode and a model reload button.
"""

from __future__ import annotations

import html
import math
from datetime import date
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import app_core as core

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FireWatch · Wildfire Risk Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESETS = {
    "Algiers": (36.7538, 3.0588),
    "Oran": (35.6969, -0.6331),
    "Constantine": (36.3650, 6.6147),
    "Annaba": (36.9000, 7.7667),
    "Batna": (35.5550, 6.1741),
    "Tlemcen": (34.8783, -1.3150),
    "Custom": None,
}
DEFAULT_PRESET = "Algiers"
DEFAULT_DATE = date(2023, 7, 15)

MANUAL_MODE = "Manual weather"
LIVE_MODE = "Live NASA POWER history"

CSS = """
<style>
:root {
  --bg: #0b0f14;
  --panel: #131a23;
  --panel-2: #18212c;
  --border: #243040;
  --text: #e6edf3;
  --muted: #8b98a8;
  --accent: #ff6b35;
}
.stApp {
  background:
    radial-gradient(circle at 15% -10%, rgba(255,107,53,0.10), transparent 35%),
    radial-gradient(circle at 85% 0%, rgba(46,204,113,0.06), transparent 30%),
    #0b0f14;
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d131a 0%, #090d12 100%);
  border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
a { color: var(--accent); }
.brand {
  display: flex; align-items: center; gap: 0.55rem;
  font-size: 1.35rem; font-weight: 800; letter-spacing: 0.08em;
  margin-bottom: 0.1rem;
}
.brand span { color: var(--accent); }
.hero {
  padding: 1.35rem 1.5rem;
  border: 1px solid var(--border);
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,107,53,0.13), rgba(19,26,35,0.92) 48%);
  margin-bottom: 1.1rem;
  box-shadow: 0 18px 45px rgba(0,0,0,0.18);
}
.hero-kicker {
  color: var(--accent); font-size: 0.72rem; font-weight: 800;
  letter-spacing: 0.2em; text-transform: uppercase;
}
.hero h1 { margin: 0.25rem 0 0.35rem 0; font-size: 2.05rem; line-height: 1.1; }
.hero p { margin: 0; color: var(--muted); max-width: 920px; line-height: 1.55; }
.status-pill {
  display: inline-block; padding: 0.27rem 0.62rem; border-radius: 999px;
  font-size: 0.70rem; font-weight: 800; letter-spacing: 0.09em;
}
.panel-card {
  border: 1px solid var(--border); border-radius: 15px;
  padding: 1.05rem 1.15rem; background: var(--panel);
}
.risk-card {
  border: 1px solid var(--border); border-radius: 20px;
  padding: 1.35rem 1.45rem;
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}
.risk-top { display: flex; justify-content: space-between; gap: 1rem; align-items: center; }
.risk-badge {
  padding: 0.24rem 0.58rem; border-radius: 999px;
  font-size: 0.67rem; font-weight: 800; letter-spacing: 0.11em;
}
.risk-source { color: var(--muted); font-size: 0.74rem; text-align: right; }
.risk-level { font-size: 2.25rem; font-weight: 800; margin-top: 0.65rem; letter-spacing: -0.03em; line-height: 1; }
.risk-prob { font-size: 2.65rem; font-weight: 800; line-height: 1.05; margin-top: 0.25rem; }
.risk-sub { color: var(--muted); font-size: 0.82rem; }
.risk-bar {
  height: 10px; border-radius: 999px; background: #0a0e13;
  overflow: hidden; margin-top: 1rem; border: 1px solid var(--border);
}
.risk-bar > div { height: 100%; border-radius: 999px; }
.risk-index { color: var(--muted); font-size: 0.75rem; margin-top: 0.45rem; }
.muted { color: var(--muted); }
.small { font-size: 0.78rem; }
.footer { color: var(--muted); font-size: 0.74rem; text-align: center; padding-top: 2.2rem; line-height: 1.6; }
.stButton > button, .stFormSubmitButton > button {
  background: linear-gradient(135deg, #ff6b35, #e8491d);
  color: white; border: 0; border-radius: 11px; font-weight: 800;
  padding: 0.55rem 1rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { filter: brightness(1.08); }
[data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 13px; background: var(--panel);
}
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
hr { border-color: var(--border); }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_bundle(model_mtime: float, card_mtime: float) -> core.ModelBundle:
    # The mtimes are cache keys only; load_model_bundle reads the real paths.
    # This makes the app re-validate when either artifact changes on disk.
    return core.load_model_bundle()


@st.cache_data(show_spinner=False)
def load_panel_cached(path_str: str, mtime: float) -> tuple[pd.DataFrame | None, str | None]:
    path = Path(path_str)
    if not path.exists():
        return None, f"Processed panel not found at {path}."
    try:
        return pd.read_parquet(path), None
    except Exception as exc:  # noqa: BLE001 - show the real read error
        return None, str(exc)


@st.cache_data(show_spinner=False)
def cell_stats_cached(path_str: str, mtime: float) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return core.cell_fire_stats(pd.read_parquet(path))
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_power_cached(lat: float, lon: float, end_iso: str, days: int) -> pd.DataFrame:
    return core.fetch_power_history(
        lat, lon, date.fromisoformat(end_iso), days=days
    )


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    return html.escape(str(value))


def status_pill(bundle: core.ModelBundle) -> str:
    styles = {
        "REAL": ("REAL MODEL", "#2ecc71", "rgba(46,204,113,0.12)"),
        "DEMO": ("DEMO MODE", "#f1c40f", "rgba(241,196,15,0.12)"),
        "ERROR": ("MODEL ERROR", "#ff3b30", "rgba(255,59,48,0.12)"),
    }
    label, color, background = styles.get(
        bundle.status, ("UNKNOWN", "#8b98a8", "rgba(139,152,168,0.12)")
    )
    return (
        f'<span class="status-pill" style="color:{color};background:{background};'
        f'border:1px solid {color}55">{label}</span>'
    )


def hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{esc(kicker)}</div>
          <h1>{esc(title)}</h1>
          <p>{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_card(
    probability: float,
    label: str,
    index: float,
    source_status: str,
    source: str,
) -> None:
    color = core.risk_color(label)
    display = core.risk_display(label)
    badge = "MODEL OUTPUT" if source_status == "REAL" else "DEMO HEURISTIC"
    sub = (
        "calibrated probability of a fire-day in this cell"
        if source_status == "REAL"
        else "demo heuristic score — not a trained probability"
    )
    st.markdown(
        f"""
        <div class="risk-card" style="border-color:{color}66">
          <div class="risk-top">
            <span class="risk-badge" style="color:{color};background:{color}18;
                  border:1px solid {color}55">{badge}</span>
            <span class="risk-source">{esc(source)}</span>
          </div>
          <div class="risk-level" style="color:{color}">{esc(display)}</div>
          <div class="risk-prob" style="color:{color}">{esc(core.format_probability(probability))}</div>
          <div class="risk-sub">{esc(sub)}</div>
          <div class="risk-bar"><div style="width:{index:.1f}%;background:{color}"></div></div>
          <div class="risk-index">Risk index {index:.0f}/100 · card band: {esc(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_banner(bundle: core.ModelBundle) -> None:
    if bundle.status == "DEMO":
        st.warning(
            "**DEMO MODE** — the trained model artifact is missing. "
            "Predictions come from a transparent weather heuristic, not the "
            "calibrated model. Restore `data/processed/fire_model.joblib` to "
            "switch to real predictions."
        )
    elif bundle.status == "ERROR":
        st.error(
            "**MODEL ERROR** — predictions are disabled because the model "
            f"artifact and/or model card failed validation. {bundle.error or ''}"
        )


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _heat_color(fraction: float) -> str:
    if fraction < 0.25:
        return "#2ecc71"
    if fraction < 0.50:
        return "#f1c40f"
    if fraction < 0.75:
        return "#ff8c42"
    return "#ff3b30"


def build_map(
    center_lat: float,
    center_lon: float,
    *,
    selected: tuple[float, float] | None = None,
    risk_label: str | None = None,
    cell_stats: pd.DataFrame | None = None,
    zoom: int = 6,
) -> folium.Map:
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Maxar, Earthstar Geographics",
        name="Esri World Imagery",
        overlay=False,
        control=False,
    ).add_to(fmap)

    folium.Rectangle(
        bounds=[
            [core.config.LAT_MIN, core.config.LON_MIN],
            [core.config.LAT_MAX, core.config.LON_MAX],
        ],
        color="#ff6b35",
        weight=1.4,
        opacity=0.85,
        fill=False,
        dash_array="6 6",
        tooltip="Training study area (northern Algeria)",
    ).add_to(fmap)

    if cell_stats is not None and not cell_stats.empty:
        max_rate = max(float(cell_stats["fire_rate"].max()), 1e-9)
        for row in cell_stats.itertuples(index=False):
            rate = float(row.fire_rate)
            if rate <= 0:
                continue
            fraction = rate / max_rate
            radius = 2.5 + 13.0 * math.sqrt(fraction)
            folium.CircleMarker(
                location=[float(row.lat_cell), float(row.lon_cell)],
                radius=radius,
                color=None,
                weight=0,
                fill=True,
                fill_color=_heat_color(fraction),
                fill_opacity=0.34,
                tooltip=(
                    f"{row.lat_cell:.2f}, {row.lon_cell:.2f} · "
                    f"fire-day rate {rate:.2%}"
                ),
            ).add_to(fmap)

    if selected is not None:
        lat, lon = selected
        color = core.risk_color(risk_label) if risk_label else "#ff6b35"
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=(
                f"Selected cell · {core.risk_display(risk_label)}"
                if risk_label
                else "Selected location"
            ),
        ).add_to(fmap)
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:22px;line-height:22px;'
                    f'filter:drop-shadow(0 0 7px {color})">🔥</div>'
                ),
                icon_size=(24, 24),
                icon_anchor=(12, 12),
            ),
            tooltip="Selected location",
        ).add_to(fmap)

    return fmap


def render_map(fmap: folium.Map, height: int = 430) -> None:
    st_folium(
        fmap,
        height=height,
        use_container_width=True,
        returned_objects=[],
    )


# ---------------------------------------------------------------------------
# Sidebar — metadata and controls only
# ---------------------------------------------------------------------------

def render_sidebar(
    bundle: core.ModelBundle,
    summary: dict[str, Any],
) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand">🔥 <span>FIREWATCH</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Wildfire risk intelligence")
        st.markdown("---")

        st.markdown(status_pill(bundle), unsafe_allow_html=True)
        if bundle.card:
            st.caption(
                f"Model v{bundle.card.get('model_version', '—')} · "
                f"{bundle.card.get('n_features', '—')} features"
            )
            trained = str(bundle.card.get("trained_at", ""))[:10]
            if trained:
                st.caption(f"Trained {trained}")
        elif bundle.status == "DEMO":
            st.caption("No model card found; using built-in demo defaults.")

        st.markdown("---")
        st.markdown("**Prediction mode**")
        mode = st.radio(
            "Prediction mode",
            [MANUAL_MODE, LIVE_MODE],
            label_visibility="collapsed",
            key="pred_mode",
            help=(
                "Manual weather updates the score instantly. Live mode "
                "fetches 120 days of NASA POWER history for the selected cell."
            ),
        )

        st.markdown("---")
        if st.button("Reload model", width="stretch", key="reload_model"):
            # Drop the cached artifact validation and any stale result, then
            # rebuild the page from disk.
            load_bundle.clear()
            st.session_state.pop("last_prediction", None)
            st.rerun()

        st.markdown("---")
        st.caption("**Study area**")
        st.caption(
            f"{core.config.LAT_MIN}–{core.config.LAT_MAX}°N\n\n"
            f"{core.config.LON_MIN}–{core.config.LON_MAX}°E\n\n"
            f"{core.config.START_YEAR}–{core.config.END_YEAR}"
        )
        st.caption("**Sources**")
        st.caption("NASA FIRMS · NASA POWER")

        if summary.get("rows"):
            st.markdown("---")
            st.caption(
                f"{summary['rows']:,} cell-days · {summary['cells']} grid cells"
            )
        return mode


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def make_signature(
    lat: float,
    lon: float,
    when: date,
    mode: str,
    manual_values: tuple[float, ...] | None,
) -> tuple[Any, ...]:
    manual = None
    if manual_values is not None:
        manual = tuple(round(float(value), 4) for value in manual_values)
    return (
        round(float(lat), 4),
        round(float(lon), 4),
        when.isoformat(),
        mode,
        manual,
    )


def compute_prediction(
    bundle: core.ModelBundle,
    lat: float,
    lon: float,
    when: date,
    mode: str,
    manual_values: tuple[float, ...] | None,
) -> dict[str, Any]:
    area = core.check_training_area(lat, lon, when)
    warnings = list(area["warnings"])

    history: pd.DataFrame | None = None
    if mode == LIVE_MODE:
        history = fetch_power_cached(lat, lon, when.isoformat(), core.HISTORY_DAYS)
        features = core.build_live_features(
            history, lat, lon, when, card=bundle.card
        )
        source = "NASA POWER live history · 120-day window"
    else:
        if manual_values is None:
            raise core.FeatureBuildError("Manual weather values are missing.")
        t2m, t2m_max, rh2m, ws2m, precip, gwetroot, dry_days = manual_values
        if t2m_max < t2m:
            raise core.FeatureBuildError(
                "Daily maximum temperature must be greater than or equal to "
                "the daily mean temperature."
            )
        features = core.build_manual_features(
            lat,
            lon,
            when,
            t2m=t2m,
            t2m_max=t2m_max,
            rh2m=rh2m,
            ws2m=ws2m,
            precip=precip,
            gwetroot=gwetroot,
            dry_days=dry_days,
            card=bundle.card,
        )
        source = "Manual weather · persistence approximation"

    probability, source_status = core.score_features(bundle, features)
    label = core.risk_level(probability, bundle.card)
    return {
        "signature": make_signature(lat, lon, when, mode, manual_values),
        "probability": probability,
        "label": label,
        "index": core.risk_index(probability, bundle.card),
        "features": features,
        "source": source,
        "source_status": source_status,
        "warnings": warnings,
        "history": history,
        "lat_cell": area["lat_cell"],
        "lon_cell": area["lon_cell"],
        "when": when,
    }


def render_feature_table(features: pd.DataFrame) -> None:
    table = features.T.reset_index()
    table.columns = ["feature", "value"]
    table["group"] = table["feature"].map(core.feature_group)
    table["description"] = (
        table["feature"].map(core.FEATURE_DESCRIPTIONS).fillna("Model feature.")
    )
    table["value"] = table["value"].map(lambda value: f"{float(value):,.3f}")
    st.dataframe(
        table[["feature", "group", "value", "description"]],
        hide_index=True,
        width="stretch",
    )


def render_sensitivity(
    bundle: core.ModelBundle,
    base_row: pd.DataFrame,
    key_prefix: str,
) -> None:
    st.markdown("#### Sensitivity")
    st.caption(
        "One driver is varied while the rest of the feature vector is held "
        "fixed. This is an illustration of model behaviour, not a causal "
        "effect estimate."
    )
    if bundle.status == "ERROR":
        st.info("Sensitivity needs a valid model artifact.")
        return

    spec_key = st.selectbox(
        "Variable to vary",
        list(core.SENSITIVITY_SPECS.keys()),
        format_func=lambda key: core.SENSITIVITY_SPECS[key]["label"],
        key=f"{key_prefix}_sens_var",
    )
    spec = core.SENSITIVITY_SPECS[spec_key]
    try:
        table = core.sensitivity_table(bundle, base_row, spec_key, spec["values"])
    except Exception as exc:  # noqa: BLE001 - keep the page usable
        st.error(f"Sensitivity unavailable: {exc}")
        return

    st.line_chart(
        table.set_index("value")["probability"],
        height=230,
        x_label=spec["label"],
        y_label="Probability",
    )
    display = table.rename(
        columns={
            "value": spec["label"],
            "probability": "Probability",
            "risk_display": "Risk band",
        }
    )
    display["Probability"] = display["Probability"].map(core.format_probability)
    st.dataframe(
        display[[spec["label"], "Probability", "Risk band"]],
        hide_index=True,
        width="stretch",
        height=190,
    )
    if bundle.status == "DEMO":
        st.caption("DEMO heuristic sweep — not a trained-model response curve.")


# ---------------------------------------------------------------------------
# Single-page predictor
# ---------------------------------------------------------------------------

def render_predictor(
    bundle: core.ModelBundle,
    cell_stats: pd.DataFrame | None,
    mode: str,
) -> None:
    hero(
        "NASA FIRMS × NASA POWER",
        "Fire risk predictor",
        "Choose a location and date, then score the snapped NASA POWER grid "
        "cell with live weather history or a manual scenario. The result is "
        "the model's estimated probability of a fire-day in that cell.",
    )
    status_banner(bundle)

    live = mode == LIVE_MODE
    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.markdown("### Location & weather")
        with st.container(border=True):
            preset = st.selectbox(
                "Location preset",
                list(PRESETS.keys()),
                index=list(PRESETS.keys()).index(DEFAULT_PRESET),
                key="pred_preset",
            )
            if preset == "Custom":
                c1, c2 = st.columns(2)
                with c1:
                    lat = st.number_input(
                        "Latitude",
                        min_value=-90.0,
                        max_value=90.0,
                        value=36.7538,
                        step=0.0001,
                        format="%.4f",
                        key="pred_lat",
                    )
                with c2:
                    lon = st.number_input(
                        "Longitude",
                        min_value=-180.0,
                        max_value=180.0,
                        value=3.0588,
                        step=0.0001,
                        format="%.4f",
                        key="pred_lon",
                    )
            else:
                lat, lon = PRESETS[preset]  # type: ignore[misc]
                st.caption(f"Using {preset}: {lat:.4f}, {lon:.4f}")

            today = date.today()
            default_date = min(DEFAULT_DATE, today)
            when = st.date_input(
                "Date",
                value=default_date,
                min_value=date(2018, 1, 1),
                max_value=today,
                key="pred_date",
                help=(
                    "Live mode fetches 120 days of history ending on this date. "
                    "The model was trained on 2018–2024."
                ),
            )

            manual_values: tuple[float, ...] | None = None
            if live:
                st.info(
                    "Live mode fetches daily NASA POWER weather for the snapped "
                    "grid cell and computes the exact 7/14/30-day features with "
                    "the pipeline's own feature builder."
                )
            else:
                st.caption(
                    "Manual mode treats the entered weather as representative "
                    "of the antecedent window. It is an approximation; live "
                    "history is more faithful."
                )
                c1, c2 = st.columns(2)
                with c1:
                    t2m = st.slider(
                        "Mean temperature (°C)",
                        min_value=-5.0,
                        max_value=50.0,
                        value=32.0,
                        step=0.5,
                        key="pred_t2m",
                    )
                    t2m_max = st.slider(
                        "Daily max temperature (°C)",
                        min_value=0.0,
                        max_value=55.0,
                        value=40.0,
                        step=0.5,
                        key="pred_t2m_max",
                    )
                    rh2m = st.slider(
                        "Relative humidity (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=1.0,
                        key="pred_rh2m",
                    )
                    ws2m = st.slider(
                        "Wind speed (m/s)",
                        min_value=0.0,
                        max_value=25.0,
                        value=5.0,
                        step=0.5,
                        key="pred_ws2m",
                    )
                with c2:
                    precip = st.slider(
                        "Precipitation (mm/day)",
                        min_value=0.0,
                        max_value=50.0,
                        value=0.0,
                        step=0.5,
                        key="pred_precip",
                    )
                    gwetroot = st.slider(
                        "Root-zone soil wetness (0–1)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.15,
                        step=0.01,
                        key="pred_gwetroot",
                    )
                    dry_days = st.slider(
                        "Days since rain > 1 mm",
                        min_value=0,
                        max_value=120,
                        value=10,
                        step=1,
                        key="pred_dry_days",
                        help=(
                            "The panel median is about 10 days. This is the one "
                            "antecedent feature that cannot be inferred from a "
                            "single day of weather."
                        ),
                    )
                manual_values = (
                    float(t2m),
                    float(t2m_max),
                    float(rh2m),
                    float(ws2m),
                    float(precip),
                    float(gwetroot),
                    float(dry_days),
                )

            area = core.check_training_area(lat, lon, when)
            st.caption(
                f"Snapped POWER cell: **{area['lat_cell']:.3f}, "
                f"{area['lon_cell']:.3f}**"
            )
            for warning in area["warnings"]:
                st.warning(warning)

            run = False
            if live:
                run = st.button(
                    "Fetch & predict",
                    type="primary",
                    width="stretch",
                    key="pred_run",
                )
            else:
                st.caption(
                    "Manual predictions update automatically as you adjust the "
                    "sliders."
                )

        signature = make_signature(lat, lon, when, mode, manual_values)

        # Manual mode scores on every rerun (no network, instant feedback).
        # Live mode only scores on an explicit fetch so the page never blocks
        # on NASA POWER during unrelated widget changes.
        result: dict[str, Any] | None = None
        if live:
            stored = st.session_state.get("last_prediction")
            if run:
                if bundle.status == "ERROR":
                    result = {
                        "signature": signature,
                        "error": bundle.error or "Model is not available.",
                    }
                else:
                    with st.spinner("Fetching NASA POWER history and scoring…"):
                        try:
                            result = compute_prediction(
                                bundle, lat, lon, when, mode, manual_values
                            )
                        except Exception as exc:  # noqa: BLE001
                            result = {"signature": signature, "error": str(exc)}
                st.session_state["last_prediction"] = result
            elif stored and stored.get("signature") == signature:
                result = stored
        else:
            if bundle.status == "ERROR":
                result = {
                    "signature": signature,
                    "error": bundle.error or "Model is not available.",
                }
            else:
                try:
                    result = compute_prediction(
                        bundle, lat, lon, when, mode, manual_values
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {"signature": signature, "error": str(exc)}
            st.session_state["last_prediction"] = result

        st.caption(
            "The score is for the snapped ~50 km POWER cell, not the exact "
            "address."
        )

    with right:
        st.markdown("### Risk assessment")
        if result is None:
            st.markdown(
                """
                <div class="panel-card">
                  <h4 style="margin-top:0">Live weather not fetched yet</h4>
                  <p class="muted small" style="line-height:1.6">
                  Choose a location and date on the left, then press
                  <strong>Fetch &amp; predict</strong> to pull 120 days of NASA
                  POWER history and score the cell. The map below already
                  centres on the snapped grid cell.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_map(
                build_map(
                    area["lat_cell"],
                    area["lon_cell"],
                    selected=(area["lat_cell"], area["lon_cell"]),
                    cell_stats=cell_stats,
                    zoom=7,
                ),
                height=430,
            )
        elif result.get("error"):
            st.error(result["error"])
            st.caption("Adjust the inputs and try again.")
            render_map(
                build_map(
                    area["lat_cell"],
                    area["lon_cell"],
                    selected=(area["lat_cell"], area["lon_cell"]),
                    cell_stats=cell_stats,
                    zoom=7,
                ),
                height=430,
            )
        else:
            risk_card(
                result["probability"],
                result["label"],
                result["index"],
                result["source_status"],
                result["source"],
            )
            for warning in result["warnings"]:
                st.warning(warning)

            st.caption(
                f"Grid cell {result['lat_cell']:.3f}, {result['lon_cell']:.3f} · "
                f"{result['when']:%d %b %Y} · {result['source']}"
            )
            render_map(
                build_map(
                    result["lat_cell"],
                    result["lon_cell"],
                    selected=(result["lat_cell"], result["lon_cell"]),
                    risk_label=result["label"],
                    cell_stats=cell_stats,
                    zoom=7,
                ),
                height=430,
            )

            with st.expander("Feature vector used by the model", expanded=False):
                render_feature_table(result["features"])

            if result.get("history") is not None:
                with st.expander("NASA POWER history used", expanded=False):
                    history = result["history"].copy()
                    history = history.set_index("date")[
                        ["T2M_MAX", "RH2M", "PRECTOTCORR"]
                    ]
                    st.line_chart(
                        history,
                        height=240,
                        x_label="Date",
                        y_label="Daily value",
                    )
                    st.caption(
                        "The rolling 7/14/30-day features are computed from "
                        "this 120-day window."
                    )

            render_sensitivity(bundle, result["features"], key_prefix="pred")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model_mtime = (
        core.MODEL_PATH.stat().st_mtime if core.MODEL_PATH.exists() else 0.0
    )
    card_mtime = (
        core.MODEL_CARD_PATH.stat().st_mtime
        if core.MODEL_CARD_PATH.exists()
        else 0.0
    )
    bundle = load_bundle(model_mtime, card_mtime)

    panel_path = core.FIRE_PANEL_PATH
    panel_mtime = panel_path.stat().st_mtime if panel_path.exists() else 0.0
    panel, _panel_error = load_panel_cached(str(panel_path), panel_mtime)
    cell_stats = (
        cell_stats_cached(str(panel_path), panel_mtime) if panel is not None else None
    )
    summary = (
        core.panel_summary(panel)
        if panel is not None
        else core.panel_summary(pd.DataFrame())
    )

    mode = render_sidebar(bundle, summary)
    render_predictor(bundle, cell_stats, mode)

    st.markdown(
        '<div class="footer">FireWatch · NASA FIRMS + NASA POWER · '
        "For research and demonstration, not operational fire management."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
