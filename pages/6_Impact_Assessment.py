"""
FIREWATCH AI — Wildfire Impact Assessment (Module 5).

Separates four evidence layers that must never be conflated:

* predicted fire risk (date-specific model probability);
* potential exposure (present-day land-cover, population and infrastructure);
* observed fire detections (NASA FIRMS thermal anomalies);
* confirmed burned area (not available until phase 2).

All per-cell risk comes from the existing calibrated model via
``dashboard.model_loader.score_panel_date``.  Exposure artifacts are read from
``data/impact/``; when they are absent the page shows an explicit "Not
available" state instead of inventing KPI values.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Wildfire Impact Assessment | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
)

from dashboard.components import (  # noqa: E402
    inject_global_css,
    render_callout,
    render_header,
    render_kpi_card,
    render_section_header,
    render_sidebar,
)
from dashboard.config import COLORS  # noqa: E402
from dashboard.data_loader import load_fire_panel  # noqa: E402
from dashboard.model_loader import load_model, score_panel_date  # noqa: E402

from impact import config as impact_config  # noqa: E402
from impact import methodology  # noqa: E402
from impact.charts import (  # noqa: E402
    landcover_composition,
    regional_comparison,
    risk_exposure_scatter,
)
from impact.data import (  # noqa: E402
    load_admin_units,
    load_burned_area,
    load_cell_exposure,
    load_infrastructure,
    load_manifest,
    load_panel_date_range,
    load_ring_exposure,
    load_settlements,
)
from impact.exposure import merge_risk_and_exposure, prepare_exposure  # noqa: E402
from impact.maps import exposure_choropleth, risk_exposure_map  # noqa: E402
from impact.report import BURNED_AREA_NOT_AVAILABLE, build_reports  # noqa: E402
from impact.severity import (  # noqa: E402
    NOT_APPLICABLE,
    TIER_LABELS,
    assess_impact,
    formula_text,
    priority_matrix_rows,
)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _fmt(value: object, digits: int = 0, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not np.isfinite(number):
        return "Not available"
    return f"{number:,.{digits}f}{suffix}"


def _fmt_percent(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not np.isfinite(number):
        return "Not available"
    return f"{number * 100:.{digits}f}%"


def _impact_dir() -> Path:
    """Return the impact artifact directory, with an AppTest override.

    The override is a session-state test seam only.  It never writes files and
    defaults to the real ``data/impact`` directory.
    """
    override = st.session_state.get("impact_data_dir")
    if override:
        return Path(str(override))
    return Path(impact_config.IMPACT_DATA_DIR)


def _render_folium(m: object, height: int = 520) -> None:
    try:
        components.html(m._repr_html_(), height=height, scrolling=True)
    except Exception as exc:  # noqa: BLE001 - a map must not break the page
        st.info(f"Map could not be rendered: {exc}")


def _unavailable(message: str, dataset: str = "Impact exposure layers") -> None:
    # The shared render_data_unavailable card hard-codes the fire-panel build
    # command, which is wrong for this module.  Use a callout with the impact
    # artifact command instead of changing the shared component.
    render_callout(
        title=f"Dataset Not Found: {dataset}",
        message=(
            f"Not available. {message} "
            "Generate the impact artifacts with "
            "<code>python scripts/build_impact_layers.py</code>."
        ),
        variant="warning",
    )


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
inject_global_css()
render_sidebar()
render_header(
    title="Wildfire Impact Assessment",
    subtitle=(
        "Predicted risk, potential exposure and observed detections kept "
        "explicitly separate"
    ),
    badge="Impact Assessment",
)

base_dir = _impact_dir()

manifest = load_manifest(base_dir / "manifest.json")
exposure_data = load_cell_exposure(base_dir / "cell_exposure.parquet")
ring_data = load_ring_exposure(base_dir / "cell_ring_exposure.parquet")
settlements_data = load_settlements(base_dir / "cell_settlements.parquet")
infrastructure_data = load_infrastructure(base_dir / "cell_infrastructure.parquet")
admin_data = load_admin_units(base_dir / "admin_units.parquet")
burned_area_data = load_burned_area(base_dir / "burned_area_monthly.parquet")

date_range = load_panel_date_range()
panel = load_fire_panel()
model = load_model()
model_ready = (
    model.status == "REAL"
    and panel is not None
    and not panel.empty
    and date_range.available
)

data_ready = exposure_data.available and manifest.available

# ---------------------------------------------------------------------------
# Assessment date
# ---------------------------------------------------------------------------
selected_date: pd.Timestamp | None = None
if date_range.available and date_range.dates:
    available_dates = list(date_range.dates)
    default_date = available_dates[len(available_dates) // 2]
    col_date, col_note = st.columns([1.1, 2.9])
    with col_date:
        picked = st.date_input(
            "Assessment date",
            value=default_date.date(),
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date(),
            help=(
                "Risk is date-specific and uses the fire panel's own range. "
                "Exposure is a present-day static snapshot."
            ),
        )
    selected_date = pd.Timestamp(picked)
    with col_note:
        st.markdown(
            f"""
            <div style="font-size:0.82rem; color:{COLORS['text_secondary']};
                        padding-top:26px; line-height:1.5;">
                Risk window read from <code>fire_panel.parquet</code>:
                <b>{available_dates[0].date()}</b> to
                <b>{available_dates[-1].date()}</b>.
                Confirmed burned area: <b>{BURNED_AREA_NOT_AVAILABLE}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    render_callout(
        title="Risk window unavailable",
        message=(
            date_range.message
            if hasattr(date_range, "message") and date_range.message
            else "No fire-panel date range could be read."
        ),
        variant="warning",
    )

if model.status != "REAL":
    render_callout(
        title="Predicted risk unavailable",
        message=model.error or "The trained model is not available.",
        variant="danger",
    )

# ---------------------------------------------------------------------------
# Prepare exposure and date-specific risk
# ---------------------------------------------------------------------------
prepared = prepare_exposure(exposure_data.frame) if exposure_data.available else None
if prepared is not None and not prepared.available:
    prepared = None

scores = None
if model_ready and selected_date is not None:
    try:
        scores = score_panel_date(model, panel, selected_date)
    except Exception as exc:  # noqa: BLE001 - surfaced, never replaced by zeros
        st.warning(f"Predicted risk could not be scored: {exc}")
        scores = None

assessed: pd.DataFrame | None = None
severity_message = "Not available."
if prepared is not None and scores is not None and not scores.empty:
    try:
        merged = merge_risk_and_exposure(prepared.frame, scores)
        if "burned" in scores.columns:
            burned = scores[["lat_cell", "lon_cell", "burned"]].drop_duplicates(
                subset=["lat_cell", "lon_cell"]
            )
            merged = merged.merge(
                burned,
                on=["lat_cell", "lon_cell"],
                how="left",
                validate="many_to_one",
            )
        if admin_data.available and "admin_name" in admin_data.frame.columns:
            admin = (
                admin_data.frame[["lat_cell", "lon_cell", "admin_name"]]
                .drop_duplicates(subset=["lat_cell", "lon_cell"])
            )
            merged = merged.merge(
                admin, on=["lat_cell", "lon_cell"], how="left", validate="many_to_one"
            )
        severity = assess_impact(
            merged,
            manifest=manifest.manifest if manifest.available else None,
        )
        severity_message = severity.message
        if severity.available:
            assessed = severity.frame
    except Exception as exc:  # noqa: BLE001 - surfaced, never silently zero-filled
        severity_message = f"Impact severity could not be computed: {exc}"

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Overview",
        "Environment",
        "Population",
        "Infrastructure",
        "Priority Matrix",
        "Report",
    ]
)

# ── Overview ──────────────────────────────────────────────────────────────
with tabs[0]:
    if not data_ready:
        _unavailable(
            "The exposure artifacts and provenance manifest in data/impact/ "
            "were not found. Run scripts/build_impact_layers.py to generate "
            "them; no KPI values are fabricated."
        )
        # Name the four layers explicitly so the absent state is still clear
        # about what the module reports and what it does not.
        for key in methodology.LAYER_ORDER:
            layer = methodology.EVIDENCE_LAYERS[key]
            st.markdown(f"**{layer['label']}** — Not available.")
    elif assessed is None or assessed.empty:
        _unavailable(severity_message)
    else:
        probabilities = pd.to_numeric(assessed["probability"], errors="coerce")
        populations = pd.to_numeric(assessed["pop_cell"], errors="coerce")
        mean_probability = float(probabilities.mean())
        total_population = float(populations.sum())
        burned_cells = int(
            (pd.to_numeric(assessed["burned"], errors="coerce") == 1).sum()
        )
        mean_iei = float(pd.to_numeric(assessed["iei"], errors="coerce").mean())

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                title="Predicted fire risk",
                value=_fmt_percent(mean_probability, digits=2),
                subtitle=(
                    "Mean model probability of a thermal detection within "
                    "3 days, selected date"
                ),
                status="danger",
            )
        with k2:
            render_kpi_card(
                title="Potential exposure",
                value=f"{_fmt(total_population)} people",
                subtitle=(
                    "Population inside assessed cells; present-day static "
                    "snapshot, not affected people"
                ),
                status="moderate",
            )
        with k3:
            render_kpi_card(
                title="Observed fire detections",
                value=f"{burned_cells:,} cells",
                subtitle=(
                    "Same-day NASA FIRMS detections; not confirmed burned area"
                ),
                status="neutral",
            )
        with k4:
            render_kpi_card(
                title="Confirmed burned area",
                value="Not available",
                subtitle="Phase 2 — independent burned-area product not ingested",
                status="neutral",
            )

        # Plain-text summary lines also act as stable page-test anchors.
        st.markdown(f"Predicted risk — mean probability: {mean_probability:.4f}")
        st.markdown(
            f"Potential exposure — population in cells: {total_population:.0f}"
        )
        st.markdown(f"Observed detections — cells with detection: {burned_cells}")
        st.markdown(
            f"Confirmed burned area — {BURNED_AREA_NOT_AVAILABLE}"
        )
        st.markdown(
            f"Potential exposure — mean Impact Exposure Index: {mean_iei:.4f}"
        )

        fuel_missing_in_assessed = bool(
            assessed.get("iei_fuel_missing", pd.Series([False])).iloc[0]
            if "iei_fuel_missing" in assessed.columns
            else ("fuel_area_km2" not in assessed.columns or pd.to_numeric(assessed["fuel_area_km2"], errors="coerce").isna().all())
        )
        if fuel_missing_in_assessed:
            render_callout(
                title="Severity computed without fuel component",
                message=(
                    "Severity was computed without the fuel component (WorldCover was not "
                    "ingested; weights renormalized: population 0.35/0.65, infrastructure 0.30/0.65)."
                ),
                variant="info",
            )
            st.markdown(
                "Severity was computed without the fuel component (pop+infra only; weights renormalized)."
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        _render_folium(
            exposure_choropleth(
                assessed,
                metric="iei",
                metric_label="Impact Exposure Index",
            )
        )
        st.caption(
            "Exposure is a present-day static snapshot. Darker cells hold more "
            "mapped people and assets; colour does not mean a fire occurred."
        )

        render_section_header(
            "Predicted risk vs. potential exposure",
            "Cell colour is the model risk band; fill opacity rises with the "
            "static exposure tier",
        )
        _render_folium(risk_exposure_map(assessed, card=model.card))

# ── Environment ───────────────────────────────────────────────────────────
with tabs[1]:
    env_unavailable = (
        assessed is not None
        and (
            "fuel_area_km2" not in assessed.columns
            or pd.to_numeric(assessed["fuel_area_km2"], errors="coerce").isna().all()
        )
    )
    if not data_ready or assessed is None:
        _unavailable(
            "Environmental exposure is unavailable because the land-cover "
            "artifact or manifest is missing."
        )
    elif env_unavailable:
        render_callout(
            title="Environmental Exposure Not Available",
            message=(
                "Not available: WorldCover not ingested. Land-cover and "
                "environmental exposure data was not downloaded. Severity was "
                "computed without the fuel component."
            ),
            variant="info",
        )
        st.markdown(
            "**Environmental exposure** — Not available (WorldCover not ingested)."
        )
    else:
        render_section_header(
            "Land-cover composition",
            "Potential exposure by ESA WorldCover class; not burned area",
        )
        st.plotly_chart(landcover_composition(assessed), use_container_width=True)

        landcover_columns = [
            column
            for column in impact_config.LC_AREA_COLUMNS.values()
            if column in assessed.columns
        ]
        display_columns = [
            column
            for column in ("lat_cell", "lon_cell", "fuel_area_km2", "fuel_fraction")
            + tuple(landcover_columns)
            if column in assessed.columns
        ]
        st.dataframe(
            assessed[display_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Flammable fuel is the sum of tree, shrub, grass and crop areas. "
            "Land-cover coverage is checked against cell area at build time."
        )

# ── Population ────────────────────────────────────────────────────────────
with tabs[2]:
    if not data_ready or assessed is None:
        _unavailable(
            "Population exposure is unavailable because the exposure artifact "
            "or manifest is missing."
        )
    else:
        populations = pd.to_numeric(assessed["pop_cell"], errors="coerce")
        densities = pd.to_numeric(
            assessed.get("pop_density_per_km2", pd.Series(dtype=float)),
            errors="coerce",
        )
        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi_card(
                title="Potential exposure — population in cells",
                value=f"{_fmt(populations.sum())} people",
                subtitle="WorldPop modelled estimate inside assessed cells",
                status="moderate",
            )
        with k2:
            render_kpi_card(
                title="Mean population density",
                value=_fmt(densities.mean(), digits=2, suffix=" /km²"),
                subtitle="Potential exposure, not affected population",
                status="neutral",
            )
        with k3:
            render_kpi_card(
                title="Settlements mapped",
                value=(
                    f"{len(settlements_data.frame):,}"
                    if settlements_data.available
                    else "Not available"
                ),
                subtitle="OSM settlement points with WorldPop estimates",
                status="neutral",
            )

        ring_columns = [
            column
            for column in ("lat_cell", "lon_cell", "ring_km", "population")
            if ring_data.available and column in ring_data.frame.columns
        ]
        if ring_columns:
            render_section_header(
                "Distance-ring potential exposure",
                "5, 10 and 25 km rings; rings overlap between cells and are not additive",
            )
            st.dataframe(
                ring_data.frame[ring_columns],
                use_container_width=True,
                hide_index=True,
            )
        elif ring_data.available:
            st.info("Ring exposure is present but lacks the expected columns.")
        else:
            st.markdown("**Not available.** Distance-ring exposure was not found.")

        if settlements_data.available:
            st.dataframe(
                settlements_data.frame,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown("**Not available.** The settlement table was not found.")

# ── Infrastructure ────────────────────────────────────────────────────────
with tabs[3]:
    if not data_ready or assessed is None:
        _unavailable(
            "Infrastructure exposure is unavailable because the exposure "
            "artifact or manifest is missing."
        )
    else:
        roads = pd.to_numeric(assessed["road_km_total"], errors="coerce")
        buildings = pd.to_numeric(assessed["building_count"], errors="coerce")
        critical = pd.to_numeric(assessed["critical_count_total"], errors="coerce")
        k1, k2, k3 = st.columns(3)
        with k1:
            render_kpi_card(
                title="Potential exposure — mapped roads",
                value=_fmt(roads.sum(), digits=1, suffix=" km"),
                subtitle="OSM road length inside assessed cells; lower bound",
                status="neutral",
            )
        with k2:
            render_kpi_card(
                title="Potential exposure — mapped buildings",
                value=f"{_fmt(buildings.sum())}",
                subtitle="OSM building footprints; present-day snapshot",
                status="neutral",
            )
        with k3:
            render_kpi_card(
                title="Potential exposure — critical facilities",
                value=f"{_fmt(critical.sum())}",
                subtitle="Explicitly tagged facilities only; OSM is sparse",
                status="moderate",
            )

        display_columns = [
            column
            for column in (
                "lat_cell",
                "lon_cell",
                "road_km_total",
                "building_count",
                "building_footprint_km2",
                "critical_count_total",
            )
            if column in assessed.columns
        ]
        st.dataframe(
            assessed[display_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Mapped assets are lower bounds. No building, road or facility is "
            "inferred where an explicit source feature is absent."
        )

        if infrastructure_data.available:
            st.dataframe(
                infrastructure_data.frame,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown(
                "**Not available.** The critical-infrastructure asset table was "
                "not found."
            )

# ── Priority Matrix ───────────────────────────────────────────────────────
with tabs[4]:
    render_section_header(
        "Risk × exposure priority matrix",
        "A transparent prioritisation heuristic; not a damage probability",
    )
    st.markdown(
        f"<div style='font-size:0.84rem; color:{COLORS['text_secondary']}; "
        f"line-height:1.6;'>{formula_text()}</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(priority_matrix_rows()),
        use_container_width=True,
        hide_index=True,
    )

    if assessed is None or assessed.empty:
        _unavailable(severity_message)
    else:
        distribution = (
            assessed["impact_priority_label"]
            .value_counts(dropna=False)
            .rename_axis("Priority label")
            .reset_index(name="Cells")
        )
        st.dataframe(distribution, use_container_width=True, hide_index=True)
        st.caption(
            "Priority 1–5 combines the predicted risk band with the static "
            "exposure tier. Cells outside the model area are labelled "
            f"'{NOT_APPLICABLE}'. Exposure tiers: {', '.join(TIER_LABELS)}."
        )

        render_section_header(
            "Risk vs. exposure by cell",
            "Predicted risk (3-day detection likelihood) against potential exposure",
        )
        st.plotly_chart(
            risk_exposure_scatter(assessed, card=model.card),
            use_container_width=True,
        )
        st.plotly_chart(
            regional_comparison(assessed),
            use_container_width=True,
        )
        st.caption(
            "Regional comparison uses administrative assignment when "
            "available, otherwise a latitude band."
        )

# ── Report ────────────────────────────────────────────────────────────────
with tabs[5]:
    report = build_reports(
        manifest.manifest if manifest.available else None,
        manifest_path=manifest.path,
        assessed=assessed,
        date=selected_date,
        risk_start=date_range.start if date_range.available else None,
        risk_end=date_range.end if date_range.available else None,
        burned_area_status=BURNED_AREA_NOT_AVAILABLE,
    )

    render_section_header(
        "Downloadable assessment report",
        "HTML, CSV and JSON; every report carries the four evidence layers, "
        "source licences and the manifest checksum",
    )
    if not report.available:
        st.markdown("**Not available.** Report generation requires a verified "
                    "provenance manifest.")
        st.info(report.message)
    else:
        st.markdown(
            "Manifest checksum (SHA-256): "
            f"`{report.manifest_checksum}`"
        )
        if assessed is not None and not assessed.empty:
            fuel_missing_in_report = bool(
                assessed.get("iei_fuel_missing", pd.Series([False])).iloc[0]
                if "iei_fuel_missing" in assessed.columns
                else ("fuel_area_km2" not in assessed.columns or pd.to_numeric(assessed["fuel_area_km2"], errors="coerce").isna().all())
            )
            if fuel_missing_in_report:
                st.info(
                    "Severity was computed without the fuel component (WorldCover was not ingested; "
                    "weights renormalized to population 0.35/0.65 and infrastructure 0.30/0.65)."
                )
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download HTML report",
                data=report.html or "",
                file_name="firewatch_impact_report.html",
                mime="text/html",
                use_container_width=True,
                key="impact_download_html",
            )
        with d2:
            st.download_button(
                "Download CSV report",
                data=report.csv or "",
                file_name="firewatch_impact_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="impact_download_csv",
            )
        with d3:
            st.download_button(
                "Download JSON report",
                data=report.json_text or "",
                file_name="firewatch_impact_report.json",
                mime="application/json",
                use_container_width=True,
                key="impact_download_json",
            )

        with st.expander("Evidence layers, sources and licences", expanded=False):
            for key in methodology.LAYER_ORDER:
                layer = methodology.EVIDENCE_LAYERS[key]
                st.markdown(f"**{layer['label']}** — {layer['definition']}")
                st.caption(f"Source: {layer['source']}")
            st.markdown("---")
            for source in methodology.SOURCES:
                st.markdown(
                    f"**{source['name']}** — {source['role']}. "
                    f"Licence: {source['license']}. {source['attribution']}."
                )

        st.markdown(f"**Scope disclaimer.** {methodology.DISCLAIMER}")
        st.caption(
            f"Confirmed burned area: {BURNED_AREA_NOT_AVAILABLE}. "
            f"Observed detections: {severity_message}"
        )
