"""
FIREWATCH AI — Geospatial Wildfire Monitoring (Module 2).

Interactive click-to-predict fire risk for a selected location, followed by
retrospective thermal-hotspot layers for northern Algeria.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Geospatial Monitoring | FIREWATCH AI",
    page_icon="🔥",
    layout="wide",
)

import app_core
from dashboard import geo
from dashboard.components import (
    inject_global_css,
    render_header,
    render_sidebar,
    render_kpi_card,
    render_section_header,
    render_callout,
    render_data_unavailable,
)
from dashboard.config import (
    COLORS,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    RISK_LEVELS,
)
from dashboard.data_loader import load_fires_daily, load_fire_panel, get_date_range
from dashboard.maps import (
    render_grid_detection_map,
    render_density_heatmap,
    render_cluster_map,
    render_selection_map,
    selection_marker_group,
    viewport_from_map_data,
)
from dashboard.model_loader import load_model


def _available_anchor_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    """Panel dates with a complete 30-day feature history."""
    dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
    if dates.empty:
        return []
    earliest = dates.min() + pd.Timedelta(days=30)
    return sorted(dates[dates >= earliest].dt.normalize().unique())


def _period_badge(when) -> None:
    period_note = app_core.data_period_note(when)
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


def _selection_summary(point, availability) -> None:
    """Render the selected-point / nearest-cell summary card."""
    if point is None:
        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['surface_white']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 10px;
                padding: 18px 20px;
            ">
                <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.06em;
                            text-transform:uppercase; color:{COLORS['text_muted']};">
                    Selected Location
                </div>
                <div style="font-size:1.0rem; font-weight:600; margin-top:6px;
                            color:{COLORS['text_primary']};">
                    No location selected
                </div>
                <div style="font-size:0.82rem; margin-top:4px; color:{COLORS['text_secondary']};">
                    Click anywhere inside the highlighted Algerian study area.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status_color = (
        COLORS["brand_accent"] if point.in_area else COLORS["risk_high"]
    )
    status_text = (
        "Inside the study area" if point.in_area else "Outside the study area"
    )
    if availability is None:
        data_text, data_color = "Not checked", COLORS["text_muted"]
    elif availability.ok:
        data_text, data_color = "Complete · 20 / 20 features", COLORS["risk_low"]
    else:
        data_text, data_color = "Unavailable", COLORS["risk_high"]

    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['surface_white']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 10px;
            padding: 18px 20px;
        ">
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.06em;
                        text-transform:uppercase; color:{COLORS['text_muted']};">
                Selected Location
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
                <span style="width:8px; height:8px; border-radius:50%;
                             background-color:{status_color}; display:inline-block;"></span>
                <span style="font-size:0.86rem; font-weight:600; color:{status_color};">
                    {status_text}
                </span>
            </div>
            <table style="width:100%; border-collapse:collapse; margin-top:12px;
                          font-size:0.82rem;">
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:4px 0;">Selected point</td>
                    <td style="text-align:right; font-weight:600; color:{COLORS['text_primary']};">
                        {point.lat:.4f}°N, {point.lon:.4f}°E
                    </td>
                </tr>
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:4px 0;">Nearest model cell</td>
                    <td style="text-align:right; font-weight:600; color:{COLORS['text_primary']};">
                        {point.lat_cell:.4f}°N, {point.lon_cell:.4f}°E
                    </td>
                </tr>
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:4px 0;">Distance to cell</td>
                    <td style="text-align:right; font-weight:600; color:{COLORS['text_primary']};">
                        {point.distance_km:.1f} km
                    </td>
                </tr>
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:4px 0;">Feature data</td>
                    <td style="text-align:right; font-weight:600; color:{data_color};">
                        {data_text}
                    </td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_prediction_result(result: dict) -> None:
    """Render the prediction result card."""
    risk = str(result["risk"]).upper()
    display = risk.title()
    style = RISK_LEVELS.get(
        display,
        {"color": COLORS["brand_primary"], "bg": COLORS["brand_subtle"]},
    )
    horizon = int(result["horizon_days"])
    start = pd.Timestamp(result["date"])
    end = start + pd.Timedelta(days=horizon)
    probability = app_core.format_percent(float(result["probability"]))

    st.markdown(
        f"""
        <div style="
            background-color: {style['bg']};
            border: 1px solid {style['color']};
            border-left: 6px solid {style['color']};
            border-radius: 10px;
            padding: 18px 20px;
            margin-top: 14px;
        ">
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.06em;
                        text-transform:uppercase; color:{COLORS['text_muted']};">
                Risk Category
            </div>
            <div style="display:flex; align-items:baseline; gap:10px; margin-top:4px;">
                <span style="font-size:2rem; font-weight:700; color:{style['color']};">
                    {risk}
                </span>
                <span style="font-size:1.35rem; font-weight:600; color:{COLORS['text_primary']};">
                    {probability}
                </span>
            </div>
            <div style="font-size:0.82rem; color:{COLORS['text_secondary']}; margin-top:2px;">
                Probability of at least one fire in the cell within {horizon} days
            </div>
            <table style="width:100%; border-collapse:collapse; margin-top:14px;
                          font-size:0.8rem; border-top:1px solid {COLORS['border_light']};">
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:6px 0 2px 0;">Selected point</td>
                    <td style="text-align:right; font-weight:600; padding:6px 0 2px 0;">
                        {float(result['lat']):.4f}°N, {float(result['lon']):.4f}°E
                    </td>
                </tr>
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:2px 0;">Model grid cell</td>
                    <td style="text-align:right; font-weight:600; padding:2px 0;">
                        {float(result['lat_cell']):.4f}°N, {float(result['lon_cell']):.4f}°E
                    </td>
                </tr>
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:2px 0;">Cell distance</td>
                    <td style="text-align:right; font-weight:600; padding:2px 0;">
                        {float(result['distance_km']):.1f} km
                    </td>
                </tr>
                <tr>
                    <td style="color:{COLORS['text_muted']}; padding:2px 0;">Horizon</td>
                    <td style="text-align:right; font-weight:600; padding:2px 0;">
                        {start:%Y-%m-%d} → {end:%Y-%m-%d} ({horizon} days)
                    </td>
                </tr>
            </table>
            <div style="font-size:0.74rem; color:{COLORS['text_muted']}; margin-top:10px;">
                The model target is at least one fire in this grid cell within the
                {horizon}-day window, not a same-day detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()
render_sidebar()
render_header(
    title="Geospatial Fire Monitoring",
    subtitle=(
        "Click-to-predict fire risk and retrospective thermal hotspot mapping "
        "(32.0°–37.5°N, -2.0°–9.0°E)"
    ),
    badge="GIS Engine Active",
)

# Load fire records and the full panel (the prediction section needs all cells)
fires_df = load_fires_daily()
panel_df = load_fire_panel()
if fires_df is None or fires_df.empty:
    if panel_df is not None and not panel_df.empty:
        fires_df = panel_df[panel_df["burned"] == 1].copy()

if (fires_df is None or fires_df.empty) and (panel_df is None or panel_df.empty):
    render_data_unavailable("fires_daily.parquet or fire_panel.parquet")
    st.stop()

try:
    study_polygon = geo.study_area_polygon()
except Exception as exc:  # noqa: BLE001 - surface a missing/corrupt boundary file
    study_polygon = None
    render_callout(
        title="Study-Area Boundary Unavailable",
        message=(
            "The interactive prediction map needs the Algeria boundary file. "
            f"Details: {exc}"
        ),
        variant="danger",
    )

# ---------------------------------------------------------------------------
# Interactive click-to-predict section
# ---------------------------------------------------------------------------
map_base_tile = "Light"

render_section_header(
    title="Interactive Fire Risk Assessment",
    subtitle=(
        "Click inside the highlighted Algerian study area, then predict the "
        "3-day fire risk for that location."
    ),
)

if study_polygon is None:
    pass
elif panel_df is None or panel_df.empty:
    render_callout(
        title="Panel Required",
        message="The prediction map needs data/processed/fire_panel.parquet.",
        variant="warning",
    )
else:
    bundle = load_model()
    anchor_dates = _available_anchor_dates(panel_df)

    if not anchor_dates:
        render_callout(
            title="No Prediction Date Available",
            message="The panel has no date at least 30 days after its start.",
            variant="warning",
        )
    else:
        for state_key, state_default in (
            ("geo_selection", None),
            ("geo_result", None),
            ("geo_notice", None),
            ("geo_viewport", None),
            ("geo_map_signature", None),
        ):
            if state_key not in st.session_state:
                st.session_state[state_key] = state_default

        col_ctrl1, col_ctrl2 = st.columns([1.5, 1.0])
        with col_ctrl1:
            anchor = st.selectbox(
                "Prediction Anchor Date",
                options=anchor_dates,
                index=len(anchor_dates) - 1,
                format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
                help=(
                    "The model target is at least one fire in the selected grid "
                    "cell within the 3 days after this date. The default is the "
                    "latest panel date."
                ),
                key="geo_anchor_date",
            )
            _period_badge(anchor)
        with col_ctrl2:
            map_base_tile = st.selectbox(
                "Base Map Cartography",
                options=["Light", "Dark", "OpenStreetMap"],
                index=0,
                key="geo_basemap",
                help="Used by the prediction map and the observation layers below.",
            )

        # A result is only valid for the anchor date it was computed from.
        existing_result = st.session_state.get("geo_result")
        if existing_result is not None and pd.Timestamp(
            existing_result["date"]
        ).normalize() != pd.Timestamp(anchor).normalize():
            st.session_state["geo_result"] = None
            st.session_state["geo_notice"] = None

        names = (
            [str(name) for name in bundle.card["feature_names"]]
            if bundle.card and "feature_names" in bundle.card
            else []
        )

        selection = st.session_state.get("geo_selection")
        point = (
            geo.resolve_point(
                float(selection["lat"]),
                float(selection["lon"]),
                ring=study_polygon,
            )
            if selection
            else None
        )
        availability = (
            geo.feature_row_for(
                panel_df,
                anchor,
                point.lat_cell,
                point.lon_cell,
                names,
            )
            if point is not None and point.in_area and names
            else None
        )

        col_map, col_side = st.columns([1.65, 1.0], gap="large")

        with col_map:
            # The base-map script must stay identical between selections.
            # streamlit-folium hashes that script to identify the component, so
            # embedding the moving marker there would remount (and re-centre)
            # the whole map on every click.  The marker is therefore passed as
            # a dynamic feature group below.
            base_signature = (map_base_tile,)
            previous_signature = st.session_state.get("geo_map_signature")
            restore_viewport = (
                previous_signature is not None
                and previous_signature != base_signature
            )
            st.session_state["geo_map_signature"] = base_signature

            viewport = st.session_state.get("geo_viewport")
            map_center = map_zoom = None
            if restore_viewport and viewport:
                map_center = (
                    float(viewport["center"]["lat"]),
                    float(viewport["center"]["lng"]),
                )
                map_zoom = int(viewport["zoom"])

            selection_map = render_selection_map(
                tiles=map_base_tile,
                polygon=study_polygon,
            )
            marker_group = selection_marker_group(
                (point.lat, point.lon) if point is not None else None,
                in_area=bool(point is not None and point.in_area),
            )
            try:
                from streamlit_folium import st_folium

                map_data = st_folium(
                    selection_map,
                    width=None,
                    height=620,
                    use_container_width=True,
                    # These are the viewport fields returned by the installed
                    # streamlit-folium version (see its frontend update payload).
                    returned_objects=[
                        "last_clicked",
                        "center",
                        "zoom",
                        "bounds",
                    ],
                    feature_group_to_add=marker_group,
                    center=map_center,
                    zoom=map_zoom,
                    key="geo_select_map",
                )
            except Exception:  # noqa: BLE001 - static fallback without clicks
                fallback_map = render_selection_map(
                    selected=(
                        (point.lat, point.lon) if point is not None else None
                    ),
                    tiles=map_base_tile,
                    in_area=bool(point is not None and point.in_area),
                    polygon=study_polygon,
                )
                st.components.v1.html(fallback_map._repr_html_(), height=630)
                map_data = None
                render_callout(
                    title="Interactive Map Unavailable",
                    message=(
                        "The map is shown in static mode because streamlit-folium "
                        "could not initialise. Install streamlit-folium to enable "
                        "click selection."
                    ),
                    variant="warning",
                )

            # Persist the user's viewport before any rerun, so a later rebuild
            # (for example after changing the basemap) can restore it.
            live_viewport = viewport_from_map_data(map_data)
            if live_viewport is not None:
                st.session_state["geo_viewport"] = live_viewport

            clicked = (map_data or {}).get("last_clicked")
            if clicked:
                lat = float(clicked["lat"])
                lon = float(clicked["lng"])
                previous = st.session_state.get("geo_selection")
                if (
                    previous is None
                    or abs(float(previous["lat"]) - lat) > 1e-9
                    or abs(float(previous["lon"]) - lon) > 1e-9
                ):
                    st.session_state["geo_selection"] = {"lat": lat, "lon": lon}
                    st.session_state["geo_result"] = None
                    st.session_state["geo_notice"] = None
                    # One rerun is needed to send the new marker to the map.
                    # Because the marker travels in a feature group, the map is
                    # not remounted and keeps its current centre and zoom.
                    st.rerun()

        with col_side:
            _selection_summary(point, availability)

            if bundle.status != "REAL":
                message = (
                    f"The model loader returned {bundle.status}. Predictions are "
                    "disabled until a REAL model/card pair is available."
                )
                if bundle.error:
                    message += f" Details: {bundle.error}"
                render_callout(
                    title="Prediction Unavailable",
                    message=message,
                    variant="danger" if bundle.status == "ERROR" else "warning",
                )

            if point is not None and not point.in_area:
                render_callout(
                    title="Outside the Study Area",
                    message=point.reason or "Choose a point inside Algeria.",
                    variant="warning",
                )
            elif availability is not None and not availability.ok:
                render_callout(
                    title="Feature Data Unavailable",
                    message=availability.reason or "Feature data is incomplete.",
                    variant="warning",
                )

            predict_disabled = bool(
                bundle.status != "REAL"
                or not names
                or point is None
                or not point.in_area
                or availability is None
                or not availability.ok
            )
            predict_clicked = st.button(
                "Predict Fire Risk",
                type="primary",
                use_container_width=True,
                disabled=predict_disabled,
                key="geo_predict_button",
            )

            if predict_clicked and point is not None:
                try:
                    with st.spinner("Scoring the selected grid cell …"):
                        prediction = geo.predict_point(
                            bundle,
                            panel_df,
                            point.lat,
                            point.lon,
                            anchor,
                            feature_names=names,
                            ring=study_polygon,
                        )
                    st.session_state["geo_result"] = {
                        "probability": prediction.probability,
                        "risk": prediction.risk,
                        "lat": prediction.lat,
                        "lon": prediction.lon,
                        "lat_cell": prediction.lat_cell,
                        "lon_cell": prediction.lon_cell,
                        "distance_km": prediction.distance_km,
                        "date": prediction.date.isoformat(),
                        "horizon_days": prediction.horizon_days,
                    }
                    st.session_state["geo_notice"] = None
                except (
                    geo.OutsideStudyAreaError,
                    geo.FeatureDataUnavailable,
                    app_core.ModelNotReadyError,
                    app_core.FeatureContractError,
                ) as exc:
                    st.session_state["geo_result"] = None
                    st.session_state["geo_notice"] = str(exc)

            notice = st.session_state.get("geo_notice")
            if notice:
                render_callout(
                    title="Prediction Not Available",
                    message=notice,
                    variant="warning",
                )

            result = st.session_state.get("geo_result")
            if result is not None:
                _render_prediction_result(result)

        with st.expander("How this prediction is produced", expanded=False):
            st.markdown(
                f"""
                <div style="font-size:0.84rem; color:{COLORS['text_secondary']}; line-height:1.65;">
                    <p style="margin:0 0 8px 0;">
                        A click only supplies latitude and longitude. The app snaps
                        that point to the nearest <b>0.5° × 0.625° model grid cell</b>
                        and uses that cell's existing feature row from
                        <code>fire_panel.parquet</code>. The cell coordinates and the
                        distance from your click are shown with every result.
                    </p>
                    <p style="margin:0 0 8px 0;">
                        The row supplies the model's exact 20-feature contract. No
                        meteorological values are invented or imputed. If any feature
                        is missing for the selected cell/date, prediction is blocked
                        and the missing feature is named.
                    </p>
                    <p style="margin:0;">
                        The shipped model predicts <b>at least one fire in the cell
                        within the next 3 days</b>. The horizon shown is the three days
                        after the selected panel anchor date.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Retrospective observation layers (unchanged functionality)
# ---------------------------------------------------------------------------
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
render_section_header(
    title="Observed Fire Detections",
    subtitle=(
        "Retrospective satellite detections for the same northern Algeria study "
        "area. These layers are not model predictions."
    ),
)

if fires_df is None or fires_df.empty:
    render_callout(
        title="No Fire Records Found",
        message="This section needs fires_daily.parquet or labeled panel rows.",
        variant="warning",
    )
else:
    min_d, max_d = get_date_range(fires_df)
    col_f1, col_f2 = st.columns([1.5, 1.2])
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
            options=[
                "Grid Frequency Markers",
                "Density HeatMap Surface",
                "Clustered Incidents",
            ],
            index=0,
            help="Select spatial visualization technique",
        )

    start_date, end_date = None, None
    if isinstance(date_filter, (tuple, list)):
        if len(date_filter) == 2:
            start_date, end_date = date_filter
        elif len(date_filter) == 1:
            start_date = end_date = date_filter[0]
    else:
        start_date = end_date = date_filter

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

    filtered_fires = fires_df.copy()
    if start_date and end_date:
        filtered_fires = filtered_fires[
            (filtered_fires["date"].dt.date >= start_date)
            & (filtered_fires["date"].dt.date <= end_date)
        ]

    filtered_fires = filtered_fires[
        filtered_fires["lat_cell"].between(lat_sub[0], lat_sub[1])
        & (
            filtered_fires["lon_sub"].between(lon_sub[0], lon_sub[1])
            if "lon_sub" in filtered_fires.columns
            else filtered_fires["lon_cell"].between(lon_sub[0], lon_sub[1])
        )
    ]

    total_fire_records = len(filtered_fires)
    unique_grid_cells = filtered_fires.groupby(["lat_cell", "lon_cell"]).ngroups
    total_satellite_hits = (
        int(filtered_fires["n_detections"].sum())
        if "n_detections" in filtered_fires.columns
        else total_fire_records
    )
    total_frp_mw = (
        float(filtered_fires["total_frp"].sum())
        if "total_frp" in filtered_fires.columns
        else 0.0
    )

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

    if filtered_fires.empty:
        render_callout(
            title="No Fire Records Found",
            message=(
                "No fire detections match the selected date window and geographic "
                "sub-bounds. Try broadening the observation filters."
            ),
            variant="warning",
        )
    else:
        if map_layer == "Density HeatMap Surface":
            folium_map = render_density_heatmap(filtered_fires, tiles=map_base_tile)
        elif map_layer == "Clustered Incidents":
            folium_map = render_cluster_map(filtered_fires, tiles=map_base_tile)
        else:
            folium_map = render_grid_detection_map(filtered_fires, tiles=map_base_tile)

        try:
            from streamlit_folium import st_folium

            st_folium(
                folium_map,
                width=None,
                height=620,
                use_container_width=True,
                key="observation_map",
            )
        except Exception:  # noqa: BLE001
            st.components.v1.html(folium_map._repr_html_(), height=630)

    with st.expander("Inspect & Export Filtered Geospatial Dataset", expanded=False):
        st.write(
            f"Displaying top 500 records from the currently filtered "
            f"{total_fire_records:,} fire-day observations."
        )
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
            key="geo_export_csv",
        )
