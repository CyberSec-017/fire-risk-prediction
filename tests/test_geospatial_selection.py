"""Tests for the click-to-select geospatial prediction workflow.

Covers:
  * the Algeria polygon + model bounding-box validation,
  * nearest-cell snapping and distance disclosure,
  * complete/missing feature data,
  * agreement with the existing panel scoring pipeline,
  * page interaction and Streamlit session-state persistence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core as core  # noqa: E402
from dashboard import geo  # noqa: E402
from dashboard.maps import (  # noqa: E402
    render_selection_map,
    selection_marker_group,
    viewport_from_map_data,
)
from dashboard.model_loader import score_panel_date  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = next((ROOT / "pages").glob("2_*Geospatial_Monitoring.py"))


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "fire_panel.parquet")


@pytest.fixture(scope="module")
def bundle() -> core.ModelBundle:
    bundle = core.load_model_bundle()
    assert bundle.status == "REAL"
    return bundle


@pytest.fixture(scope="module")
def feature_names(bundle) -> list[str]:
    return [str(name) for name in bundle.card["feature_names"]]


@pytest.fixture(scope="module")
def latest_date(panel) -> pd.Timestamp:
    return pd.Timestamp(panel["date"].max()).normalize()


# ---------------------------------------------------------------------------
# Polygon / study-area validation
# ---------------------------------------------------------------------------

def test_algeria_boundary_file_is_present():
    assert geo.STUDY_POLYGON_PATH.exists()
    ring = geo.study_area_polygon()
    assert len(ring) > 100
    assert all(len(point) == 2 for point in ring)


def test_point_in_polygon_ray_casting_with_synthetic_square():
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert geo.point_in_polygon(5.0, 5.0, ring)
    assert not geo.point_in_polygon(15.0, 5.0, ring)
    assert not geo.point_in_polygon(-1.0, 5.0, ring)


def test_inland_algeria_is_inside_study_area():
    assert geo.point_in_study_area(35.0, 5.0)
    assert geo.point_in_study_area(36.75, 3.06)  # Algiers


@pytest.mark.parametrize(
    "lat,lon,label",
    [
        (37.5, 3.0, "Mediterranean Sea"),
        (35.0, 9.0, "Tunisia"),
        (34.0, -2.0, "Morocco"),
        (31.0, 3.0, "south of the model domain"),
    ],
)
def test_outside_points_are_rejected(lat, lon, label):
    assert not geo.point_in_study_area(lat, lon), label


def test_resolve_point_reports_nearest_cell_and_distance():
    point = geo.resolve_point(35.1, 5.1)

    assert point.in_area
    assert point.lat_cell == pytest.approx(35.0)
    assert point.lon_cell == pytest.approx(5.0)
    assert 5.0 < point.distance_km < 25.0
    assert point.reason is None


def test_resolve_outside_point_reports_reason():
    point = geo.resolve_point(37.5, 3.0)

    assert not point.in_area
    assert point.in_bbox
    assert not point.in_polygon
    assert "outside algeria" in (point.reason or "").lower()


# ---------------------------------------------------------------------------
# Feature availability
# ---------------------------------------------------------------------------

def _single_cell_panel(
    panel: pd.DataFrame,
    date: pd.Timestamp,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    mask = (
        (dates == pd.Timestamp(date).normalize())
        & np.isclose(panel["lat_cell"], lat)
        & np.isclose(panel["lon_cell"], lon)
    )
    return panel.loc[mask].head(1).copy()


def test_complete_feature_row_is_accepted(panel, feature_names, latest_date):
    availability = geo.feature_row_for(
        panel, latest_date, 35.0, 5.0, feature_names
    )

    assert availability.ok
    assert availability.missing == ()
    assert availability.row is not None
    assert availability.row[feature_names].notna().all()


def test_missing_feature_value_is_reported(panel, feature_names, latest_date):
    single = _single_cell_panel(panel, latest_date, 35.0, 5.0)
    single.loc[:, "GWETROOT"] = np.nan

    availability = geo.feature_row_for(
        single, latest_date, 35.0, 5.0, feature_names
    )

    assert not availability.ok
    assert "GWETROOT" in availability.missing
    assert "GWETROOT" in (availability.reason or "")
    assert "blocked" in (availability.reason or "").lower()


def test_missing_cell_or_date_is_reported(panel, feature_names):
    availability = geo.feature_row_for(
        panel, "1999-01-01", 35.0, 5.0, feature_names
    )

    assert not availability.ok
    assert "No panel row" in (availability.reason or "")


def test_off_grid_cell_is_rejected(panel, feature_names, latest_date):
    availability = geo.feature_row_for(
        panel, latest_date, 35.5, 5.5, feature_names
    )

    assert not availability.ok
    assert "not on the model's" in (availability.reason or "").lower()


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def test_prediction_matches_existing_panel_scoring(
    panel, bundle, latest_date
):
    prediction = geo.predict_point(bundle, panel, 35.0, 5.0, latest_date)
    scores = score_panel_date(bundle, panel, latest_date)
    match = scores[
        np.isclose(scores["lat_cell"], 35.0)
        & np.isclose(scores["lon_cell"], 5.0)
    ].iloc[0]

    assert prediction.probability == pytest.approx(float(match["probability"]))
    assert prediction.risk == str(match["risk"]).upper()
    assert prediction.lat_cell == pytest.approx(35.0)
    assert prediction.lon_cell == pytest.approx(5.0)
    assert prediction.horizon_days == 3
    assert prediction.date == latest_date


def test_prediction_outside_study_area_raises(panel, bundle, latest_date):
    with pytest.raises(geo.OutsideStudyAreaError):
        geo.predict_point(bundle, panel, 37.5, 3.0, latest_date)


def test_prediction_with_missing_feature_raises(
    panel, bundle, feature_names, latest_date
):
    single = _single_cell_panel(panel, latest_date, 35.0, 5.0)
    single.loc[:, "GWETROOT"] = np.nan

    with pytest.raises(geo.FeatureDataUnavailable) as excinfo:
        geo.predict_point(
            bundle,
            single,
            35.0,
            5.0,
            latest_date,
            feature_names=feature_names,
        )

    assert "GWETROOT" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Streamlit page / session state
# ---------------------------------------------------------------------------

def run_page() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=180)
    app.run()
    assert not app.exception
    return app


def predict_button(app: AppTest):
    return next(
        button for button in app.button if button.label == "Predict Fire Risk"
    )


def rendered_markdown(app: AppTest) -> str:
    return "\n".join(item.value for item in app.markdown)


def test_predict_button_disabled_without_selection():
    app = run_page()

    assert predict_button(app).disabled
    assert app.session_state.get("geo_result") is None


def test_valid_selection_predicts_and_displays_result():
    app = run_page()
    app.session_state["geo_selection"] = {"lat": 35.0, "lon": 5.0}
    app.run()

    assert not app.exception
    button = predict_button(app)
    assert not button.disabled

    button.click().run()

    assert not app.exception
    result = app.session_state.get("geo_result")
    assert result is not None
    assert result["risk"] in {"LOW", "MODERATE", "HIGH", "VERY HIGH"}
    assert 0.0 <= float(result["probability"]) <= 1.0
    assert result["lat_cell"] == pytest.approx(35.0)
    assert result["lon_cell"] == pytest.approx(5.0)

    rendered = rendered_markdown(app)
    assert "Risk Category" in rendered
    assert "3 days" in rendered
    assert "35.0000" in rendered


def test_outside_selection_warns_and_disables_prediction():
    app = run_page()
    app.session_state["geo_selection"] = {"lat": 37.5, "lon": 3.0}
    app.run()

    assert not app.exception
    assert predict_button(app).disabled
    assert app.session_state.get("geo_result") is None
    assert "outside the study area" in rendered_markdown(app).lower()


def test_selection_and_result_survive_a_rerun():
    app = run_page()
    app.session_state["geo_selection"] = {"lat": 35.0, "lon": 5.0}
    app.run()
    predict_button(app).click().run()

    result = app.session_state.get("geo_result")
    assert result is not None

    app.run()

    assert not app.exception
    assert app.session_state.get("geo_selection") == {
        "lat": 35.0,
        "lon": 5.0,
    }
    assert app.session_state.get("geo_result") == result


# ---------------------------------------------------------------------------
# Map viewport and marker stability
# ---------------------------------------------------------------------------

def _map_payload(center, zoom, clicked=None):
    lat, lng = center
    return {
        "last_clicked": clicked,
        "center": {"lat": lat, "lng": lng},
        "zoom": zoom,
        "bounds": {
            "_southWest": {"lat": lat - 1.0, "lng": lng - 1.0},
            "_northEast": {"lat": lat + 1.0, "lng": lng + 1.0},
        },
    }


def _install_fake_folium(monkeypatch):
    """Replace st_folium so tests can drive viewport/click payloads."""
    import streamlit_folium

    state = {"payload": None}
    calls = []

    def fake_st_folium(fig, **kwargs):
        calls.append(kwargs)
        return state["payload"]

    monkeypatch.setattr(streamlit_folium, "st_folium", fake_st_folium)
    return state, calls


def test_viewport_from_map_data_normalises_center_zoom_and_bounds():
    payload = _map_payload((34.25, 3.5), 9)

    assert viewport_from_map_data(payload) == {
        "center": {"lat": 34.25, "lng": 3.5},
        "zoom": 9,
        "bounds": payload["bounds"],
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"center": None, "zoom": 8},
        {"center": {"lat": 34.0, "lng": 3.0}},
        {"center": {"lat": 34.0, "lng": 3.0}, "zoom": None},
    ],
)
def test_viewport_from_map_data_rejects_incomplete_payload(payload):
    assert viewport_from_map_data(payload) is None


def test_selection_marker_group_contains_exactly_one_marker():
    group = selection_marker_group((35.0, 5.0))

    assert group is not None
    markers = [
        child
        for child in group._children.values()
        if isinstance(child, folium.Marker)
    ]
    assert len(markers) == 1
    assert markers[0].location == [35.0, 5.0]


def test_selection_marker_group_is_none_without_selection():
    assert selection_marker_group(None) is None


def test_selection_does_not_change_the_base_map_script():
    """The base map (and therefore the widget identity) must stay stable."""
    from streamlit_folium import _get_map_string, generate_js_hash

    def script_hash(selected=None):
        base = render_selection_map()
        if selected is not None:
            selection_marker_group(selected).add_to(base)
        return generate_js_hash(_get_map_string(base), "geo_select_map", False)

    baseline = script_hash()

    assert script_hash() == baseline
    assert script_hash(selected=(35.0, 5.0)) != baseline
    assert script_hash(selected=(36.0, 7.0)) != baseline


def test_viewport_is_persisted_and_restored_when_basemap_changes(monkeypatch):
    state, calls = _install_fake_folium(monkeypatch)
    app = AppTest.from_file(str(PAGE), default_timeout=180)

    state["payload"] = _map_payload((34.75, 3.25), 9)
    app.run()
    assert not app.exception
    assert app.session_state["geo_viewport"] == {
        "center": {"lat": 34.75, "lng": 3.25},
        "zoom": 9,
        "bounds": state["payload"]["bounds"],
    }

    basemap = next(
        widget for widget in app.selectbox if widget.key == "geo_basemap"
    )
    basemap.set_value("Dark").run()
    assert not app.exception

    # Changing the basemap rewrites the base-map script, so the map must be
    # explicitly put back at the user's last viewport.
    selection_calls = [
        call for call in calls if call.get("key") == "geo_select_map"
    ]
    last_call = selection_calls[-1]
    assert last_call["center"] == (34.75, 3.25)
    assert last_call["zoom"] == 9


def test_click_updates_only_the_marker_and_keeps_the_viewport(monkeypatch):
    state, calls = _install_fake_folium(monkeypatch)
    app = AppTest.from_file(str(PAGE), default_timeout=180)

    # Simulate the user zooming in and panning before clicking.
    state["payload"] = _map_payload((34.75, 3.25), 11)
    app.run()
    assert app.session_state["geo_viewport"]["zoom"] == 11

    state["payload"] = _map_payload(
        (34.75, 3.25),
        11,
        clicked={"lat": 35.0, "lng": 5.0},
    )
    app.run()

    assert not app.exception
    assert app.session_state["geo_selection"] == {"lat": 35.0, "lon": 5.0}
    assert app.session_state["geo_viewport"] == {
        "center": {"lat": 34.75, "lng": 3.25},
        "zoom": 11,
        "bounds": state["payload"]["bounds"],
    }

    # The selection widget is never told to jump to a centre/zoom on a click;
    # it keeps the user's live view and only the marker feature group changes.
    selection_calls = [
        call for call in calls if call.get("key") == "geo_select_map"
    ]
    assert all(call.get("center") is None for call in selection_calls)
    assert all(call.get("zoom") is None for call in selection_calls)

    marker_group = selection_calls[-1]["feature_group_to_add"]
    assert marker_group is not None
    markers = [
        child
        for child in marker_group._children.values()
        if isinstance(child, folium.Marker)
    ]
    assert len(markers) == 1
    assert markers[0].location == [35.0, 5.0]
