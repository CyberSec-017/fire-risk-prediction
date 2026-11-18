"""Unit and integration tests for map click synchronization and target position handling in UI V2.

Verifies:
  - First click on the map (both last_clicked and last_object_clicked).
  - Second click on a different position.
  - Effective update of target coordinates.
  - Synchronization of map marker and target position panel.
  - Click on a covered cell (displays ✅ Couvert and enables prediction).
  - Click on an uncovered cell/sea (displays ❌ Hors maillage and disables prediction).
  - Absence of selection loss after reruns.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
from streamlit.testing.v1 import AppTest
from ui_v2.components.satellite_map import build_satellite_map
from ui_v2.components.point_context import is_point_in_active_cells, get_wilaya_and_region
from ui_v2.dashboard_view import extract_folium_click

APP_PATH = ROOT / "app_v2.py"


class MockSessionState(dict):
    """Mock session state for pure function testing."""
    pass


# -----------------------------------------------------------------------------
# 1. Unit Tests for extract_folium_click logic
# -----------------------------------------------------------------------------

def test_extract_folium_click_empty(monkeypatch: pytest.MonkeyPatch):
    """Verify that empty or None map_data returns None."""
    state = MockSessionState()
    monkeypatch.setattr("streamlit.session_state", state)

    assert extract_folium_click(None) is None
    assert extract_folium_click({}) is None
    assert extract_folium_click({"last_clicked": None, "last_object_clicked": None}) is None


def test_extract_folium_click_first_map_click(monkeypatch: pytest.MonkeyPatch):
    """Verify that a first direct map click (last_clicked) is extracted and cached."""
    state = MockSessionState()
    monkeypatch.setattr("streamlit.session_state", state)

    map_data = {"last_clicked": {"lat": 36.75432, "lng": 3.05678}}
    pt = extract_folium_click(map_data)
    assert pt == (36.7543, 3.0568)
    assert state["_last_seen_map_click"] == (36.7543, 3.0568)

    # Identical click should return None to prevent unwanted rerun loops
    assert extract_folium_click(map_data) is None


def test_extract_folium_click_first_object_click(monkeypatch: pytest.MonkeyPatch):
    """Verify that a click on a vector layer (last_object_clicked) is correctly extracted."""
    state = MockSessionState()
    monkeypatch.setattr("streamlit.session_state", state)

    map_data = {
        "last_clicked": None,
        "last_object_clicked": {"lat": 36.46789, "lng": 2.83124},
        "last_object_clicked_count": 1,
    }
    pt = extract_folium_click(map_data)
    assert pt == (36.4679, 2.8312)
    assert state["_last_seen_obj_click"] == (36.4679, 2.8312)
    assert state["_last_seen_obj_count"] == 1

    # Identical object click without counter change returns None
    assert extract_folium_click(map_data) is None


def test_extract_folium_click_second_click_different_position(monkeypatch: pytest.MonkeyPatch):
    """Verify that a second click on another location updates correctly."""
    state = MockSessionState()
    monkeypatch.setattr("streamlit.session_state", state)

    # Click 1: Tizi Ouzou
    d1 = {"last_object_clicked": {"lat": 36.72, "lng": 4.05}, "last_object_clicked_count": 1}
    assert extract_folium_click(d1) == (36.72, 4.05)

    # Click 2: Oran
    d2 = {"last_object_clicked": {"lat": 35.70, "lng": -0.63}, "last_object_clicked_count": 2}
    assert extract_folium_click(d2) == (35.70, -0.63)
    assert state["_last_seen_obj_click"] == (35.70, -0.63)


def test_extract_folium_click_transition_between_object_and_map(monkeypatch: pytest.MonkeyPatch):
    """Verify smooth transition between vector object click and canvas map click."""
    state = MockSessionState()
    monkeypatch.setattr("streamlit.session_state", state)

    # Step 1: Click on grid cell object
    d1 = {
        "last_clicked": None,
        "last_object_clicked": {"lat": 36.5, "lng": 3.0},
        "last_object_clicked_count": 1,
    }
    assert extract_folium_click(d1) == (36.5, 3.0)

    # Step 2: Click on sea canvas outside cells (last_object_clicked is preserved by folium)
    d2 = {
        "last_clicked": {"lat": 38.0, "lng": 3.5},
        "last_object_clicked": {"lat": 36.5, "lng": 3.0},
        "last_object_clicked_count": 1,
    }
    assert extract_folium_click(d2) == (38.0, 3.5)

    # Step 3: Click back on another cell
    d3 = {
        "last_clicked": {"lat": 38.0, "lng": 3.5},
        "last_object_clicked": {"lat": 35.2, "lng": -0.6},
        "last_object_clicked_count": 2,
    }
    assert extract_folium_click(d3) == (35.2, -0.6)


# -----------------------------------------------------------------------------
# 2. Satellite Map Marker & Target Position Synchronization
# -----------------------------------------------------------------------------

def test_satellite_map_selected_point_marker():
    """Verify that build_satellite_map adds a dedicated highlight marker for the selected position."""
    m_no_sel = build_satellite_map(selected_point=None)
    circle_markers_none = [
        child for child in m_no_sel._children.values()
        if isinstance(child, folium.CircleMarker)
    ]
    assert len(circle_markers_none) == 0

    # With selected point
    target = (36.7200, 4.0500)
    m_sel = build_satellite_map(selected_point=target)
    circle_markers = [
        child for child in m_sel._children.values()
        if isinstance(child, folium.CircleMarker)
    ]
    assert len(circle_markers) == 1
    marker = circle_markers[0]
    assert marker.location == [36.7200, 4.0500]
    assert marker.options["fillColor"] == "#DC2626"
    tooltips = [c for c in marker._children.values() if isinstance(c, folium.Tooltip)]
    assert len(tooltips) == 1
    assert "Position ciblée" in tooltips[0].text


# -----------------------------------------------------------------------------
# 3. Integration Tests via AppTest (Streamlit UI V2)
# -----------------------------------------------------------------------------

def test_dashboard_initial_default_position_is_covered():
    """Verify that the initial target position defaults to Tizi Ouzou (covered)."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()
    assert not app.exception

    # Session state active_click
    assert app.session_state["active_click"] == (36.72, 4.05)

    # Coords inputs reflect the position
    lat_val = app.number_input(key="coord_input_lat").value
    lon_val = app.number_input(key="coord_input_lon").value
    assert lat_val == 36.72
    assert lon_val == 4.05

    # Panel shows covered status and active prediction button
    html_text = " ".join(item.value for item in app.html)
    assert "✅ Couvert" in html_text
    assert "Tizi Ouzou / Massif du Djurdjura" in html_text
    assert any("Prédire" in b.label for b in app.button)


def test_dashboard_click_update_on_covered_location():
    """Verify clicking on a covered cell (e.g. Béjaïa) updates the target panel and maintains active prediction."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()

    # Simulate click on Béjaïa (36.75, 5.08)
    app.session_state["active_click"] = (36.75, 5.08)
    app.run()
    assert not app.exception

    lat_val = app.number_input(key="coord_input_lat").value
    lon_val = app.number_input(key="coord_input_lon").value
    assert lat_val == 36.75
    assert lon_val == 5.08

    html_text = " ".join(item.value for item in app.html)
    assert "Béjaïa" in html_text
    assert "✅ Couvert" in html_text
    assert any("Prédire le Risque" in b.label for b in app.button)


def test_dashboard_click_update_on_uncovered_location():
    """Verify clicking outside active coverage (e.g. deep Sahara or sea) disables prediction and displays warning."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()

    # Offshore sea coordinates: 38.0000°N, 4.0000°E
    app.session_state["active_click"] = (38.0, 4.0)
    app.run()
    assert not app.exception

    lat_val = app.number_input(key="coord_input_lat").value
    lon_val = app.number_input(key="coord_input_lon").value
    assert lat_val == 38.0
    assert lon_val == 4.0

    html_text = " ".join(item.value for item in app.html)
    assert "❌ Hors maillage" in html_text
    assert any("hors maillage" in w.value.lower() for w in app.warning)

    # Disabled button must be shown, NOT active predict button
    disabled_btns = [b for b in app.button if "Prédiction désactivée" in b.label]
    assert len(disabled_btns) == 1
    assert disabled_btns[0].disabled is True


def test_dashboard_multiple_successive_clicks():
    """Verify that multiple successive clicks properly transition coordinates without losing state."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()

    # First click: Jijel
    app.session_state["active_click"] = (36.80, 5.76)
    app.run()
    assert app.number_input(key="coord_input_lat").value == 36.80
    assert app.number_input(key="coord_input_lon").value == 5.76
    assert "Jijel" in " ".join(item.value for item in app.html)

    # Second click: Oran
    app.session_state["active_click"] = (35.70, -0.63)
    app.run()
    assert app.number_input(key="coord_input_lat").value == 35.70
    assert app.number_input(key="coord_input_lon").value == -0.63
    assert "Oran" in " ".join(item.value for item in app.html)

    # Third click: Deep Sahara (Adrar / In Salah)
    app.session_state["active_click"] = (27.50, 0.20)
    app.run()
    assert app.number_input(key="coord_input_lat").value == 27.50
    assert app.number_input(key="coord_input_lon").value == 0.20
    assert "❌ Hors maillage" in " ".join(item.value for item in app.html)


def test_dashboard_selection_persistence_after_rerun():
    """Verify that a selected position is NOT reset or lost on arbitrary script reruns."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()

    # Set custom position
    custom_pos = (36.47, 2.83)  # Blida / Chréa
    app.session_state["active_click"] = custom_pos
    app.run()

    assert app.session_state["active_click"] == custom_pos

    # Trigger second run with no action
    app.run()
    assert not app.exception
    assert app.session_state["active_click"] == custom_pos
    assert app.number_input(key="coord_input_lat").value == 36.47
    assert app.number_input(key="coord_input_lon").value == 2.83
    assert "Blida" in " ".join(item.value for item in app.html)


# -----------------------------------------------------------------------------
# 4. Basemap Persistence Tests (Satellite vs Carte Routière)
# -----------------------------------------------------------------------------

def test_satellite_persists_after_first_click():
    """Verify: Satellite mode -> click on map -> Satellite remains active."""
    # Check folium layer configuration
    m_sat = build_satellite_map(active_basemap="satellite")
    sat_layer = next(c for c in m_sat._children.values() if isinstance(c, folium.TileLayer) and "Satellite" in c.layer_name)
    osm_layer = next(c for c in m_sat._children.values() if isinstance(c, folium.TileLayer) and "Carte routière" in c.layer_name)
    assert sat_layer.show is True
    assert osm_layer.show is False

    # Check in app execution
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()
    assert app.session_state.get("active_basemap") == "satellite"
    assert app.selectbox(key="basemap_layer_selector").value == "🛰️ Satellite"

    # Simulate click 1
    app.session_state["active_click"] = (36.75, 5.08)  # Béjaïa
    app.run()
    assert not app.exception
    assert app.session_state["active_basemap"] == "satellite"
    assert app.selectbox(key="basemap_layer_selector").value == "🛰️ Satellite"
    assert app.number_input(key="coord_input_lat").value == 36.75
    assert app.number_input(key="coord_input_lon").value == 5.08


def test_satellite_persists_after_second_click():
    """Verify: Satellite mode -> second click on map -> Satellite remains active."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()

    # Click 1
    app.session_state["active_click"] = (36.75, 5.08)
    app.run()

    # Click 2
    app.session_state["active_click"] = (35.40, 0.14)  # Mascara
    app.run()
    assert not app.exception
    assert app.session_state["active_basemap"] == "satellite"
    assert app.selectbox(key="basemap_layer_selector").value == "🛰️ Satellite"
    assert app.number_input(key="coord_input_lat").value == 35.40
    assert app.number_input(key="coord_input_lon").value == 0.14
    assert "Mascara" in " ".join(item.value for item in app.html)


def test_carte_routiere_persists_after_click():
    """Verify: Carte routière (OSM) mode -> click on map -> Carte routière remains active."""
    # Check folium layer configuration
    m_osm = build_satellite_map(active_basemap="osm")
    sat_layer = next(c for c in m_osm._children.values() if isinstance(c, folium.TileLayer) and "Satellite" in c.layer_name)
    osm_layer = next(c for c in m_osm._children.values() if isinstance(c, folium.TileLayer) and "Carte routière" in c.layer_name)
    assert sat_layer.show is False
    assert osm_layer.show is True

    # Check in app execution
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.session_state["active_basemap"] = "osm"
    app.run()
    assert app.session_state["active_basemap"] == "osm"
    assert app.selectbox(key="basemap_layer_selector").value == "🗺️ Carte routière"

    # Simulate click on map in OSM mode
    app.session_state["active_click"] = (36.88, 6.90)  # Skikda
    app.run()
    assert not app.exception
    assert app.session_state["active_basemap"] == "osm"
    assert app.selectbox(key="basemap_layer_selector").value == "🗺️ Carte routière"
    assert app.number_input(key="coord_input_lat").value == 36.88
    assert app.number_input(key="coord_input_lon").value == 6.90
    assert "Skikda" in " ".join(item.value for item in app.html)


def test_deliberate_switch_satellite_to_carte_routiere():
    """Verify: Deliberate user switch from Satellite to Carte routière in UI."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()
    assert app.session_state["active_basemap"] == "satellite"

    # Select Carte routière
    app.selectbox(key="basemap_layer_selector").select("🗺️ Carte routière")
    app.run()
    assert not app.exception
    assert app.session_state["active_basemap"] == "osm"
    assert app.selectbox(key="basemap_layer_selector").value == "🗺️ Carte routière"


def test_position_persistence_after_basemap_change():
    """Verify: Selected target position and red marker are preserved after layer change."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["app_level"] = "dashboard"
    app.run()

    # User targets Sétif (36.19, 5.41)
    setif_pos = (36.19, 5.41)
    app.session_state["active_click"] = setif_pos
    app.run()
    assert "Sétif" in " ".join(item.value for item in app.html)

    # Change basemap to Carte routière
    app.selectbox(key="basemap_layer_selector").select("🗺️ Carte routière")
    app.run()
    assert not app.exception
    assert app.session_state["active_basemap"] == "osm"
    assert app.session_state["active_click"] == setif_pos
    assert app.number_input(key="coord_input_lat").value == 36.19
    assert app.number_input(key="coord_input_lon").value == 5.41
    assert "Sétif" in " ".join(item.value for item in app.html)

    # Switch back to Satellite
    app.selectbox(key="basemap_layer_selector").select("🛰️ Satellite")
    app.run()
    assert not app.exception
    assert app.session_state["active_basemap"] == "satellite"
    assert app.session_state["active_click"] == setif_pos
    assert app.number_input(key="coord_input_lat").value == 36.19
    assert app.number_input(key="coord_input_lon").value == 5.41
    assert "Sétif" in " ".join(item.value for item in app.html)


# -----------------------------------------------------------------------------
# 5. Tests for Clean Map Appearance (Suppression of Scale Bar & Attribution)
# -----------------------------------------------------------------------------

def test_satellite_map_scale_and_attribution_controls_suppressed():
    """Verify: Technical mentions (scale bar 50 mi / 300 km and Leaflet/Esri attribution) are suppressed."""
    m = build_satellite_map()
    assert m.control_scale is False, "control_scale must be False to eliminate the scale bar"

    html = m.get_root().render()

    # 1. Leaflet map options must specify attributionControl: false
    assert '"attributionControl": false' in html

    # 2. Scoped CSS must hide .leaflet-control-attribution and .leaflet-control-scale
    assert ".leaflet-control-attribution" in html
    assert ".leaflet-control-scale" in html
    assert "display: none !important" in html


def test_both_basemaps_still_functional_without_bottom_mentions():
    """Verify: Both Satellite and Carte routière layers remain functional and toggle properly."""
    # Satellite mode
    m_sat = build_satellite_map(active_basemap="satellite")
    sat_layers = [c for c in m_sat._children.values() if isinstance(c, folium.TileLayer)]
    esri_layer = next(l for l in sat_layers if "Satellite" in l.layer_name)
    osm_layer = next(l for l in sat_layers if "Carte routière" in l.layer_name)
    assert esri_layer.show is True
    assert osm_layer.show is False

    # OSM mode
    m_osm = build_satellite_map(active_basemap="osm")
    osm_layers = [c for c in m_osm._children.values() if isinstance(c, folium.TileLayer)]
    esri_osm_layer = next(l for l in osm_layers if "Satellite" in l.layer_name)
    osm_osm_layer = next(l for l in osm_layers if "Carte routière" in l.layer_name)
    assert esri_osm_layer.show is False
    assert osm_osm_layer.show is True


def test_about_page_contains_legal_cartographic_attributions():
    """Verify: Esri and OpenStreetMap legal attributions are present in the About page."""
    import inspect
    from ui_v2.pages import about
    about_source = inspect.getsource(about.render_page)
    assert "Sources Cartographiques & Attributions Légales" in about_source
    assert "Esri World Imagery" in about_source
    assert "Esri Boundaries & Places" in about_source
    assert "OpenStreetMap" in about_source

