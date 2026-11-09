"""Tests for the predicted-risk layer's scoring and map helpers.

These cover the pure, UI-free pieces behind page 2's new
"Predicted Fire Risk (model)" layer:

* ``dashboard.model_loader.model_area_mask`` / ``score_panel_date``
* ``dashboard.maps.render_risk_map`` and the keyless basemap selection
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_core as core  # noqa: E402
from dashboard import maps  # noqa: E402
from dashboard.config import COLORS  # noqa: E402
from dashboard.maps import create_base_map, render_risk_map  # noqa: E402
from dashboard.model_loader import model_area_mask, score_panel_date  # noqa: E402


CARD = core.load_model_bundle().card


def feature_row(lat: float, lon: float, **overrides) -> dict:
    """A complete 20-feature model row for one grid cell."""
    values = {
        "lat_cell": lat,
        "lon_cell": lon,
        "T2M": 30.0,
        "T2M_MAX": 40.0,
        "RH2M": 25.0,
        "WS2M": 6.0,
        "PRECTOTCORR": 0.0,
        "GWETROOT": 0.15,
        "rain_7d": 0.0,
        "tmax_7d": 39.0,
        "rhmin_7d": 18.0,
        "rain_14d": 0.0,
        "tmax_14d": 38.0,
        "rhmin_14d": 20.0,
        "rain_30d": 0.0,
        "tmax_30d": 37.0,
        "rhmin_30d": 22.0,
        "dry_days": 12.0,
        "doy_sin": 0.5,
        "doy_cos": -0.8,
    }
    values.update(overrides)
    return values


def sample_panel() -> pd.DataFrame:
    """Three on-grid cells plus one off-grid cell on 2023-07-15.

    Longitudes on the model grid are multiples of 0.625°; 3.0° is not, so that
    cell must be treated as outside the model's study area.
    """
    rows = []
    for lat, lon in [(36.0, 2.5), (36.5, 3.125), (37.0, 3.75), (36.0, 3.0)]:
        row = feature_row(lat, lon)
        row["date"] = pd.Timestamp("2023-07-15")
        row["burned"] = 1 if (lat, lon) == (36.0, 2.5) else 0
        rows.append(row)
    for lat, lon in [(35.0, 2.5), (35.5, 3.125)]:
        row = feature_row(lat, lon)
        row["date"] = pd.Timestamp("2023-07-16")
        row["burned"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def risk_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_cell": [36.0, 36.5, 37.0],
            "lon_cell": [2.5, 3.125, 3.75],
            "probability": [0.001, 0.5, 0.07],
            "risk": ["LOW", "VERY HIGH", "HIGH"],
            "burned": [1, 0, 1],
        }
    )


def outside_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_cell": [36.0, 36.5],
            "lon_cell": [3.0, 3.125],
            "probability": [np.nan, 0.5],
            "risk": ["OUTSIDE", "VERY HIGH"],
            "burned": [0, 0],
            "in_model_area": [False, True],
        }
    )


def circle_markers(m: folium.Map) -> list[folium.CircleMarker]:
    return [
        child
        for child in m._children.values()
        if isinstance(child, folium.CircleMarker)
    ]


def tile_layer(m: folium.Map) -> folium.TileLayer:
    layers = [
        child
        for child in m._children.values()
        if isinstance(child, folium.TileLayer)
    ]
    assert len(layers) == 1
    return layers[0]


def tooltip_text(marker: folium.CircleMarker) -> str:
    children = list(marker._children.values())
    assert len(children) == 1
    return children[0].text


# ---------------------------------------------------------------------------
# model_area_mask / score_panel_date
# ---------------------------------------------------------------------------

def test_model_area_mask_flags_off_grid_longitudes():
    panel = sample_panel()
    day = panel[pd.to_datetime(panel["date"]) == pd.Timestamp("2023-07-15")]
    mask = model_area_mask(day.reset_index(drop=True))

    assert mask.tolist() == [True, True, True, False]
    assert int(mask.sum()) == 3


def test_score_panel_date_scores_only_the_requested_day():
    bundle = core.load_model_bundle()
    scored = score_panel_date(bundle, sample_panel(), "2023-07-15")

    assert len(scored) == 4
    assert list(scored.columns) == [
        "lat_cell",
        "lon_cell",
        "probability",
        "risk",
        "burned",
        "in_model_area",
    ]
    assert scored[["lat_cell", "lon_cell"]].values.tolist() == [
        [36.0, 2.5],
        [36.5, 3.125],
        [37.0, 3.75],
        [36.0, 3.0],
    ]


def test_score_panel_date_probabilities_are_bounded_and_banded():
    bundle = core.load_model_bundle()
    scored = score_panel_date(bundle, sample_panel(), "2023-07-15")
    inside = scored[scored["in_model_area"]]
    outside = scored[~scored["in_model_area"]]

    assert len(inside) == 3
    assert len(outside) == 1
    assert inside["probability"].between(0.0, 1.0).all()
    assert inside["probability"].notna().all()
    assert set(inside["risk"]).issubset(set(CARD["risk_labels"]))
    assert scored["burned"].isin([0, 1]).all()

    assert outside["probability"].isna().all()
    assert (outside["risk"] == "OUTSIDE").all()


def test_score_panel_date_matches_predict_batch_for_in_area_cells():
    bundle = core.load_model_bundle()
    panel = sample_panel()
    scored = score_panel_date(bundle, panel, "2023-07-15")

    day = panel[pd.to_datetime(panel["date"]) == pd.Timestamp("2023-07-15")]
    day = day.reset_index(drop=True)
    mask = model_area_mask(day).to_numpy()
    expected = core.predict_batch(bundle, day.loc[mask])

    actual = scored.loc[scored["in_model_area"], "probability"].to_numpy()
    assert actual == pytest.approx(expected)


def test_score_panel_date_missing_feature_raises():
    bundle = core.load_model_bundle()
    panel = sample_panel().drop(columns=["GWETROOT"])

    with pytest.raises(core.FeatureContractError):
        score_panel_date(bundle, panel, "2023-07-15")


def test_score_panel_date_unknown_date_returns_empty_frame():
    bundle = core.load_model_bundle()
    scored = score_panel_date(bundle, sample_panel(), "2030-01-01")

    assert scored.empty
    assert list(scored.columns) == [
        "lat_cell",
        "lon_cell",
        "probability",
        "risk",
        "burned",
        "in_model_area",
    ]


def test_score_panel_date_non_real_bundle_raises():
    demo_bundle = core.ModelBundle(
        status="DEMO",
        model=None,
        card=CARD,
        error=None,
        model_path=core.MODEL_PATH,
        card_path=core.MODEL_CARD_PATH,
    )

    with pytest.raises(core.ModelNotReadyError):
        score_panel_date(demo_bundle, sample_panel(), "2023-07-15")


# ---------------------------------------------------------------------------
# render_risk_map
# ---------------------------------------------------------------------------

def test_render_risk_map_legend_uses_card_bands():
    m = render_risk_map(risk_scores(), CARD, tiles="OpenStreetMap")
    html = m.get_root().render()

    assert "Predicted Fire Risk" in html
    assert "Low (0.0% – 1.0%)" in html
    assert "Moderate (1.0% – 5.0%)" in html
    assert "High (5.0% – 15.0%)" in html
    assert "Very High (15.0%+)" in html
    assert "Outside the model's study area (no prediction)" in html


def test_render_risk_map_tooltips_and_colors_follow_risk_band():
    m = render_risk_map(risk_scores(), CARD, tiles="OpenStreetMap")
    markers = circle_markers(m)

    assert len(markers) == 3

    low, very_high, high = markers
    assert tooltip_text(low) == "36.00°N, 2.50°E • 0.1% • Low"
    assert tooltip_text(very_high) == "36.50°N, 3.12°E • 50.0% • Very High"
    assert tooltip_text(high) == "37.00°N, 3.75°E • 7.0% • High"

    assert low.options["fillColor"] == COLORS["risk_low"]
    assert very_high.options["fillColor"] == COLORS["risk_very_high"]
    assert high.options["fillColor"] == COLORS["risk_high"]


def test_render_risk_map_derives_band_when_risk_column_missing():
    scores = risk_scores().drop(columns=["risk"])
    m = render_risk_map(scores, CARD, tiles="OpenStreetMap")
    markers = circle_markers(m)

    assert [tooltip_text(marker) for marker in markers] == [
        "36.00°N, 2.50°E • 0.1% • Low",
        "36.50°N, 3.12°E • 50.0% • Very High",
        "37.00°N, 3.75°E • 7.0% • High",
    ]
    assert markers[0].options["fillColor"] == COLORS["risk_low"]
    assert markers[1].options["fillColor"] == COLORS["risk_very_high"]


def test_render_risk_map_outside_cells_are_gray_and_unscored():
    m = render_risk_map(outside_scores(), CARD, tiles="OpenStreetMap")
    markers = circle_markers(m)

    assert len(markers) == 2
    outside, inside = markers
    assert outside.options["fillColor"] == COLORS["text_muted"]
    assert tooltip_text(outside) == (
        "Outside the model's study area (no prediction)"
    )
    assert inside.options["fillColor"] == COLORS["risk_very_high"]
    assert "50.0% • Very High" in tooltip_text(inside)


def test_render_risk_map_outside_cells_inferred_from_nan_probability():
    scores = pd.DataFrame(
        {
            "lat_cell": [36.0, 36.5],
            "lon_cell": [3.0, 3.125],
            "probability": [np.nan, 0.5],
            "risk": ["OUTSIDE", "VERY HIGH"],
            "burned": [0, 0],
        }
    )
    m = render_risk_map(scores, CARD, tiles="OpenStreetMap")
    outside, inside = circle_markers(m)

    assert outside.options["fillColor"] == COLORS["text_muted"]
    assert tooltip_text(outside) == (
        "Outside the model's study area (no prediction)"
    )
    assert inside.options["fillColor"] == COLORS["risk_very_high"]


def test_render_risk_map_observed_overlay_is_black_outline():
    m = render_risk_map(
        risk_scores(), CARD, tiles="OpenStreetMap", show_observed=True
    )
    observed = [
        marker
        for marker in circle_markers(m)
        if marker.options.get("color") == "#000000"
    ]

    assert len(observed) == 2
    for marker in observed:
        assert marker.options["fill"] is False
        assert marker.options["weight"] == 2
        assert "Observed fire" in tooltip_text(marker)


def test_render_risk_map_observed_overlay_can_be_hidden():
    m = render_risk_map(
        risk_scores(), CARD, tiles="OpenStreetMap", show_observed=False
    )
    observed = [
        marker
        for marker in circle_markers(m)
        if marker.options.get("color") == "#000000"
    ]

    assert observed == []


def test_render_risk_map_empty_scores_still_draws_legend():
    m = render_risk_map(pd.DataFrame(), CARD, tiles="OpenStreetMap")
    html = m.get_root().render()

    assert "Predicted Fire Risk" in html
    assert "Very High (15.0%+)" in html
    assert circle_markers(m) == []


def test_render_risk_map_missing_probability_raises():
    scores = risk_scores().drop(columns=["probability"])

    with pytest.raises(ValueError, match="probability"):
        render_risk_map(scores, CARD, tiles="OpenStreetMap")


# ---------------------------------------------------------------------------
# Keyless basemaps
# ---------------------------------------------------------------------------

def test_create_base_map_defaults_to_light_esri():
    layer = tile_layer(create_base_map())

    assert "World_Light_Gray_Base" in layer.tiles
    assert layer.options["attribution"] == "Tiles © Esri"
    assert layer.options["no_wrap"] is True
    assert layer.layer_name == "Light"


def test_create_base_map_dark_is_esri_dark():
    layer = tile_layer(create_base_map(tiles="Dark"))

    assert "World_Dark_Gray_Base" in layer.tiles
    assert layer.options["attribution"] == "Tiles © Esri"
    assert layer.options["no_wrap"] is True
    assert layer.layer_name == "Dark"


def test_create_base_map_openstreetmap_keeps_own_attribution():
    layer = tile_layer(create_base_map(tiles="OpenStreetMap"))

    assert "openstreetmap.org" in layer.tiles
    assert "OpenStreetMap" in layer.options["attribution"]
    assert layer.options["no_wrap"] is True


def test_every_map_builder_defaults_to_light():
    functions = [
        maps.create_base_map,
        maps.render_grid_detection_map,
        maps.render_density_heatmap,
        maps.render_cluster_map,
        maps.render_risk_map,
    ]
    for function in functions:
        assert inspect.signature(function).parameters["tiles"].default == "Light"


def test_no_cartodb_tiles_remain():
    source = Path(maps.__file__).read_text(encoding="utf-8")

    assert "CartoDB" not in source
