"""Unit tests for ui_v2/components/point_context.py.

Validates:
  - Exact count and uniqueness of the 209 NASA POWER active cells.
  - Snapping logic to the model's grid resolution (0.5° lat, 0.625° lon).
  - Geographic bounds and cell coverage verification (in-domain vs out-of-domain).
  - Wilaya and regional topography identification.
  - Cell statistics calculation strictly respecting pre-test dates (< 2023-01-01).
  - Feature building and inference compliance with the 20-feature model contract (Simulation and Live modes).
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core
from ui_v2.components.point_context import (
    get_active_209_cells,
    snap_to_grid,
    get_cell_bounding_box,
    is_point_in_active_cells,
    get_wilaya_and_region,
    get_elevation_estimate,
    get_cell_statistics,
    evaluate_point_prediction,
    evaluate_live_point_prediction,
)


def test_active_209_cells_count():
    """Verify that get_active_209_cells returns exactly 209 distinct cell coordinates."""
    cells = get_active_209_cells()
    assert isinstance(cells, set)
    assert len(cells) == 209
    for lat_c, lon_c in cells:
        assert app_core.MODEL_LAT_MIN <= lat_c <= app_core.MODEL_LAT_MAX
        assert app_core.MODEL_LON_MIN <= lon_c <= app_core.MODEL_LON_MAX


def test_snap_to_grid():
    """Verify coordinate snapping to NASA POWER 0.5° x 0.625° grid."""
    lat_s, lon_s = snap_to_grid(36.75, 3.05)
    assert abs(36.75 - lat_s) <= app_core.MODEL_LAT_STEP / 2.0 + 1e-5
    assert abs(3.05 - lon_s) <= app_core.MODEL_LON_STEP / 2.0 + 1e-5


def test_cell_bounding_box():
    """Verify calculation of cell polygon corners."""
    bbox = get_cell_bounding_box(36.5, 3.125)
    assert len(bbox) == 2
    south, west = bbox[0]
    north, east = bbox[1]
    assert south == pytest.approx(36.25, abs=1e-4)
    assert north == pytest.approx(36.75, abs=1e-4)
    assert west == pytest.approx(2.8125, abs=1e-4)
    assert east == pytest.approx(3.4375, abs=1e-4)


def test_is_point_in_active_cells_covered_points():
    """Points inside active cells should be recognized as covered."""
    is_cov, lat_c, lon_c = is_point_in_active_cells(36.0, 3.125)
    assert is_cov is True
    assert (lat_c, lon_c) in get_active_209_cells()

    is_cov2, lat_c2, lon_c2 = is_point_in_active_cells(36.5, 3.75)
    assert is_cov2 is True
    assert (lat_c2, lon_c2) in get_active_209_cells()


def test_is_point_in_active_cells_excluded_cell():
    """Specific excluded marine cells must return False."""
    is_cov, lat_c, lon_c = is_point_in_active_cells(36.5, 3.125)
    assert is_cov is False


def test_is_point_in_active_cells_offshore_sea():
    """Point in the Mediterranean Sea well north of Algeria must be flagged as out-of-bounds."""
    is_cov, lat_c, lon_c = is_point_in_active_cells(38.8, 3.0)
    assert is_cov is False


def test_is_point_in_active_cells_deep_sahara():
    """Point in deep southern Sahara outside the study box must be flagged as out-of-bounds."""
    is_cov, lat_c, lon_c = is_point_in_active_cells(26.0, 3.0)
    assert is_cov is False


def test_wilaya_and_elevation_resolution():
    """Verify wilaya mapping and elevation approximation."""
    wilaya_tizi = get_wilaya_and_region(36.72, 4.05)
    assert "Tizi Ouzou" in wilaya_tizi or "Djurdjura" in wilaya_tizi

    wilaya_tarf = get_wilaya_and_region(36.77, 8.31)
    assert "El Tarf" in wilaya_tarf or "El Kala" in wilaya_tarf

    alt = get_elevation_estimate(36.72, 4.05)
    assert isinstance(alt, int)
    assert alt > 0


def test_get_cell_statistics_pre_test_isolation():
    """Test that cell statistics strictly filter out test set dates (>= 2023-01-01)."""
    data = {
        "date": ["2021-07-01", "2022-08-01", "2023-07-15", "2024-08-10"],
        "lat_cell": [36.5, 36.5, 36.5, 36.5],
        "lon_cell": [3.125, 3.125, 3.125, 3.125],
        "burned": [1, 0, 1, 1],
        "T2M": [30.0, 32.0, 40.0, 42.0],
        "T2M_MAX": [36.0, 38.0, 45.0, 46.0],
        "RH2M": [30.0, 40.0, 15.0, 12.0],
        "WS2M": [4.0, 5.0, 8.0, 9.0],
        "PRECTOTCORR": [0.0, 0.0, 0.0, 0.0],
        "GWETROOT": [0.2, 0.3, 0.1, 0.05],
        "dry_days": [15.0, 20.0, 35.0, 40.0],
    }
    df = pd.DataFrame(data)

    stats = get_cell_statistics(36.5, 3.125, panel_df=df)
    assert stats["total_days"] == 2
    assert stats["fire_days"] == 1
    assert stats["fire_rate"] == 0.5
    assert stats["t2m_median"] == pytest.approx(31.0, abs=1e-4)


def test_evaluate_point_prediction_simulation_contract():
    """Verify that evaluate_point_prediction satisfies model contract in simulation mode."""
    bundle = app_core.load_model_bundle()
    weather_inputs = {
        "T2M": 32.0,
        "T2M_MAX": 38.5,
        "RH2M": 22.0,
        "WS2M": 6.5,
        "PRECTOTCORR": 0.0,
        "GWETROOT": 0.18,
        "dry_days": 18.0,
    }

    res = evaluate_point_prediction(
        bundle=bundle,
        lat=36.0,
        lon=3.125,
        when="2022-07-20",
        weather_inputs=weather_inputs,
    )

    assert isinstance(res, dict)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_level"] in ("LOW", "MODERATE", "HIGH", "VERY HIGH")
    assert res["score_source"] in ("REAL", "DEMO")
    assert res["mode"] == "MANUAL_SIMULATION"
    assert "persistence_note" in res
    assert len(res["feature_row"].columns) == 20
    assert list(res["feature_row"].columns) == bundle.feature_names


def test_evaluate_live_point_prediction_contract():
    """Verify that evaluate_live_point_prediction satisfies model contract in live mode."""
    bundle = app_core.load_model_bundle()
    # Create synthetic 40-day history matching NASA POWER output
    dates = pd.date_range("2022-06-10", "2022-07-20")
    n = len(dates)
    history_df = pd.DataFrame({
        "date": dates,
        "lat_cell": [36.0] * n,
        "lon_cell": [3.125] * n,
        "T2M": np.linspace(25.0, 32.0, n),
        "T2M_MAX": np.linspace(30.0, 38.0, n),
        "RH2M": np.linspace(60.0, 25.0, n),
        "WS2M": [4.0] * n,
        "PRECTOTCORR": [0.0] * n,
        "GWETROOT": np.linspace(0.4, 0.2, n),
    })

    res = evaluate_live_point_prediction(
        bundle=bundle,
        lat=36.0,
        lon=3.125,
        when="2022-07-20",
        history_df=history_df,
    )

    assert isinstance(res, dict)
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_level"] in ("LOW", "MODERATE", "HIGH", "VERY HIGH")
    assert res["mode"] == "AUTOMATIC_NASA_POWER"
    assert len(res["feature_row"].columns) == 20
    assert list(res["feature_row"].columns) == bundle.feature_names
