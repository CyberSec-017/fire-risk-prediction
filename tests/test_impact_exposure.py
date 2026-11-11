"""Unit tests for impact exposure calculations.

All fixtures are synthetic frames: these tests never read the real exposure
artifacts and never touch the model or the fire panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact import config  # noqa: E402
from impact import exposure  # noqa: E402


def exposure_frame() -> pd.DataFrame:
    """Three plausible cells whose land-cover areas sum to the cell area."""
    return pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3"],
            "lat_cell": [36.0, 36.5, 35.0],
            "lon_cell": [3.125, 3.75, 2.5],
            "cell_area_km2": [3100.0, 3100.0, 3100.0],
            "lc_tree_km2": [500.0, 0.0, 100.0],
            "lc_shrub_km2": [300.0, 0.0, 100.0],
            "lc_grass_km2": [200.0, 0.0, 100.0],
            "lc_crop_km2": [100.0, 0.0, 100.0],
            "lc_bare_km2": [1500.0, 3100.0, 2000.0],
            "lc_built_km2": [100.0, 0.0, 200.0],
            "lc_water_km2": [50.0, 0.0, 100.0],
            "lc_other_km2": [350.0, 0.0, 400.0],
            "pop_cell": [10_000.0, 0.0, 500.0],
            "pop_5km": [20_000.0, 0.0, 1_000.0],
            "pop_10km": [30_000.0, 0.0, 1_500.0],
            "pop_25km": [50_000.0, 0.0, 2_000.0],
            "settlement_count": [3.0, 0.0, 1.0],
            "settlement_population": [9_000.0, 0.0, 400.0],
            "road_km_total": [120.0, 0.0, 20.0],
            "road_km_primary": [40.0, 0.0, 5.0],
            "building_count": [400.0, 0.0, 25.0],
            "building_footprint_km2": [1.2, 0.0, 0.05],
            "critical_count_total": [5.0, 0.0, 1.0],
            "critical_hospital_count": [1.0, 0.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_cell_bounds_extend_half_a_grid_step():
    lat_min, lat_max, lon_min, lon_max = exposure.cell_bounds(36.0, 3.0)
    assert lat_min == pytest.approx(35.75)
    assert lat_max == pytest.approx(36.25)
    assert lon_min == pytest.approx(3.0 - config.MODEL_LON_STEP / 2)
    assert lon_max == pytest.approx(3.0 + config.MODEL_LON_STEP / 2)


def test_cell_area_matches_independent_spherical_formula():
    lat_cell, lon_cell = 36.0, 3.0
    lat_min, lat_max, lon_min, lon_max = exposure.cell_bounds(lat_cell, lon_cell)
    expected = config.EARTH_RADIUS_KM**2 * np.radians(lon_max - lon_min) * (
        np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min))
    )
    assert exposure.cell_area_km2(lat_cell, lon_cell) == pytest.approx(
        float(expected), rel=1e-12
    )
    assert 3_000 < exposure.cell_area_km2(lat_cell, lon_cell) < 3_400


def test_cell_area_decreases_toward_the_north():
    assert exposure.cell_area_km2(32.0, 3.0) > exposure.cell_area_km2(37.5, 3.0)


def test_is_on_model_grid_accepts_valid_and_rejects_off_grid():
    assert exposure.is_on_model_grid(36.0, 3.125)
    assert exposure.is_on_model_grid(32.0, -1.875)
    assert not exposure.is_on_model_grid(36.1, 3.125)
    assert not exposure.is_on_model_grid(36.0, 3.1)
    assert not exposure.is_on_model_grid(40.0, 3.125)
    assert not exposure.is_on_model_grid(36.0, 12.0)


def test_grid_alignment_issues_are_reported():
    clean = exposure_frame()
    assert exposure.grid_alignment_issues(clean) == []

    duplicated = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)
    issues = exposure.grid_alignment_issues(duplicated)
    assert any("duplicate" in issue for issue in issues)

    outside = clean.copy()
    outside.loc[0, "lat_cell"] = 99.0
    assert any("outside" in issue for issue in exposure.grid_alignment_issues(outside))


# ---------------------------------------------------------------------------
# Environmental dimension
# ---------------------------------------------------------------------------
def test_environmental_exposure_computes_flammable_area():
    out = exposure.environmental_exposure(exposure_frame())

    assert out.loc[0, "fuel_area_km2"] == pytest.approx(1100.0)
    assert out.loc[0, "non_fuel_area_km2"] == pytest.approx(2000.0)
    assert out.loc[0, "fuel_fraction"] == pytest.approx(1100.0 / 3100.0)
    assert out.loc[0, "forest_fraction"] == pytest.approx(500.0 / 3100.0)
    assert out.loc[0, "landcover_coverage_fraction"] == pytest.approx(1.0)
    assert bool(out.loc[0, "landcover_complete"]) is True

    # A cell with no vegetation has no fuel, but that is a real zero.
    assert out.loc[1, "fuel_area_km2"] == pytest.approx(0.0)
    assert out.loc[1, "fuel_fraction"] == pytest.approx(0.0)


def test_environmental_missing_column_raises_without_zeroing():
    frame = exposure_frame().drop(columns=["lc_shrub_km2"])
    with pytest.raises(exposure.MissingColumnsError) as excinfo:
        exposure.environmental_exposure(frame)
    assert "lc_shrub_km2" in str(excinfo.value)
    assert "never replaced with zeros" in str(excinfo.value)


def test_environmental_nan_raises_instead_of_becoming_zero():
    frame = exposure_frame()
    frame.loc[0, "lc_tree_km2"] = np.nan
    with pytest.raises(exposure.IncompleteDataError) as excinfo:
        exposure.environmental_exposure(frame)
    assert "lc_tree_km2" in str(excinfo.value)


def test_environmental_negative_area_raises():
    frame = exposure_frame()
    frame.loc[0, "lc_grass_km2"] = -1.0
    with pytest.raises(exposure.InconsistentDataError):
        exposure.environmental_exposure(frame)


def test_environmental_area_larger_than_cell_raises():
    frame = exposure_frame()
    frame.loc[0, "lc_bare_km2"] = 4_000.0
    with pytest.raises(exposure.InconsistentDataError) as excinfo:
        exposure.environmental_exposure(frame)
    assert "exceed the cell area" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Population dimension
# ---------------------------------------------------------------------------
def test_population_exposure_adds_density_and_preserves_rings():
    out = exposure.population_exposure(exposure_frame())
    assert out.loc[0, "pop_density_per_km2"] == pytest.approx(10_000.0 / 3100.0)
    assert out.loc[0, "pop_25km"] == pytest.approx(50_000.0)
    assert bool(out.loc[0, "settlement_data_available"]) is True
    assert out.loc[1, "pop_density_per_km2"] == pytest.approx(0.0)


def test_population_missing_ring_column_raises():
    frame = exposure_frame().drop(columns=["pop_10km"])
    with pytest.raises(exposure.MissingColumnsError) as excinfo:
        exposure.population_exposure(frame)
    assert "pop_10km" in str(excinfo.value)


def test_population_negative_value_raises():
    frame = exposure_frame()
    frame.loc[0, "pop_cell"] = -5.0
    with pytest.raises(exposure.InconsistentDataError):
        exposure.population_exposure(frame)


def test_optional_settlement_nan_is_flagged_not_zeroed():
    frame = exposure_frame()
    frame.loc[2, "settlement_count"] = np.nan
    out = exposure.population_exposure(frame)
    assert bool(out.loc[0, "settlement_data_available"]) is False
    assert pd.isna(out.loc[2, "settlement_count"])


# ---------------------------------------------------------------------------
# Infrastructure dimension
# ---------------------------------------------------------------------------
def test_infrastructure_exposure_adds_densities():
    out = exposure.infrastructure_exposure(exposure_frame())
    assert out.loc[0, "road_density_km_per_km2"] == pytest.approx(120.0 / 3100.0)
    assert out.loc[0, "building_density_per_km2"] == pytest.approx(400.0 / 3100.0)
    assert out.loc[0, "built_fraction"] == pytest.approx(1.2 / 3100.0)
    assert bool(out.loc[0, "infrastructure_detail_available"]) is True


def test_infrastructure_missing_column_raises():
    frame = exposure_frame().drop(columns=["road_km_total"])
    with pytest.raises(exposure.MissingColumnsError) as excinfo:
        exposure.infrastructure_exposure(frame)
    assert "road_km_total" in str(excinfo.value)


def test_infrastructure_footprint_larger_than_cell_raises():
    frame = exposure_frame()
    frame.loc[0, "building_footprint_km2"] = 4_000.0
    with pytest.raises(exposure.InconsistentDataError):
        exposure.infrastructure_exposure(frame)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def test_dimension_availability_reports_missing_columns():
    frame = exposure_frame().drop(columns=["critical_count_total"])
    report = exposure.dimension_availability(frame)
    assert report["environmental"]["status"] == config.STATUS_AVAILABLE
    assert report["population"]["status"] == config.STATUS_AVAILABLE
    assert report["infrastructure"]["status"] == config.STATUS_UNAVAILABLE
    assert "critical_count_total" in report["infrastructure"]["missing"]


def test_prepare_exposure_returns_explicit_unavailable_state():
    frame = exposure_frame().drop(columns=["road_km_total"])
    result = exposure.prepare_exposure(frame)
    assert result.status == config.STATUS_UNAVAILABLE
    assert result.frame is None
    assert result.unavailable is True
    assert "road_km_total" in result.message


def test_prepare_exposure_returns_all_dimensions_when_complete():
    result = exposure.prepare_exposure(exposure_frame())
    assert result.status == config.STATUS_AVAILABLE
    assert result.frame is not None
    assert {"fuel_area_km2", "pop_density_per_km2", "road_density_km_per_km2"} <= set(
        result.frame.columns
    )


def test_prepare_exposure_none_is_unavailable():
    result = exposure.prepare_exposure(None)
    assert result.status == config.STATUS_UNAVAILABLE
    assert result.frame is None


def test_merge_risk_and_exposure_keeps_rows_and_flags_outside():
    prepared = exposure.prepare_exposure(exposure_frame()).frame
    scores = pd.DataFrame(
        {
            "lat_cell": [36.0, 36.5, 35.0],
            "lon_cell": [3.125, 3.75, 2.5],
            "probability": [0.4, np.nan, 0.01],
            "risk": ["HIGH", "OUTSIDE", "LOW"],
            "in_model_area": [True, False, True],
        }
    )
    merged = exposure.merge_risk_and_exposure(prepared, scores)
    assert len(merged) == len(prepared)
    assert merged["outside_model_area"].tolist() == [False, True, False]
    # Outside cells keep their missing probability; nothing is filled.
    assert pd.isna(merged.loc[1, "probability"])


def test_merge_risk_duplicate_cells_raise():
    prepared = exposure.prepare_exposure(exposure_frame()).frame
    duplicated = pd.DataFrame(
        {
            "lat_cell": [36.0, 36.0],
            "lon_cell": [3.125, 3.125],
            "probability": [0.1, 0.2],
            "risk": ["LOW", "LOW"],
            "in_model_area": [True, True],
        }
    )
    with pytest.raises(exposure.InconsistentDataError):
        exposure.merge_risk_and_exposure(prepared, duplicated)


def test_merge_risk_missing_score_column_raises():
    prepared = exposure.prepare_exposure(exposure_frame()).frame
    scores = pd.DataFrame(
        {"lat_cell": [36.0], "lon_cell": [3.125], "risk": ["LOW"]}
    )
    with pytest.raises(exposure.MissingColumnsError):
        exposure.merge_risk_and_exposure(prepared, scores)
