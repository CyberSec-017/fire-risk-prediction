"""Unit tests for the Impact Exposure Index, tiers and priority matrix."""

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
from impact import severity  # noqa: E402


def raw_frame() -> pd.DataFrame:
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
            "road_km_total": [120.0, 0.0, 20.0],
            "building_count": [400.0, 0.0, 25.0],
            "building_footprint_km2": [1.2, 0.0, 0.05],
            "critical_count_total": [5.0, 0.0, 1.0],
        }
    )


def prepared_frame() -> pd.DataFrame:
    result = exposure.prepare_exposure(raw_frame())
    assert result.available
    return result.frame


def test_weights_sum_to_one():
    assert sum(severity.IEI_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(severity.INFRA_SUBWEIGHTS.values()) == pytest.approx(1.0)


def test_compute_index_matches_formula_on_one_row():
    one_row = raw_frame().iloc[[0]].copy()
    one_row["lc_tree_km2"] = 800.0
    one_row["lc_shrub_km2"] = 300.0
    one_row["lc_grass_km2"] = 200.0
    one_row["lc_crop_km2"] = 250.0
    one_row["lc_bare_km2"] = 1200.0
    one_row["lc_built_km2"] = 150.0
    one_row["lc_water_km2"] = 100.0
    one_row["lc_other_km2"] = 100.0
    prepared = exposure.prepare_exposure(one_row).frame

    out = severity.compute_index(prepared)
    # With one row every p95 equals that row's value, so those norms are 1.
    assert out.loc[0, "fuel_norm"] == pytest.approx(1550.0 / 3100.0)
    assert out.loc[0, "pop_norm"] == pytest.approx(1.0)
    assert out.loc[0, "infra_norm"] == pytest.approx(1.0)
    expected = 0.35 * (1550.0 / 3100.0) + 0.35 + 0.30
    assert out.loc[0, "iei"] == pytest.approx(expected)
    assert out.loc[0, "exposure_tier"] == "Very High"


def test_iei_is_the_weighted_sum_of_its_components():
    out = severity.compute_index(prepared_frame())
    expected = (
        severity.IEI_WEIGHTS["fuel"] * out["fuel_norm"]
        + severity.IEI_WEIGHTS["population"] * out["pop_norm"]
        + severity.IEI_WEIGHTS["infrastructure"] * out["infra_norm"]
    )
    pd.testing.assert_series_equal(out["iei"], expected, check_names=False)

    expected_infra = (
        severity.INFRA_SUBWEIGHTS["road"] * out["road_norm"]
        + severity.INFRA_SUBWEIGHTS["building"] * out["building_norm"]
        + severity.INFRA_SUBWEIGHTS["critical"] * out["critical_norm"]
    )
    np.testing.assert_allclose(out["infra_norm"], expected_infra)


def test_components_are_bounded_and_ordered():
    out = severity.compute_index(prepared_frame())
    for column in (
        "fuel_norm",
        "pop_norm",
        "road_norm",
        "building_norm",
        "critical_norm",
        "infra_norm",
        "iei",
    ):
        assert out[column].between(0.0, 1.0).all(), column


def test_empty_cell_gets_zero_index_but_is_not_missing():
    out = severity.compute_index(prepared_frame())
    row = out[out["cell_id"] == "c2"].iloc[0]
    assert row["fuel_norm"] == pytest.approx(0.0)
    assert row["pop_norm"] == pytest.approx(0.0)
    assert row["infra_norm"] == pytest.approx(0.0)
    assert row["iei"] == pytest.approx(0.0)
    assert row["exposure_tier"] == "Very Low"


def test_tier_boundaries_are_exact_and_left_closed():
    values = [0.0, 0.0999, 0.10, 0.1999, 0.20, 0.3499, 0.35, 0.5499, 0.55, 1.0]
    labels = severity.classify_iei(values).tolist()
    assert labels == [
        "Very Low",
        "Very Low",
        "Low",
        "Low",
        "Moderate",
        "Moderate",
        "High",
        "High",
        "Very High",
        "Very High",
    ]
    assert severity.tier_for_iei(0.10) == "Low"
    assert severity.tier_for_iei(0.20) == "Moderate"
    assert severity.tier_for_iei(0.35) == "High"
    assert severity.tier_for_iei(0.55) == "Very High"


def test_nan_tier_is_explicitly_unavailable():
    assert severity.tier_for_iei(float("nan")) == severity.TIER_UNAVAILABLE
    assert severity.classify_iei([np.nan]).tolist() == [severity.TIER_UNAVAILABLE]


def test_resolve_thresholds_prefers_manifest_then_defaults():
    manifest = {"severity": {"tier_thresholds": [0.05, 0.15, 0.30, 0.50]}}
    assert severity.resolve_thresholds(manifest=manifest) == (0.05, 0.15, 0.30, 0.50)
    assert severity.resolve_thresholds() == severity.DEFAULT_TIER_THRESHOLDS
    assert severity.resolve_thresholds(thresholds=(0.1, 0.2, 0.3, 0.4), manifest=manifest) == (
        0.1,
        0.2,
        0.3,
        0.4,
    )


def test_invalid_thresholds_raise():
    with pytest.raises(severity.SeverityInputError):
        severity.resolve_thresholds(thresholds=(0.2, 0.1, 0.3, 0.4))
    with pytest.raises(severity.SeverityInputError):
        severity.resolve_thresholds(thresholds=(0.1, 0.2, 0.3))
    with pytest.raises(severity.SeverityInputError):
        severity.resolve_thresholds(thresholds=(0.1, 0.2, 0.3, 1.2))


def test_priority_matrix_is_monotonic_and_bounded():
    for band in severity.RISK_BANDS:
        row = [severity.PRIORITY_MATRIX[band][tier] for tier in severity.TIER_LABELS]
        assert row == sorted(row)
        assert all(1 <= value <= 5 for value in row)
    for tier in severity.TIER_LABELS:
        column = [severity.PRIORITY_MATRIX[band][tier] for band in severity.RISK_BANDS]
        assert column == sorted(column)
    assert severity.PRIORITY_MATRIX["LOW"]["Very Low"] == 1
    assert severity.PRIORITY_MATRIX["VERY HIGH"]["Very High"] == 5


def test_priority_for_examples_and_unknowns():
    assert severity.priority_for("Very High", "Very High") == 5
    assert severity.priority_for("MODERATE", "Low") == 2
    assert severity.priority_for("OUTSIDE", "High") is None
    assert severity.priority_for(None, "High") is None
    assert severity.priority_for("HIGH", "Unknown") is None


def test_assign_priority_marks_outside_cells_not_applicable():
    frame = pd.DataFrame(
        {
            "exposure_tier": ["High", "High", "Very Low", "Very Low"],
            "risk": ["HIGH", "OUTSIDE", "LOW", "MODERATE"],
        }
    )
    out = severity.assign_priority(frame)
    assert out.loc[0, "impact_priority"] == 4
    assert out.loc[0, "impact_priority_label"] == "High priority"
    assert pd.isna(out.loc[1, "impact_priority"])
    assert out.loc[1, "impact_priority_label"] == severity.NOT_APPLICABLE
    assert out.loc[2, "impact_priority"] == 1
    assert out.loc[3, "impact_priority"] == 1


def test_assign_priority_requires_tier_column():
    with pytest.raises(severity.SeverityInputError):
        severity.assign_priority(pd.DataFrame({"risk": ["LOW"]}))


def test_compute_index_requires_prepared_columns():
    with pytest.raises(severity.SeverityInputError) as excinfo:
        severity.compute_index(raw_frame())
    assert "fuel_area_km2" in str(excinfo.value)
    assert "never treated as zero" in str(excinfo.value)


def test_zero_population_percentile_raises_instead_of_zeroing():
    frame = prepared_frame()
    for column in ("pop_cell", "pop_5km", "pop_10km", "pop_25km"):
        frame[column] = 0.0
    with pytest.raises(severity.SeverityInputError) as excinfo:
        severity.compute_index(frame)
    assert "95th percentile" in str(excinfo.value)


def test_nan_required_value_raises():
    frame = prepared_frame()
    frame.loc[0, "road_km_total"] = np.nan
    with pytest.raises(severity.SeverityInputError):
        severity.compute_index(frame)


def test_assess_impact_returns_available_result():
    frame = prepared_frame()
    frame["risk"] = ["VERY HIGH", "LOW", "MODERATE"]
    result = severity.assess_impact(frame)
    assert result.available
    assert result.frame is not None
    assert {"iei", "exposure_tier", "impact_priority"} <= set(result.frame.columns)
    assert result.frame["impact_priority"].notna().all()


def test_assess_impact_returns_explicit_unavailable_state():
    frame = prepared_frame().drop(columns=["fuel_area_km2"])
    result = severity.assess_impact(frame)
    assert result.status == config.STATUS_UNAVAILABLE
    assert result.frame is None
    assert "fuel_area_km2" in result.message


def test_assess_impact_without_risk_column_keeps_priority_not_applicable():
    result = severity.assess_impact(prepared_frame())
    assert result.available
    assert result.frame["impact_priority"].isna().all()
    assert (result.frame["impact_priority_label"] == severity.NOT_APPLICABLE).all()


def test_formula_text_documents_the_weights():
    text = severity.formula_text()
    assert "0.35" in text and "0.30" in text and "0.45" in text


def test_priority_matrix_rows_cover_the_full_matrix():
    rows = severity.priority_matrix_rows()
    assert len(rows) == len(severity.RISK_BANDS) * len(severity.TIER_LABELS)
    assert {row["priority"] for row in rows} == {1, 2, 3, 4, 5}
