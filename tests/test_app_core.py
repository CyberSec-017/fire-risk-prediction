"""Unit tests for the dashboard's model/feature logic.

Run with:
    python -m pytest -q

These tests intentionally do **not** import Streamlit.  The UI is covered by
the smoke test in the final verification step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_core as core  # noqa: E402


CARD = json.loads(core.MODEL_CARD_PATH.read_text(encoding="utf-8"))


def manual_row(**overrides):
    values = {
        "lat": 36.7538,
        "lon": 3.0588,
        "when": "2023-07-15",
        "t2m": 30.0,
        "t2m_max": 40.0,
        "rh2m": 25.0,
        "ws2m": 6.0,
        "precip": 0.0,
        "gwetroot": 0.15,
        "dry_days": 12.0,
        "card": CARD,
    }
    values.update(overrides)
    lat = values.pop("lat")
    lon = values.pop("lon")
    when = values.pop("when")
    return core.build_manual_features(lat, lon, when, **values)


# ---------------------------------------------------------------------------
# Contract and artifact loading
# ---------------------------------------------------------------------------

def test_feature_hash_matches_model_card():
    assert core.feature_hash(CARD["feature_names"]) == CARD["feature_hash"]


def test_load_model_bundle_real():
    bundle = core.load_model_bundle()

    assert bundle.status == "REAL"
    assert bundle.model is not None
    assert bundle.card is not None
    assert bundle.card["n_features"] == 20
    assert bundle.error is None


def test_missing_model_is_demo(tmp_path):
    bundle = core.load_model_bundle(
        tmp_path / "missing.joblib", core.MODEL_CARD_PATH
    )

    assert bundle.status == "DEMO"
    assert bundle.model is None
    assert bundle.card is not None


def test_missing_model_and_card_is_still_demo(tmp_path):
    bundle = core.load_model_bundle(
        tmp_path / "missing.joblib", tmp_path / "missing.json"
    )

    assert bundle.status == "DEMO"
    assert bundle.model is None
    assert bundle.card is None

    row = manual_row(card=None)
    assert list(row.columns) == core.FALLBACK_FEATURE_NAMES


def test_corrupt_model_is_error(tmp_path):
    bad_model = tmp_path / "bad.joblib"
    bad_model.write_text("this is not a joblib artifact", encoding="utf-8")

    bundle = core.load_model_bundle(bad_model, core.MODEL_CARD_PATH)

    assert bundle.status == "ERROR"
    assert bundle.model is None
    assert "load" in (bundle.error or "").lower()


def test_missing_card_is_error_when_model_exists(tmp_path):
    bundle = core.load_model_bundle(
        core.MODEL_PATH, tmp_path / "missing.json"
    )

    assert bundle.status == "ERROR"
    assert "card" in (bundle.error or "").lower()


def test_card_hash_mismatch_is_error(tmp_path):
    card = dict(CARD)
    card["feature_hash"] = "0" * 64
    bad_card = tmp_path / "model_card.json"
    bad_card.write_text(json.dumps(card), encoding="utf-8")

    bundle = core.load_model_bundle(core.MODEL_PATH, bad_card)

    assert bundle.status == "ERROR"
    assert "hash" in (bundle.error or "").lower()


def test_model_feature_order_mismatch_is_error(tmp_path):
    card = dict(CARD)
    card["feature_names"] = list(reversed(CARD["feature_names"]))
    card["feature_hash"] = core.feature_hash(card["feature_names"])
    bad_card = tmp_path / "model_card.json"
    bad_card.write_text(json.dumps(card), encoding="utf-8")

    bundle = core.load_model_bundle(core.MODEL_PATH, bad_card)

    assert bundle.status == "ERROR"
    assert "feature" in (bundle.error or "").lower()


# ---------------------------------------------------------------------------
# Grid, date and feature construction
# ---------------------------------------------------------------------------

def test_snap_to_grid_uses_power_cell_size():
    lat_cell, lon_cell = core.snap_to_grid(36.7538, 3.0588)

    assert lat_cell == pytest.approx(37.0)
    assert lon_cell == pytest.approx(3.125)


def test_doy_features_match_pipeline_formula():
    when = pd.Timestamp("2023-07-15")
    doy = when.day_of_year
    sin_value, cos_value = core.doy_features(when)

    assert sin_value == pytest.approx(np.sin(2 * np.pi * doy / 365.25))
    assert cos_value == pytest.approx(np.cos(2 * np.pi * doy / 365.25))


def test_build_manual_features_follows_card_order_and_derivations():
    row = manual_row()

    assert list(row.columns) == CARD["feature_names"]
    assert row.loc[0, "lat_cell"] == pytest.approx(37.0)
    assert row.loc[0, "lon_cell"] == pytest.approx(3.125)
    assert row.loc[0, "rain_7d"] == 0.0
    assert row.loc[0, "rain_14d"] == 0.0
    assert row.loc[0, "rain_30d"] == 0.0
    assert row.loc[0, "tmax_14d"] == 40.0
    assert row.loc[0, "rhmin_7d"] == 25.0
    assert row.loc[0, "dry_days"] == 12.0
    assert row.loc[0, "doy_sin"] == pytest.approx(-0.228058, abs=1e-5)


def test_manual_persistence_assumption_for_rain():
    row = manual_row(precip=2.0, dry_days=0.0)

    assert row.loc[0, "rain_7d"] == 14.0
    assert row.loc[0, "rain_14d"] == 28.0
    assert row.loc[0, "rain_30d"] == 60.0


def test_build_manual_features_rejects_unknown_card_feature():
    card = dict(CARD)
    card["feature_names"] = list(CARD["feature_names"]) + ["mystery"]
    card["n_features"] = len(card["feature_names"])
    card["feature_hash"] = core.feature_hash(card["feature_names"])

    with pytest.raises(core.FeatureContractError):
        manual_row(card=card)


def test_build_manual_features_rejects_non_finite_input():
    with pytest.raises(core.FeatureBuildError):
        manual_row(t2m=float("nan"))


def test_check_training_area_flags_out_of_box_and_years():
    inside = core.check_training_area(36.7538, 3.0588, "2023-07-15")
    outside = core.check_training_area(48.8566, 2.3522, "2023-07-15")
    old_date = core.check_training_area(36.7538, 3.0588, "2026-07-15")

    assert inside["in_box"] and inside["in_years"]
    assert inside["warnings"] == []
    assert not outside["in_box"]
    assert outside["warnings"]
    assert not old_date["in_years"]
    assert not core.point_in_training_box(48.8566, 2.3522, "2023-07-15")
    assert not core.point_in_training_box(36.7538, 3.0588, "2026-07-15")


# ---------------------------------------------------------------------------
# NASA POWER parsing
# ---------------------------------------------------------------------------

def test_parse_power_point_json_handles_dates_and_sentinels():
    payload = {
        "properties": {
            "parameter": {
                "T2M": {"20240101": 10.0, "20240102": -999.0},
                "T2M_MAX": {"20240101": 18.0, "20240102": 20.0},
            }
        }
    }

    frame = core.parse_power_point_json(payload, ["T2M", "T2M_MAX"])

    assert len(frame) == 2
    assert frame.loc[0, "T2M"] == 10.0
    assert np.isnan(frame.loc[1, "T2M"])
    assert frame.loc[1, "T2M_MAX"] == 20.0


def test_parse_power_point_json_rejects_missing_parameter():
    payload = {
        "properties": {
            "parameter": {
                "T2M": {"20240101": 10.0},
            }
        }
    }

    with pytest.raises(core.PowerFetchError):
        core.parse_power_point_json(payload, ["T2M", "RH2M"])


# ---------------------------------------------------------------------------
# Risk bands, demo scoring and model scoring
# ---------------------------------------------------------------------------

def test_risk_level_boundaries_follow_card_bins():
    assert core.risk_level(0.0, CARD) == "LOW"
    assert core.risk_level(0.005, CARD) == "LOW"
    assert core.risk_level(0.0051, CARD) == "MODERATE"
    assert core.risk_level(0.02, CARD) == "MODERATE"
    assert core.risk_level(0.0201, CARD) == "HIGH"
    assert core.risk_level(0.06, CARD) == "HIGH"
    assert core.risk_level(0.0601, CARD) == "VERY HIGH"
    assert core.risk_level(1.0, CARD) == "VERY HIGH"


def test_risk_index_spans_each_band():
    assert core.risk_index(0.0, CARD) == pytest.approx(0.0)
    assert core.risk_index(0.005, CARD) == pytest.approx(25.0)
    assert core.risk_index(0.02, CARD) == pytest.approx(50.0)
    assert core.risk_index(0.06, CARD) == pytest.approx(75.0)
    assert core.risk_index(1.0, CARD) == pytest.approx(100.0)


def test_demo_probability_is_bounded_and_directional():
    hot = manual_row(
        t2m=35.0, t2m_max=48.0, rh2m=10.0, ws2m=12.0,
        precip=0.0, gwetroot=0.10, dry_days=40.0,
    )
    cool = manual_row(
        t2m=10.0, t2m_max=15.0, rh2m=90.0, ws2m=1.0,
        precip=10.0, gwetroot=0.80, dry_days=0.0,
    )

    hot_probability = core.demo_probability(hot, CARD)
    cool_probability = core.demo_probability(cool, CARD)

    assert 0.0 <= hot_probability <= 1.0
    assert 0.0 <= cool_probability <= 1.0
    assert hot_probability > cool_probability


def test_real_model_prediction_is_in_range():
    bundle = core.load_model_bundle()
    row = manual_row(card=bundle.card)

    probability = core.predict_probability(bundle, row)
    label = core.risk_level(probability, bundle.card)

    assert 0.0 <= probability <= 1.0
    assert label in bundle.card["risk_labels"]
    assert core.score_features(bundle, row) == pytest.approx((probability, "REAL"))


def test_error_bundle_refuses_to_score():
    bundle = core.ModelBundle(
        status="ERROR",
        model=None,
        card=CARD,
        error="bad artifact",
        model_path=Path("missing"),
        card_path=Path("missing"),
    )

    with pytest.raises(core.ModelNotReadyError):
        core.score_features(bundle, manual_row())


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def test_perturb_feature_row_updates_derived_windows():
    base = manual_row()

    tmax_row = core.perturb_feature_row(base, "T2M_MAX", 50.0)
    assert tmax_row.loc[0, "T2M_MAX"] == 50.0
    assert tmax_row.loc[0, "tmax_7d"] == 50.0
    assert tmax_row.loc[0, "tmax_30d"] == 50.0
    assert list(tmax_row.columns) == CARD["feature_names"]

    rain_row = core.perturb_feature_row(base, "PRECTOTCORR", 2.0)
    assert rain_row.loc[0, "rain_7d"] == 14.0
    assert rain_row.loc[0, "rain_30d"] == 60.0

    rh_row = core.perturb_feature_row(base, "RH2M", 80.0)
    assert rh_row.loc[0, "rhmin_14d"] == 80.0


def test_sensitivity_table_scores_each_value():
    bundle = core.load_model_bundle()
    base = manual_row(card=bundle.card)

    table = core.sensitivity_table(bundle, base, "T2M_MAX", [20.0, 30.0, 40.0])

    assert len(table) == 3
    assert list(table.columns) == ["value", "probability", "risk", "risk_display"]
    assert table["probability"].between(0.0, 1.0).all()
    assert set(table["risk"]).issubset(set(bundle.card["risk_labels"]))


# ---------------------------------------------------------------------------
# Panel analytics
# ---------------------------------------------------------------------------

def test_panel_summary_and_aggregations():
    panel = pd.DataFrame({
        "lat_cell": [32.0, 32.0, 32.5],
        "lon_cell": [0.0, 0.0, 0.0],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01"]),
        "burned": [0, 1, 0],
    })

    summary = core.panel_summary(panel)
    assert summary["rows"] == 3
    assert summary["cells"] == 2
    assert summary["fire_days"] == 1
    assert summary["fire_rate"] == pytest.approx(1 / 3)
    assert summary["years"] == 1

    stats = core.cell_fire_stats(panel)
    assert len(stats) == 2
    top = stats.iloc[0]
    assert top["lat_cell"] == 32.0
    assert top["fire_days"] == 1

    monthly = core.monthly_fire_stats(panel)
    assert len(monthly) == 2
    assert list(monthly["month_name"]) == ["Jan", "Feb"]

    yearly = core.yearly_fire_stats(panel)
    assert len(yearly) == 1
    assert yearly.iloc[0]["fire_days"] == 1


def test_median_feature_row_preserves_contract_order():
    panel = pd.DataFrame([
        {name: float(index) for index, name in enumerate(CARD["feature_names"])}
    ])

    row = core.median_feature_row(panel, CARD)

    assert list(row.columns) == CARD["feature_names"]
    assert row.loc[0, CARD["feature_names"][0]] == 0.0


def test_format_helpers():
    assert core.format_probability(0.0) == "0.00%"
    assert core.format_probability(0.00001) == "<0.01%"
    assert core.format_probability(0.005) == "0.50%"
    assert core.format_probability(0.1234) == "12.3%"
    assert core.format_percent(0.0237) == "2.4%"
    assert core.format_number(1234.5, 1) == "1,234.5"
    assert core.risk_display("VERY HIGH") == "Extreme"
    assert core.risk_color("LOW") == "#2ecc71"
