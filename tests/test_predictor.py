"""Tests for the prediction-first UI helpers and the landing page.

The helper tests exercise pure logic (no Streamlit); the AppTest smoke test
executes the real ``app.py`` script end to end.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core  # noqa: E402
from dashboard.config import LAT_MAX, MODEL_FEATURES  # noqa: E402
from dashboard.predictor import (  # noqa: E402
    PRESET_ORDER,
    PRESETS,
    baseline_comparison,
    build_feature_row,
    changed_features,
    compare_presets,
    condition_summary,
    feature_label,
    feature_sensitivity,
    format_pp,
    guidance_for,
    prediction_payload,
    read_share_params,
    risk_narrative,
    score_conditions,
    share_query,
    share_url,
)

BUNDLE = app_core.load_model_bundle()
CARD = BUNDLE.card or {}


def conditions(**overrides):
    values = dict(
        lat=36.5,
        lon=3.125,
        when="2023-08-15",
        t2m_max=43.5,
        rh2m=15.0,
        ws2m=8.5,
        precip=0.0,
        gwetroot=0.12,
        dry_days=42.0,
    )
    values.update(overrides)
    return values


def test_build_feature_row_matches_model_contract():
    row = build_feature_row(**conditions(), card=CARD)

    assert list(row) == list(CARD["feature_names"])
    assert len(row) == len(MODEL_FEATURES)
    assert row["lat_cell"] == pytest.approx(36.5)
    assert row["lon_cell"] == pytest.approx(3.125)
    assert row["T2M_MAX"] == pytest.approx(43.5)
    # Persistence assumption: today's rain repeated over the window.
    assert row["rain_7d"] == pytest.approx(0.0)
    assert row["dry_days"] == pytest.approx(42.0)
    assert np.isfinite(list(row.values())).all()


def test_build_feature_row_applies_antecedent_overrides():
    row = build_feature_row(
        **conditions(precip=2.0),
        card=CARD,
        antecedent={"rain_30d": 120.0, "tmax_30d": 33.0},
    )
    assert row["rain_30d"] == pytest.approx(120.0)
    assert row["tmax_30d"] == pytest.approx(33.0)
    assert row["rain_7d"] == pytest.approx(14.0)  # persistence untouched


def test_score_conditions_returns_valid_band():
    if BUNDLE.status != "REAL":
        pytest.skip("trained model artifact not available")

    row = build_feature_row(**conditions(), card=CARD)
    prediction = score_conditions(BUNDLE, row)

    assert prediction.source == "REAL"
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.risk_level in {"Low", "Moderate", "High", "Very High"}
    assert 0.0 <= prediction.risk_index <= 100.0
    assert prediction.raw_prediction in {0, 1}


def test_heatwave_scores_higher_than_wet_winter():
    if BUNDLE.status != "REAL":
        pytest.skip("trained model artifact not available")

    hot = score_conditions(BUNDLE, build_feature_row(**conditions(), card=CARD))
    cold = score_conditions(
        BUNDLE,
        build_feature_row(
            **conditions(
                t2m_max=14.0,
                rh2m=84.0,
                ws2m=5.0,
                precip=16.5,
                gwetroot=0.86,
                dry_days=0.0,
                when="2023-01-15",
            ),
            card=CARD,
        ),
    )
    assert hot.probability > cold.probability


def test_feature_sensitivity_is_sorted_and_explained():
    if BUNDLE.status != "REAL":
        pytest.skip("trained model artifact not available")

    row = build_feature_row(**conditions(), card=CARD)
    table = feature_sensitivity(BUNDLE, row, top_n=6)

    assert not table.empty
    assert set(table.columns) == {
        "feature",
        "label",
        "value",
        "reference",
        "delta_pp",
        "direction",
    }
    assert len(table) <= 6
    assert table["delta_pp"].abs().is_monotonic_decreasing
    assert set(table["direction"]).issubset({"increases", "reduces"})
    # A fire-suppressing reference should not raise the score for a heatwave.
    assert (table["delta_pp"] > 0).any()


def test_guidance_covers_every_band():
    for label in ("LOW", "MODERATE", "HIGH", "VERY HIGH"):
        assert guidance_for(label)
        assert risk_narrative(label)
    assert guidance_for("unknown") == guidance_for("MODERATE")


def test_presets_are_complete_and_ordered():
    from dashboard.predictor import PRESET_ORDER

    assert PRESET_ORDER[0] == "Summer heatwave"
    assert set(PRESET_ORDER) == set(PRESETS)
    required = {"T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT", "dry_days"}
    for name, preset in PRESETS.items():
        assert required.issubset(preset), name


def test_baseline_comparison_and_formatting():
    comparison = baseline_comparison(0.10, 0.025)
    assert comparison["multiple"] == pytest.approx(4.0)
    assert comparison["delta_pp"] == pytest.approx(7.5)
    assert format_pp(1.234) == "+1.23 pp"
    assert format_pp(-0.5) == "-0.50 pp"
    assert format_pp(float("nan")) == "—"


def test_condition_summary_reads_engineered_rain():
    row = build_feature_row(**conditions(precip=1.0), card=CARD)
    summary = condition_summary(row)
    assert summary["max_temp"] == pytest.approx(43.5)
    assert summary["rain_7d"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------

def test_compare_presets_orders_fire_weather():
    if BUNDLE.status != "REAL":
        pytest.skip("trained model artifact not available")

    table = compare_presets(
        BUNDLE,
        36.5,
        3.125,
        "2023-08-15",
        card=CARD,
    )

    assert list(table["scenario"]) == PRESET_ORDER
    assert len(table) == len(PRESET_ORDER)
    assert set(table["risk_level"]).issubset(
        {"Low", "Moderate", "High", "Very High"}
    )
    scores = table.set_index("scenario")["percent"]
    assert scores["Summer heatwave"] > scores["Winter wet"]
    assert (table["multiple"] >= 0).all()


# ---------------------------------------------------------------------------
# Shareable links
# ---------------------------------------------------------------------------

def test_share_query_round_trip():
    params = share_query(
        lat=36.5,
        lon=3.125,
        when="2023-08-15",
        scenario="Summer heatwave",
    )

    assert params["lat"] == "36.5"
    assert params["date"] == "2023-08-15"
    assert params["run"] == "1"

    restored = read_share_params(params)
    assert restored["lat"] == pytest.approx(36.5)
    assert restored["lon"] == pytest.approx(3.125)
    assert restored["date"].isoformat() == "2023-08-15"
    assert restored["scenario"] == "Summer heatwave"
    assert restored["run"] is True
    assert restored["from_link"] is True


def test_read_share_params_clamps_and_ignores_garbage():
    fallback = dt.date(2024, 7, 1)
    parsed = read_share_params(
        {
            "lat": "999",
            "lon": "not-a-number",
            "date": "not-a-date",
            "scenario": "Monsoon",
            "run": "0",
        },
        default_date=fallback,
    )

    assert parsed["lat"] == pytest.approx(LAT_MAX)
    assert parsed["lon"] == pytest.approx(3.125)
    assert parsed["date"] == fallback
    assert parsed["scenario"] is None
    assert parsed["run"] is False


def test_share_url_encodes_the_query():
    url = share_url(
        "http://localhost:8501/",
        {"lat": "36.5", "scenario": "Summer heatwave"},
    )
    assert url.startswith("http://localhost:8501/?")
    assert "scenario=Summer+heatwave" in url


# ---------------------------------------------------------------------------
# Staleness and export
# ---------------------------------------------------------------------------

def test_changed_features_detects_only_real_edits():
    row = build_feature_row(**conditions(), card=CARD)
    assert changed_features(row, dict(row)) == []

    edited = dict(row)
    edited["T2M_MAX"] += 0.5
    edited["RH2M"] -= 3.0
    changed = changed_features(edited, row)

    assert set(changed) == {"T2M_MAX", "RH2M"}
    assert feature_label("T2M_MAX") == "Daily max temperature"


def test_prediction_payload_is_json_serialisable():
    if BUNDLE.status != "REAL":
        pytest.skip("trained model artifact not available")

    row = build_feature_row(**conditions(), card=CARD)
    prediction = score_conditions(BUNDLE, row)
    drivers = feature_sensitivity(BUNDLE, row, top_n=3)
    payload = prediction_payload(
        prediction=prediction,
        features=row,
        lat=36.5,
        lon=3.125,
        when="2023-08-15",
        scenario="Summer heatwave",
        model_version=CARD.get("model_version", "unknown"),
        base_rate=CARD["metrics"]["base_rate"],
        sensitivity=drivers,
    )

    encoded = json.dumps(payload)
    assert json.loads(encoded)["result"]["probability"] == pytest.approx(
        prediction.probability, abs=1e-6
    )
    assert payload["location"]["grid_cell"]["lat_cell"] == pytest.approx(36.5)
    assert payload["model"]["horizon_days"] == 3
    assert len(payload["drivers"]) == len(drivers)
    assert payload["disclaimer"]


# ---------------------------------------------------------------------------
# Landing page smoke test
# ---------------------------------------------------------------------------
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = ROOT / "app.py"


def run_page() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=120)
    app.run()
    return app


def test_landing_page_is_prediction_first():
    app = run_page()

    assert not app.exception, app.exception
    labels = [button.label for button in app.button]
    assert any("Predict fire risk" in label for label in labels)

    rendered = "\n".join(item.value for item in app.markdown)
    assert "Will a fire start here?" in rendered
    assert "RISK INDEX" in rendered or "Prediction" in rendered
    assert "Where and when?" in rendered
    assert "What is the fire weather?" in rendered


def test_clicking_predict_shows_probability_and_guidance():
    app = run_page()

    button = next(b for b in app.button if "Predict fire risk" in b.label)
    assert not button.disabled
    button.click().run()

    assert not app.exception, app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "probability of an observed fire" in rendered
    assert "Recommended actions" in rendered
    assert "Calibrated model v" in rendered or "Demo heuristic" in rendered


def test_landing_page_compares_scenarios():
    app = run_page()

    assert not app.exception, app.exception
    labels = [expander.label for expander in app.expander]
    assert any("Compare the four scenarios" in label for label in labels)
    captions = " ".join(item.value for item in app.caption)
    assert "isolates how" in captions
    # The comparison table must expose all four presets at once.
    tables = [frame.value for frame in app.dataframe]
    scenario_tables = [table for table in tables if "Scenario" in list(table.columns)]
    assert scenario_tables
    assert scenario_tables[0]["Scenario"].tolist() == list(PRESET_ORDER)


def test_result_stays_stable_and_flags_changed_inputs():
    import re

    app = run_page()
    next(b for b in app.button if "Predict fire risk" in b.label).click().run()
    assert not app.exception, app.exception

    rendered = "\n".join(item.value for item in app.markdown)
    match = re.search(r'class="fwx-prob"[^>]*>([\d.]+)%', rendered)
    assert match, "probability headline not found"
    original = match.group(1)

    # Edit one weather input without pressing Predict again.
    app.number_input[2].set_value(45.0).run()
    assert not app.exception, app.exception

    rendered = "\n".join(item.value for item in app.markdown)
    warnings = " ".join(item.value for item in app.warning)
    assert "Inputs changed since this prediction" in warnings
    assert "Predict fire risk" in warnings
    # The displayed score must not silently drift with the edited input.
    match = re.search(r'class="fwx-prob"[^>]*>([\d.]+)%', rendered)
    assert match and match.group(1) == original
