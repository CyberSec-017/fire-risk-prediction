"""Tests for the prediction-first UI helpers and the landing page.

The helper tests exercise pure logic (no Streamlit); the AppTest smoke test
executes the real ``app.py`` script end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_core  # noqa: E402
from dashboard.config import MODEL_FEATURES  # noqa: E402
from dashboard.predictor import (  # noqa: E402
    PRESETS,
    baseline_comparison,
    build_feature_row,
    condition_summary,
    feature_sensitivity,
    format_pp,
    guidance_for,
    risk_narrative,
    score_conditions,
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
