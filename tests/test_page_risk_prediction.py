"""Smoke tests for the Risk Prediction page (Module 4).

Uses Streamlit's AppTest harness to execute the real page script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


PAGE = ROOT / "pages" / "4_🎯_Risk_Prediction.py"


def run_page() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    return app


def test_risk_page_loads():
    app = run_page()

    assert not app.exception
    labels = [button.label for button in app.button]
    assert "Evaluate Wildfire Risk" in labels
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Grid cell used: lat 36.500, lon 3.125" in rendered
    assert "MODEL STATE:" in rendered
    assert "20 features" in rendered
    assert "models/fire_model.joblib" in rendered


def test_clicking_evaluate_shows_risk_result():
    app = run_page()

    buttons = [b for b in app.button if b.label == "Evaluate Wildfire Risk"]
    assert buttons, "Evaluate button not found"
    evaluate = buttons[0]
    assert not evaluate.disabled

    evaluate.click().run()

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Risk Assessment" in rendered
    assert (
        "Calibrated 3-day probability" in rendered
        or "Demo heuristic, not a trained model" in rendered
    )
    assert "20 features" in rendered
    assert "models/fire_model.joblib" in rendered


def test_page_does_not_mention_obsolete_contract_text():
    app = run_page()

    rendered = "\n".join(item.value for item in app.markdown)
    # Built from fragments so this regression guard does not itself put the
    # obsolete strings back into the repository.
    obsolete = [
        "19 " + "predictors",
        "(1, " + "19)",
        "Member " + "4",
        "fire_risk_" + "model.pkl",
    ]
    for token in obsolete:
        assert token not in rendered
