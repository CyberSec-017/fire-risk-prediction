"""Streamlit startup smoke test.

This test executes ``app.py`` through Streamlit's official test harness and
asserts that the single-page predictor renders without an uncaught exception.
It does not make any NASA POWER network calls: the default mode is manual
weather, which scores instantly from local inputs.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_single_page_renders_without_exception():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.run()

    assert not app.exception
    # One sidebar control remains: the prediction-mode toggle. The old
    # multi-page navigation radio is gone.
    assert len(app.sidebar.radio) == 1
    assert app.sidebar.radio[0].value == "Manual weather"


def test_app_manual_mode_renders_a_risk_card():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.run()

    assert not app.exception
    # Manual mode scores automatically on load, so a result card should be
    # present (REAL or DEMO depending on whether the artifact exists).
    assert any('class="risk-card"' in block.value for block in app.markdown)
