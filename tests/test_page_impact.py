"""AppTest coverage for the Wildfire Impact Assessment page (Module 5)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = ROOT / "pages" / "6_Impact_Assessment.py"
FIXTURE = ROOT / "tests" / "fixtures" / "impact"

TAB_LABELS = [
    "Overview",
    "Environment",
    "Population",
    "Infrastructure",
    "Priority Matrix",
    "Report",
]


def run_page(data_dir: Path | None = None) -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=120)
    if data_dir is not None:
        app.session_state["impact_data_dir"] = str(data_dir)
    app.run()
    return app


def rendered_markdown(app: AppTest) -> str:
    return "\n".join(item.value for item in app.markdown)


def line_with(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if prefix in line:
            return line.strip()
    raise AssertionError(f"Could not find {prefix!r} in page output.")


def test_page_loads_when_impact_data_is_absent():
    app = run_page()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == TAB_LABELS
    body = rendered_markdown(app)
    assert "Not available" in body
    assert "Dataset Not Found" in body
    assert "Impact exposure layers" in body
    # The missing-data card must reference the impact builder, not the
    # fire-panel preprocessing script.
    assert "python scripts/build_impact_layers.py" in body
    assert "python src/build_panel.py" not in body
    # No fabricated KPI widgets when the impact artifacts are missing.
    assert len(app.metric) == 0
    # The page still names the four evidence layers it would report.
    for phrase in (
        "Predicted fire risk",
        "Potential exposure",
        "Observed fire detections",
        "Confirmed burned area",
    ):
        assert phrase in body


def test_page_loads_with_fixture_data():
    app = run_page(FIXTURE)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == TAB_LABELS
    body = rendered_markdown(app)

    for phrase in (
        "Predicted fire risk",
        "Potential exposure",
        "Observed fire detections",
        "Confirmed burned area",
        "Not available (phase 2)",
        "Impact Exposure Index",
        "Risk × exposure priority matrix",
    ):
        assert phrase in body, phrase

    assert "Dataset Not Found" not in body
    # HTML, CSV and JSON downloads are offered once the manifest exists.
    assert len(app.download_button) == 3


def test_date_change_alters_risk_but_not_exposure():
    app = run_page(FIXTURE)
    assert not app.exception

    app.date_input[0].set_value(dt.date(2019, 7, 1)).run()
    assert not app.exception
    first = rendered_markdown(app)
    first_risk = line_with(first, "Predicted risk — mean probability:")
    first_population = line_with(
        first, "Potential exposure — population in cells:"
    )

    app.date_input[0].set_value(dt.date(2019, 12, 1)).run()
    assert not app.exception
    second = rendered_markdown(app)
    second_risk = line_with(second, "Predicted risk — mean probability:")
    second_population = line_with(
        second, "Potential exposure — population in cells:"
    )

    assert first_risk != second_risk
    assert first_population == second_population


def test_observed_detections_are_not_labelled_burned_area():
    app = run_page(FIXTURE)
    body = rendered_markdown(app)

    assert "Observed detections — cells with detection:" in body
    assert "Confirmed burned area — Not available (phase 2)" in body
    assert "confirmed damage" not in body.lower()
    assert "people affected" not in body.lower()
