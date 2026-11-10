"""Regression tests for the page 2 geographic sub-region filter.

The filter must apply the latitude and longitude sliders with ``&``; a bare
``A & B if cond else C`` expression drops the latitude mask whenever the
``lon_sub`` column is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


PAGE = next((ROOT / "pages").glob("2_*Geospatial_Monitoring.py"))


def run_page() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=180)
    app.run()
    assert not app.exception
    return app


def slider(app: AppTest, prefix: str):
    return next(item for item in app.slider if item.label.startswith(prefix))


def displayed_rows(app: AppTest) -> pd.DataFrame:
    assert app.dataframe, "the filtered dataset table was not rendered"
    return app.dataframe[0].value


def test_latitude_slider_keeps_only_cells_inside_range():
    app = run_page()
    slider(app, "Latitude Range").set_value((35.5, 36.5))
    app.run()
    assert not app.exception

    frame = displayed_rows(app)
    assert len(frame) > 0
    assert frame["lat_cell"].between(35.5, 36.5).all()
    assert frame["lat_cell"].nunique() > 1


def test_longitude_slider_keeps_only_cells_inside_range():
    app = run_page()
    slider(app, "Longitude Range").set_value((2.0, 4.0))
    app.run()
    assert not app.exception

    frame = displayed_rows(app)
    assert len(frame) > 0
    assert frame["lon_cell"].between(2.0, 4.0).all()
    assert frame["lon_cell"].nunique() > 1
