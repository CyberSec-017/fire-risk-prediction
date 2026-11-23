"""AppTest integration coverage for the interactive EDA section and its navigation."""

from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "app.py"


def _open_eda() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_level"] = "eda"
    app.run()
    assert not app.exception, app.exception
    return app


def test_eda_page_renders_all_visualizations_and_quality_tables():
    app = _open_eda()

    assert len(app.get("plotly_chart")) >= 10
    assert len(app.download_button) == 1
    assert len(app.metric) >= 3
    rendered = " ".join(
        item.value
        for group in (app.markdown, app.html, app.caption, app.info)
        for item in group
    )
    assert "Qualité des données & Analyse exploratoire" in rendered
    assert "Aucun risque prédit" in rendered
    assert "Période analysée" in rendered

    column_sets = [set(table.value.columns) for table in app.dataframe]
    assert any({"start", "end", "outside_study_area"} <= columns for columns in column_sets)
    assert any({"column", "missing_count", "missing_percent"} <= columns for columns in column_sets)
    assert any({"column", "invalid_count", "invalid_percent"} <= columns for columns in column_sets)


def test_date_filter_changes_the_analysed_period_and_row_counts():
    app = _open_eda()
    before = app.caption[0].value

    app.date_input(key="eda_date_range").set_value((date(2021, 1, 1), date(2021, 12, 31))).run()

    assert not app.exception, app.exception
    after = app.caption[0].value
    assert "2021-01-01 → 2021-12-31" in after
    assert before != after
    assert len(app.get("plotly_chart")) >= 10


def test_geographic_filter_narrows_the_dataset_without_breaking_charts():
    app = _open_eda()
    before = app.caption[0].value

    app.slider(key="eda_lat_range").set_value((36.0, 37.5)).run()

    assert not app.exception, app.exception
    assert app.caption[0].value != before
    assert len(app.get("plotly_chart")) >= 10


def test_chart_selector_controls_how_many_visualizations_are_rendered():
    app = _open_eda()
    assert len(app.get("plotly_chart")) >= 10

    app.multiselect(key="eda_charts").set_value(["monthly_detections"]).run()

    assert not app.exception, app.exception
    assert len(app.get("plotly_chart")) == 1


def test_variable_selector_and_csv_export_are_wired():
    app = _open_eda()

    app.selectbox(key="eda_variable").set_value("GWETROOT").run()

    assert not app.exception, app.exception
    charts = app.get("plotly_chart")
    assert len(charts) >= 10
    download = app.download_button[0]
    assert download.proto.url.endswith(".csv")
    assert "Télécharger" in download.proto.label


def test_navigation_from_landing_and_cockpit_into_eda_and_back():
    landing = AppTest.from_file(str(APP_PATH), default_timeout=180)
    landing.run()
    landing.button(key="btn_explore_eda").click().run()
    assert not landing.exception, landing.exception
    assert landing.session_state["app_level"] == "eda"
    assert len(landing.get("plotly_chart")) >= 10

    landing.button(key="eda_back_to_dashboard").click().run()
    assert not landing.exception, landing.exception
    assert landing.session_state["app_level"] == "dashboard"

    cockpit = AppTest.from_file(str(APP_PATH), default_timeout=180)
    cockpit.session_state["app_level"] = "dashboard"
    cockpit.run()
    assert not cockpit.exception, cockpit.exception
    cockpit.button(key="btn_open_eda").click().run()
    assert not cockpit.exception, cockpit.exception
    assert cockpit.session_state["app_level"] == "eda"


def test_eda_page_handles_empty_fire_frames(monkeypatch):
    import dashboard.eda_view as eda_view

    panel = pd.DataFrame({
        "lat_cell": [36.0], "lon_cell": [3.0],
        "date": pd.to_datetime(["2021-01-01"]),
        "T2M_MAX": [25.0], "burned": [0],
    })
    monkeypatch.setattr(
        eda_view, "get_eda_datasets", lambda: (panel, pd.DataFrame(), pd.DataFrame())
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_level"] = "eda"
    app.run()

    assert not app.exception, app.exception
    assert "Détections agrégées :** 0 lignes" in app.caption[0].value
    assert any("Aucune donnée disponible" in info.value for info in app.info)


def test_eda_page_explains_absent_datasets(monkeypatch):
    import dashboard.eda_view as eda_view

    monkeypatch.setattr(eda_view, "get_eda_datasets", lambda: (None, None, None))
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_level"] = "eda"
    app.run()

    assert not app.exception, app.exception
    assert any("Aucun jeu de données disponible" in warning.value for warning in app.warning)
    assert len(app.get("plotly_chart")) == 0
