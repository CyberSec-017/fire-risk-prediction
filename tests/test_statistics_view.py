"""AppTest coverage for the statistical-validation and benchmark section."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "app.py"


def _open_statistics() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_level"] = "statistics"
    app.run()
    assert not app.exception, app.exception
    return app


def _frame_with(app: AppTest, columns: set[str]) -> pd.DataFrame:
    for table in app.dataframe:
        if columns <= set(table.value.columns):
            return table.value
    raise AssertionError(f"No dataframe with columns {columns} found")


def test_statistics_view_renders_tests_benchmarks_and_tables():
    app = _open_statistics()

    assert len(app.get("plotly_chart")) >= 5
    assert len(app.dataframe) >= 5
    assert len(app.download_button) == 2
    assert len(app.metric) >= 5
    rendered = " ".join(
        item.value
        for group in (app.markdown, app.html, app.caption, app.info)
        for item in group
    )
    assert "Validation statistique & Benchmark des modèles" in rendered
    assert any("Seuil de significativité" in box.label for box in app.selectbox)
    assert "statistiquement significatif" in rendered.lower()
    assert "performance de validation" in rendered.lower()
    assert "Accuracy : trompeuse" in rendered

    summary = _frame_with(app, {"test", "p_value", "decision", "alpha"})
    assert len(summary) == 4
    assert set(summary["test"].str.contains("χ²|Kruskal|Shapiro|Mann", regex=True)) == {True}

    algorithms = _frame_with(app, {"algorithme", "inclus", "raison"})
    assert set(algorithms["algorithme"]) >= {"XGBoost", "LightGBM", "CatBoost"}


def test_alpha_selector_updates_decision_without_changing_pvalues():
    app = _open_statistics()
    before = _frame_with(app, {"test", "p_value", "decision", "alpha"}).copy()

    app.selectbox(key="stats_alpha").set_value(0.01).run()

    assert not app.exception, app.exception
    after = _frame_with(app, {"test", "p_value", "decision", "alpha"})
    assert (after["p_value"].tolist() == before["p_value"].tolist())
    assert after["alpha"].tolist() == ["0.01"] * 4
    assert after["decision"].str.contains("0.01").all()


def test_benchmark_sort_and_reference_toggle_are_wired():
    app = _open_statistics()
    primary = _frame_with(app, {"model_label", "pr_auc", "threshold_source"})
    assert len(primary) == 7
    assert primary.iloc[0]["model_label"] == "HistGradientBoosting"

    app.checkbox(key="stats_include_tuned").set_value(True).run()

    assert not app.exception, app.exception
    visible = _frame_with(app, {"model_label", "pr_auc", "threshold_source"})
    assert len(visible) == 13
    assert visible["threshold_source"].str.contains("validation").any()
    assert not visible["threshold_source"].str.contains("optimiste").all()

    app.selectbox(key="stats_benchmark_metric").set_value("Brier (plus bas est meilleur)").run()

    assert not app.exception, app.exception
    assert len(app.get("plotly_chart")) >= 5


def test_navigation_from_landing_and_cockpit_into_statistics_and_back():
    landing = AppTest.from_file(str(APP_PATH), default_timeout=180)
    landing.run()
    landing.button(key="btn_explore_statistics").click().run()

    assert not landing.exception, landing.exception
    assert landing.session_state["app_level"] == "statistics"
    assert len(landing.get("plotly_chart")) >= 5

    landing.button(key="stats_back_to_dashboard").click().run()
    assert not landing.exception, landing.exception
    assert landing.session_state["app_level"] == "dashboard"

    cockpit = AppTest.from_file(str(APP_PATH), default_timeout=180)
    cockpit.session_state["app_level"] = "dashboard"
    cockpit.run()
    assert not cockpit.exception, cockpit.exception
    cockpit.button(key="btn_open_statistics").click().run()

    assert not cockpit.exception, cockpit.exception
    assert cockpit.session_state["app_level"] == "statistics"

    cockpit.button(key="stats_back_to_landing").click().run()
    assert not cockpit.exception, cockpit.exception
    assert cockpit.session_state["app_level"] == "landing"


def test_statistics_page_explains_missing_artifacts(monkeypatch):
    import dashboard.statistics_view as statistics_view

    monkeypatch.setattr(statistics_view, "get_statistical_report", lambda: None)
    monkeypatch.setattr(
        statistics_view,
        "get_benchmark_artifacts",
        lambda: (pd.DataFrame(), {}),
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_level"] = "statistics"
    app.run()

    assert not app.exception, app.exception
    assert any("Rapport statistique indisponible" in warning.value for warning in app.warning)
    assert any("Benchmark indisponible" in warning.value for warning in app.warning)
