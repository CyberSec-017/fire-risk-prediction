"""AppTest coverage for the CSV batch-inference workflow and navigation."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "app.py"
CARD = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))


def _valid_row() -> dict[str, float]:
    row = {name: 10.0 for name in CARD["feature_names"]}
    row.update(
        {
            "lat_cell": 36.5,
            "lon_cell": 3.125,
            "T2M": 28.0,
            "T2M_MAX": 35.0,
            "RH2M": 35.0,
            "WS2M": 4.0,
            "PRECTOTCORR": 0.0,
            "GWETROOT": 0.2,
            "rain_7d": 0.0,
            "tmax_7d": 33.0,
            "rhmin_7d": 30.0,
            "rain_14d": 0.0,
            "tmax_14d": 32.0,
            "rhmin_14d": 32.0,
            "rain_30d": 5.0,
            "tmax_30d": 30.0,
            "rhmin_30d": 35.0,
            "dry_days": 12.0,
            "doy_sin": 0.5,
            "doy_cos": 0.866,
        }
    )
    return row


def _batch_csv_bytes() -> bytes:
    invalid = dict(_valid_row())
    invalid["RH2M"] = ""
    frame = pd.DataFrame([_valid_row(), invalid])
    return frame.to_csv(index=False).encode("utf-8")


def _open_batch() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180)
    app.session_state["app_level"] = "batch"
    app.run()
    assert not app.exception, app.exception
    return app


def _frame_with(app: AppTest, columns: set[str]) -> pd.DataFrame:
    for table in app.dataframe:
        if columns <= set(table.value.columns):
            return table.value
    raise AssertionError(f"No dataframe with columns {columns} found")


def test_batch_page_renders_schema_and_uploader():
    app = _open_batch()

    assert len(app.file_uploader) == 1
    schema = _frame_with(app, {"colonne", "plage acceptée"})
    assert len(schema) == 20
    assert schema["colonne"].tolist() == CARD["feature_names"]
    assert any("SHAP" in warning.value for warning in app.warning) is False
    assert len(app.download_button) == 1  # template only before upload


def test_upload_runs_batch_scores_and_exports_results():
    app = _open_batch()
    app.file_uploader(key="batch_csv_uploader").upload("batch.csv", _batch_csv_bytes())
    app.run()

    assert not app.exception, app.exception
    results = _frame_with(app, {"validation_status", "predicted_probability", "rejection_reasons"})
    assert results["validation_status"].tolist() == ["ACCEPTED", "REJECTED"]
    assert results["horizon_days"].tolist() == [3, 3]
    assert "Valeur manquante pour RH2M" in results["rejection_reasons"].iloc[1]
    assert len(app.download_button) == 2
    assert any(button.key == "batch_download_csv" for button in app.download_button)
    assert any("SHAP n'est pas utilisé" in warning.value for warning in app.warning)
    assert any("pas une attribution causale" in caption.value for caption in app.caption)


def test_local_explanation_table_uses_selected_accepted_row():
    app = _open_batch()
    app.file_uploader(key="batch_csv_uploader").upload("batch.csv", _batch_csv_bytes())
    app.run()

    app.selectbox(key="batch_explain_row").set_value(1).run()

    assert not app.exception, app.exception
    sensitivity = _frame_with(app, {"feature", "delta_pp", "direction"})
    assert not sensitivity.empty
    assert set(sensitivity["direction"]) <= {"increases", "reduces"}


def test_navigation_from_landing_and_cockpit_into_batch_and_back():
    landing = _open_batch()
    landing.button(key="batch_back_to_landing").click().run()
    assert landing.session_state["app_level"] == "landing"

    landing.button(key="btn_explore_batch").click().run()
    assert not landing.exception, landing.exception
    assert landing.session_state["app_level"] == "batch"

    cockpit = AppTest.from_file(str(APP_PATH), default_timeout=180)
    cockpit.session_state["app_level"] = "dashboard"
    cockpit.run()
    cockpit.button(key="btn_open_batch").click().run()
    assert not cockpit.exception, cockpit.exception
    assert cockpit.session_state["app_level"] == "batch"

    cockpit.button(key="batch_back_to_dashboard").click().run()
    assert cockpit.session_state["app_level"] == "dashboard"
