"""Unit coverage for CSV batch validation, production scoring and export."""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import app_core  # noqa: E402
import batch_inference as batch_logic  # noqa: E402


@pytest.fixture(scope="module")
def bundle() -> app_core.ModelBundle:
    card_path = ROOT / "models" / "model_card.json"
    model_path = ROOT / "models" / "fire_model.joblib"
    loaded = app_core.load_model_bundle(model_path, card_path)
    assert loaded.status == "REAL", loaded.error
    return loaded


def _card() -> dict:
    return json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))


def _valid_row() -> dict[str, float]:
    row = {name: 10.0 for name in batch_logic.model_features(_card())}
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


def test_validate_batch_detects_missing_invalid_out_of_range_and_off_grid():
    valid = _valid_row()
    blank = dict(valid, T2M="")
    non_numeric = dict(valid, RH2M="abc")
    out_of_range = dict(valid, GWETROOT=1.8)
    off_grid = dict(valid, lat_cell=36.51)
    raw = pd.DataFrame([valid, blank, non_numeric, out_of_range, off_grid])

    validation = batch_logic.validate_batch(raw, _card())

    assert validation.valid_mask.tolist() == [True, False, False, False, False]
    assert any("Valeur manquante pour T2M" in reason for reason in validation.reasons[1])
    assert any("non numérique pour RH2M" in reason for reason in validation.reasons[2])
    assert any("GWETROOT hors plage" in reason for reason in validation.reasons[3])
    assert any("lat_cell n'est pas aligné" in reason for reason in validation.reasons[4])
    assert list(validation.numeric.columns) == batch_logic.model_features(_card())


def test_missing_required_column_rejects_every_row_with_explicit_reason(bundle):
    valid = _valid_row()
    missing_column = dict(valid)
    del missing_column["WS2M"]
    raw = pd.DataFrame([valid, missing_column]).drop(columns=["WS2M"])

    result = batch_logic.run_batch_inference(raw, bundle)

    assert result.accepted_count == 0
    assert result.rejected_count == 2
    assert result.table["validation_status"].tolist() == ["REJECTED", "REJECTED"]
    assert result.table["predicted_probability"].isna().all()
    assert all(
        "Colonne requise absente: WS2M" in reason
        for reason in result.table["rejection_reasons"]
    )


def test_run_batch_inference_scores_only_valid_rows_with_production_contract(bundle):
    valid = _valid_row()
    invalid = dict(valid, RH2M="")
    raw = pd.DataFrame([valid, invalid])

    result = batch_logic.run_batch_inference(raw, bundle)

    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.accepted_row_numbers == [1]
    assert math.isfinite(result.accepted_probabilities[0])
    assert 0.0 <= result.accepted_probabilities[0] <= 1.0
    assert result.table.loc[0, "risk_category"] in _card()["risk_labels"]
    assert result.table.loc[1, "risk_category"] == ""
    assert math.isnan(result.table.loc[1, "predicted_probability"])
    assert set(result.table["horizon_days"].tolist()) == {3}
    assert result.table.loc[0, "lat_cell"] == 36.5
    assert result.table.loc[1, "rejection_reasons"] == "Valeur manquante pour RH2M"


def test_export_contains_original_inputs_probability_reasons_and_horizon(bundle):
    raw = pd.DataFrame([_valid_row(), dict(_valid_row(), RH2M="")])
    result = batch_logic.run_batch_inference(raw, bundle)

    csv_text = batch_logic.to_csv_bytes(result.table).decode("utf-8-sig")
    exported = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)

    expected = set(batch_logic.DERIVED_COLUMNS) | set(batch_logic.model_features(_card()))
    assert expected <= set(exported.columns)
    assert exported["validation_status"].tolist() == ["ACCEPTED", "REJECTED"]
    assert exported["horizon_days"].tolist() == ["3", "3"]
    assert exported["predicted_probability"].iloc[0]
    assert exported["predicted_probability"].iloc[1] == ""
    assert "Valeur manquante pour RH2M" in exported["rejection_reasons"].iloc[1]
    assert exported["T2M_MAX"].iloc[0] == "35.0"


def test_empty_batch_returns_empty_result(bundle):
    raw = pd.DataFrame(columns=batch_logic.model_features(_card()))

    result = batch_logic.run_batch_inference(raw, bundle)

    assert result.table.empty
    assert result.accepted_count == 0
    assert result.rejected_count == 0
    assert result.accepted_probabilities.size == 0


def test_template_has_exactly_card_features_in_order():
    card = _card()
    template = batch_logic.read_csv_bytes(batch_logic.template_csv_bytes(card))
    assert template.columns.tolist() == card["feature_names"]
    assert template.empty
