"""Tests for benchmark harmonisation, protocol guards and output structure."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import model_benchmark as mb  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archived_frame(model: str = "RandomForestClassifier") -> pd.DataFrame:
    return pd.DataFrame([{
        "model": model,
        "train_samples": 1000,
        "val_samples": 200,
        "fit_time_sec": 1.5,
        "eval_time_sec": 0.2,
        "roc_auc": 0.87,
        "pr_auc": 0.26,
        "brier_score": 0.04,
        "optimal_threshold": 0.2,
        "f1_optimal": 0.35,
        "precision_optimal": 0.28,
        "recall_optimal": 0.47,
        "f1_default_05": 0.19,
        "precision_default_05": 0.4,
        "recall_default_05": 0.13,
        "accuracy_default_05": 0.95,
        "tuning_time_sec": 10.0,
        "mean_cv_pr_auc": 0.3,
    }])


def _baseline_record() -> dict[str, object]:
    return {
        "model": "DummyClassifier (prior)",
        "train_samples": 1000,
        "val_samples": 200,
        "oof_time_sec": 0.1,
        "fit_time_sec": 0.01,
        "eval_time_sec": 0.0,
        "roc_auc": 0.5,
        "pr_auc": 0.047,
        "brier_score": 0.045,
        "optimal_threshold": 0.043,
        "f1_optimal": 0.09,
        "precision_optimal": 0.047,
        "recall_optimal": 1.0,
        "f1_default_05": 0.0,
        "precision_default_05": 0.0,
        "recall_default_05": 0.0,
        "accuracy_default_05": 0.953,
    }


def test_validate_benchmark_frame_rejects_missing_schema():
    with pytest.raises(ValueError, match="Colonnes de benchmark manquantes"):
        mb.validate_benchmark_frame(pd.DataFrame({"model": ["DummyClassifier"]}))


def test_load_benchmark_results_returns_typed_empty_frame_when_absent(tmp_path):
    frame = mb.load_benchmark_results(tmp_path / "missing.csv")

    assert frame.empty
    assert list(frame.columns) == list(mb.EXPECTED_COLUMNS)


def test_load_benchmark_results_parses_boolean_flags(tmp_path):
    row = {column: np.nan for column in mb.EXPECTED_COLUMNS}
    row.update({
        "experiment": mb.PRIMARY_EXPERIMENT,
        "model": "DummyClassifier (prior)",
        "threshold_metrics_unbiased": True,
        "include_in_ranking": False,
    })
    frame = pd.DataFrame([row])
    path = tmp_path / "benchmark.csv"
    frame.to_csv(path, index=False)

    loaded = mb.load_benchmark_results(path)
    assert bool(loaded.loc[0, "include_in_ranking"]) is False
    assert bool(loaded.loc[0, "threshold_metrics_unbiased"]) is True


def test_optional_algorithm_availability_is_explicit():
    details = mb.optional_algorithm_availability()

    assert set(details) == {"XGBoost", "LightGBM", "CatBoost"}
    for info in details.values():
        assert isinstance(info["available"], bool)
        assert info["reason"]
        if not info["available"]:
            assert "aucun résultat" in info["reason"]


def test_normalize_comparison_frame_adds_family_labels_and_ranking_flags():
    normalized = mb.normalize_comparison_frame(
        _archived_frame(),
        experiment=mb.TUNED_EXPERIMENT,
        threshold_source="validation",
        threshold_metrics_unbiased=False,
        include_in_ranking=False,
        source_file="tuned.csv",
        notes="seuil optimiste",
    )

    assert len(normalized) == 1
    row = normalized.iloc[0]
    assert row["model_label"] == "Random Forest"
    assert row["family"] == "Random forest"
    assert row["experiment"] == mb.TUNED_EXPERIMENT
    assert bool(row["include_in_ranking"]) is False
    assert row["pr_auc"] == pytest.approx(0.26)


def test_build_benchmark_artifacts_creates_valid_csv_and_protocol(tmp_path):
    features = mb.load_production_feature_contract()
    assert len(features) == 20
    untuned_path = tmp_path / "classifier_comparison.csv"
    tuned_path = tmp_path / "tuned_classifier_comparison.csv"
    _archived_frame().to_csv(untuned_path, index=False)
    _archived_frame().to_csv(tuned_path, index=False)
    train = pd.DataFrame({"date": pd.to_datetime(["2021-01-01"] * 10),
                          "target": [0] * 10})
    val = pd.DataFrame({"date": pd.to_datetime(["2022-01-01"] * 5),
                        "target": [0, 1, 0, 0, 0]})
    model_hashes = {
        path: _hash(path)
        for path in (ROOT / "models" / "fire_model.joblib", ROOT / "models" / "model_card.json")
    }

    benchmark, protocol = mb.build_benchmark_artifacts(
        output_dir=tmp_path,
        archived_untuned_path=untuned_path,
        archived_tuned_path=tuned_path,
        train_df=train,
        val_df=val,
        features=features,
        baseline_record=_baseline_record(),
        seed=7,
    )

    mb.validate_benchmark_frame(benchmark)
    assert len(benchmark) == 3
    assert set(benchmark["experiment"]) == {mb.PRIMARY_EXPERIMENT, mb.TUNED_EXPERIMENT}
    primary = benchmark[benchmark["experiment"] == mb.PRIMARY_EXPERIMENT]
    assert set(primary["model"]) == {"DummyClassifier (prior)", "RandomForestClassifier"}
    assert primary["include_in_ranking"].all()
    assert not benchmark.loc[benchmark["experiment"] == mb.TUNED_EXPERIMENT, "include_in_ranking"].any()
    assert protocol["seed"] == 7
    assert protocol["model_contract"]["n_features"] == 20
    assert protocol["model_contract"]["artifacts_modified"] is False
    assert protocol["class_balance"]["val_positive_rate"] == pytest.approx(0.2)
    assert set(protocol["algorithms"]["excluded"]) == {"XGBoost", "LightGBM", "CatBoost"}
    for path, digest in model_hashes.items():
        assert _hash(path) == digest

    reloaded = mb.load_benchmark_results(tmp_path / mb.BENCHMARK_CSV.name)
    assert len(reloaded) == 3
    assert (tmp_path / mb.BENCHMARK_PROTOCOL_JSON.name).exists()


def test_committed_benchmark_artifact_matches_the_shared_split_protocol():
    frame = mb.load_benchmark_results()
    protocol = mb.load_benchmark_protocol()

    assert len(frame) >= 13
    families = set(frame["family"])
    assert {"Simple baseline", "Random forest", "Gradient boosting"} <= families
    assert (frame["experiment"] == mb.PRIMARY_EXPERIMENT).sum() == 7
    assert (frame["experiment"] == mb.TUNED_EXPERIMENT).sum() == 6
    assert frame.loc[frame["model"] == "DummyClassifier (prior)", "roc_auc"].iloc[0] == pytest.approx(0.5)
    assert protocol["split"]["val_rows"] == 75658
    assert protocol["split"]["purge_days"] == 3
    assert protocol["class_balance"]["val_positive_rate"] < 0.06
