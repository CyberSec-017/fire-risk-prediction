"""Unit tests for the 6-classifier comparison module (src/compare_classifiers.py).

Verifies model pipelines, date-based OOF fold generation with strict purge,
F1 threshold selection via precision_recall_curve, metric calculation,
and hermetic CSV export using tmp_path.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compare_classifiers import (
    build_date_folds,
    evaluate_predictions,
    find_optimal_threshold_oof,
    fit_and_evaluate_classifier,
    get_classifiers,
    save_comparison_results,
    subsample_if_needed,
)


def test_get_classifiers_returns_six_models():
    """Verify that get_classifiers returns exactly the 6 required algorithms."""
    models = get_classifiers(random_state=42)
    assert len(models) == 6

    expected_names = {
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
        "KNeighborsClassifier",
        "GaussianNB",
    }
    assert set(models.keys()) == expected_names


def test_model_pipelines_and_scalers():
    """Verify StandardScaler is used for linear/distance models and absent for trees/Bayes.

    Also verifies SimpleImputer is included for models that do not handle NaNs natively,
    and absent for HistGradientBoostingClassifier.
    """
    models = get_classifiers(random_state=42)

    # Models requiring scaling
    for scaled_name in ("LogisticRegression", "KNeighborsClassifier"):
        model = models[scaled_name]
        assert isinstance(model, Pipeline), f"{scaled_name} must be a Pipeline"
        assert "scaler" in model.named_steps
        assert isinstance(model.named_steps["scaler"], StandardScaler)
        assert "imputer" in model.named_steps

    # Models that must NOT use scaling
    for unscaled_name in (
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GaussianNB",
    ):
        model = models[unscaled_name]
        assert isinstance(model, Pipeline)
        assert "scaler" not in model.named_steps
        assert "imputer" in model.named_steps

    # HistGradientBoostingClassifier natively supports NaNs
    hgb = models["HistGradientBoostingClassifier"]
    assert not isinstance(hgb, Pipeline)


def test_build_date_folds_purge_and_no_leakage():
    """Verify that folds are split by unique dates with a strict 3-day purge."""
    # Create synthetic panel spanning 40 consecutive days with 5 cells per day
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    records = []
    for d in dates:
        for cell_id in range(5):
            records.append({
                "date": d,
                "lat_cell": 35.0 + cell_id * 0.5,
                "lon_cell": 3.0 + cell_id * 0.5,
                "val": np.random.randn(),
            })
    synthetic_df = pd.DataFrame(records)

    folds = build_date_folds(synthetic_df, n_splits=3, purge_days=3)
    assert len(folds) == 3

    for fold_i, (tr_idx, val_idx) in enumerate(folds, 1):
        assert len(tr_idx) > 0, f"Fold {fold_i} train set is empty"
        assert len(val_idx) > 0, f"Fold {fold_i} val set is empty"

        tr_dates = set(synthetic_df.iloc[tr_idx]["date"])
        val_dates = set(synthetic_df.iloc[val_idx]["date"])

        # 1. Zero date overlap
        assert tr_dates.isdisjoint(val_dates)

        # 2. Strict 3-day purge: max(tr_dates) < min(val_dates) - 3 days
        min_val_date = min(val_dates)
        purge_limit = min_val_date - pd.Timedelta(days=3)
        assert max(tr_dates) < purge_limit, (
            f"Fold {fold_i}: Train date {max(tr_dates)} violates strict 3-day purge "
            f"before val start {min_val_date}"
        )

        # 3. Indivisibility: Every date present has all 5 cells
        for d in tr_dates:
            assert (synthetic_df.iloc[tr_idx]["date"] == d).sum() == 5
        for d in val_dates:
            assert (synthetic_df.iloc[val_idx]["date"] == d).sum() == 5


def test_find_optimal_threshold_oof_logic():
    """Verify threshold selection from continuous precision_recall_curve and NaN handling."""
    # 1. Verification of NaN exclusion (warm-up block 0)
    y_train = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    p_oof = np.array([np.nan, np.nan, 0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.2])

    thresh = find_optimal_threshold_oof(y_train, p_oof)
    assert 0.0 < thresh < 1.0

    # 2. Verification of optimal F1 threshold choice on known probabilities
    # Classes with clear separation around 0.55
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_preds = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    best_th = find_optimal_threshold_oof(y_true, p_preds)
    # Threshold should separate 0.4 and 0.6 to get F1 = 1.0
    assert 0.4 < best_th <= 0.6

    # 3. Degenerate case fallback
    all_nan = np.full(10, np.nan)
    assert find_optimal_threshold_oof(y_train, all_nan) == 0.5


def test_find_optimal_threshold_tied_candidates_middle_selection():
    """Verify that tied optimal thresholds select the middle candidate and strictly belong to thresholds.

    Covers both even (2) and odd (3) non-consecutive candidate ties, verifying:
    1. The selected threshold strictly belongs to candidate thresholds (not an interpolated average).
    2. The F1 score at the selected threshold matches the exact maximum F1.
    3. In contrast to numerical median (which can yield suboptimal F1), middle candidate preserves max F1.
    """
    from sklearn.metrics import precision_recall_curve, f1_score

    # Case 1: Two non-consecutive tied optimal thresholds (indices 0 and 4)
    # Designed specifically where numerical median produces a non-candidate with inferior F1
    y1 = np.array([1, 0, 0, 0, 1, 0])
    p1 = np.array([0.10, 0.26, 0.42, 0.58, 0.74, 0.90])

    prec1, rec1, th1 = precision_recall_curve(y1, p1)
    denom1 = prec1[:-1] + rec1[:-1]
    f1_1 = np.zeros_like(denom1, dtype=float)
    nz1 = denom1 > 0
    f1_1[nz1] = 2 * prec1[:-1][nz1] * rec1[:-1][nz1] / denom1[nz1]
    max_f1_1 = np.max(f1_1)
    best1 = np.where(np.isclose(f1_1, max_f1_1))[0]

    # Confirm ties are non-consecutive
    assert len(best1) == 2
    assert np.diff(best1)[0] > 1  # indices [0, 4] separated by lower F1 values

    opt_th1 = find_optimal_threshold_oof(y1, p1)

    # 1. Returned threshold must strictly belong to candidate thresholds
    assert opt_th1 in th1
    # 2. Returned threshold must equal the middle candidate (index 4 in length 2: 2 // 2 = 1)
    assert opt_th1 == float(th1[best1[len(best1) // 2]])
    # 3. Must achieve the exact maximum F1
    actual_f1_1 = f1_score(y1, (p1 >= opt_th1).astype(int))
    assert np.isclose(actual_f1_1, max_f1_1, atol=1e-5)

    # 4. Demonstrate why numerical median would be flawed here
    numerical_median_th = float(np.median(th1[best1]))
    numerical_median_f1 = f1_score(y1, (p1 >= numerical_median_th).astype(int))
    assert numerical_median_f1 < max_f1_1

    # Case 2: Three non-consecutive tied optimal thresholds (indices 0, 3, 6)
    y2 = np.array([1, 0, 0, 1, 0, 0, 1, 1])
    p2 = np.array([0.10, 0.21, 0.33, 0.44, 0.56, 0.67, 0.79, 0.90])

    prec2, rec2, th2 = precision_recall_curve(y2, p2)
    denom2 = prec2[:-1] + rec2[:-1]
    f1_2 = np.zeros_like(denom2, dtype=float)
    nz2 = denom2 > 0
    f1_2[nz2] = 2 * prec2[:-1][nz2] * rec2[:-1][nz2] / denom2[nz2]
    max_f1_2 = np.max(f1_2)
    best2 = np.where(np.isclose(f1_2, max_f1_2))[0]

    assert len(best2) == 3
    assert np.all(np.diff(best2) > 1)  # indices [0, 3, 6] non-consecutive

    opt_th2 = find_optimal_threshold_oof(y2, p2)

    assert opt_th2 in th2
    # Middle candidate of [0, 3, 6] is index 3 (3 // 2 = 1)
    assert opt_th2 == float(th2[best2[len(best2) // 2]])
    actual_f1_2 = f1_score(y2, (p2 >= opt_th2).astype(int))
    assert np.isclose(actual_f1_2, max_f1_2, atol=1e-5)


def test_evaluate_predictions_metrics():
    """Verify all metrics are computed accurately on synthetic ground truth."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probas = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = evaluate_predictions(y_true, probas, optimal_threshold=0.5)

    expected_keys = {
        "roc_auc", "pr_auc", "brier_score", "optimal_threshold",
        "f1_optimal", "precision_optimal", "recall_optimal",
        "f1_default_05", "precision_default_05", "recall_default_05",
        "accuracy_default_05",
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1_optimal"] == 1.0
    assert metrics["accuracy_default_05"] == 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_subsample_if_needed():
    """Verify stratified subsampling preserves requested sample cap and class ratio."""
    np.random.seed(42)
    X = pd.DataFrame({"feat1": np.random.randn(100), "feat2": np.random.randn(100)})
    y = pd.Series([1] * 20 + [0] * 80)

    # Under cap -> returns original
    X_same, y_same = subsample_if_needed(X, y, max_samples=150)
    assert len(X_same) == 100

    # Over cap -> returns exactly max_samples with stratified base rate
    X_sub, y_sub = subsample_if_needed(X, y, max_samples=40, random_state=42)
    assert len(X_sub) == 40
    assert len(y_sub) == 40
    # Stratification check (20% of 40 = 8 positives)
    assert y_sub.sum() == 8


def test_results_directory_creation(tmp_path):
    """Verify hermetic CSV writing using pytest tmp_path without touching project results/."""
    dummy_df = pd.DataFrame({
        "model": ["DummyClassifier"],
        "pr_auc": [0.42],
        "optimal_threshold": [0.15],
        "f1_optimal": [0.65],
    })

    target_dir = tmp_path / "custom_results"
    output_path = save_comparison_results(dummy_df, output_dir=target_dir)

    # 1. Output file exists
    assert output_path.is_file()
    # 2. Parent directory is exactly target_dir
    assert output_path.parent == target_dir
    # 3. Path is hermetically contained within tmp_path
    assert output_path.is_relative_to(tmp_path)
    # 4. Content is readable and matches input
    loaded = pd.read_csv(output_path)
    assert len(loaded) == 1
    assert loaded["model"].iloc[0] == "DummyClassifier"
    assert loaded["optimal_threshold"].iloc[0] == 0.15


def test_fit_and_evaluate_classifier_fast_dummy():
    """Verify fit_and_evaluate_classifier runs end-to-end on a tiny synthetic dataset."""
    dates_tr = pd.date_range("2020-01-01", periods=30, freq="D")
    dates_va = pd.date_range("2020-02-15", periods=10, freq="D")

    def make_dataset(dates):
        rows = []
        for d in dates:
            for cell_id in range(3):
                rows.append({
                    "date": d,
                    "lat_cell": 35.0 + cell_id,
                    "lon_cell": 3.0 + cell_id,
                    "feat1": np.random.randn(),
                    "feat2": np.random.randn(),
                    "target": int(np.random.rand() > 0.8),
                })
        return pd.DataFrame(rows)

    train_dummy = make_dataset(dates_tr)
    val_dummy = make_dataset(dates_va)
    features = ["feat1", "feat2"]

    model = get_classifiers(random_state=42)["DecisionTreeClassifier"]
    record = fit_and_evaluate_classifier(
        name="DecisionTreeClassifier",
        model=model,
        train_df=train_dummy,
        val_df=val_dummy,
        features=features,
        label="target",
        random_state=42,
    )

    assert record["model"] == "DecisionTreeClassifier"
    assert "pr_auc" in record
    assert "roc_auc" in record
    assert "f1_optimal" in record
    assert record["train_samples"] == len(train_dummy)
    assert record["val_samples"] == len(val_dummy)
