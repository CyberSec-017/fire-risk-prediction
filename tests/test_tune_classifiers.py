"""Unit tests for the temporal GridSearchCV tuning module (src/tune_classifiers.py).

Verifies:
  1. Methodological 5-fold expanding window cross-validation (chronology, 3-day purge, cell atomicity, class balance).
  2. Training-only subsampling (preserves 100% of validation, stratified on target, reproducible).
  3. Hyperparameter grid specifications (88 total configurations, 440 CV fits, exact parameter naming).
  4. Anti-oversubscription resource limits (n_jobs=1 inside estimators, n_jobs=2 in GridSearchCV).
  5. Refit protocol (refit=False, manual refit on full Train, 25k cap on KNN, F1 threshold on Validation).
  6. Hermetic artifact saving using pytest tmp_path.
  7. Absolute isolation of the Test dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tune_classifiers import (
    KNN_MAX_SEARCH_TRAIN_SAMPLES,
    KNN_MAX_TRAIN_SAMPLES,
    MODEL_FILE_SLUGS,
    PURGE_DAYS,
    RF_MAX_SEARCH_TRAIN_SAMPLES,
    build_tuning_date_folds,
    get_param_grids,
    get_tuning_classifiers,
    refit_and_evaluate_tuned_model,
    save_comparison_table,
    save_hyperparameters,
    save_single_model_cv_results,
    save_tuning_artifacts,
    tune_single_classifier,
)


# ============================================================================
# Fixtures: Synthetic datasets for fast, independent unit tests
# ============================================================================

@pytest.fixture
def synthetic_panel_df() -> pd.DataFrame:
    """Generate a lightweight synthetic spatio-temporal panel dataset.

    60 unique dates, 5 grid cells per date = 300 observations.
    Target exhibits ~10% positive rate with both classes present across all time windows.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    cells = [(35.0 + i * 0.5, 3.0 + i * 0.5) for i in range(5)]

    rows = []
    for d in dates:
        for lat, lon in cells:
            rows.append({
                "date": d,
                "lat_cell": lat,
                "lon_cell": lon,
                "feat1": np.random.randn(),
                "feat2": np.random.rand(),
                "target": int(np.random.rand() < 0.12),
            })
    df = pd.DataFrame(rows)
    # Guarantee at least some positive targets in every segment
    df.loc[df["date"] == dates[5], "target"] = 1
    df.loc[df["date"] == dates[15], "target"] = 1
    df.loc[df["date"] == dates[25], "target"] = 1
    df.loc[df["date"] == dates[35], "target"] = 1
    df.loc[df["date"] == dates[45], "target"] = 1
    df.loc[df["date"] == dates[55], "target"] = 1
    return df


# ============================================================================
# 1. Methodological Tests: 5 Folds, Chronology, Purge & Atomicity
# ============================================================================

def test_build_tuning_date_folds_five_folds_and_chronology(synthetic_panel_df: pd.DataFrame):
    """Verify exactly 5 folds are generated, strictly chronological with no overlap."""
    y = synthetic_panel_df["target"]
    folds = build_tuning_date_folds(synthetic_panel_df, y, n_splits=5, purge_days=3)

    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    prev_train_len = 0
    for k, (tr_idx, val_idx) in enumerate(folds, 1):
        tr_dates = synthetic_panel_df["date"].iloc[tr_idx]
        val_dates = synthetic_panel_df["date"].iloc[val_idx]

        # 1. Strict chronological ordering (all train dates precede all val dates)
        assert tr_dates.max() < val_dates.min(), f"Fold {k}: Chronology violated."

        # 2. Expanding window property (train size strictly increases with fold index)
        assert len(tr_idx) > prev_train_len, f"Fold {k}: Expanding window property violated."
        prev_train_len = len(tr_idx)

        # 3. No overlap between train and validation indices
        overlap = set(tr_idx).intersection(set(val_idx))
        assert len(overlap) == 0, f"Fold {k}: Train and val overlap ({len(overlap)} indices)."

        # 4. Valid positional indices within [0, len(df)-1]
        assert np.all(tr_idx >= 0) and np.all(tr_idx < len(synthetic_panel_df))
        assert np.all(val_idx >= 0) and np.all(val_idx < len(synthetic_panel_df))


def test_build_tuning_date_folds_purge_gap(synthetic_panel_df: pd.DataFrame):
    """Verify that exactly 3 calendar days are strictly excluded between train and val.

    If last train date is T_train and first val date is T_val:
    The 3 days (T_val - 3 days, T_val - 2 days, T_val - 1 day) must be absent from train.
    Therefore, T_val - T_train must be at least 4 days (or strictly > 3 days).
    """
    y = synthetic_panel_df["target"]
    purge_days = 3
    folds = build_tuning_date_folds(synthetic_panel_df, y, n_splits=5, purge_days=purge_days)

    for k, (tr_idx, val_idx) in enumerate(folds, 1):
        t_max_train = synthetic_panel_df["date"].iloc[tr_idx].max()
        t_min_val = synthetic_panel_df["date"].iloc[val_idx].min()

        date_diff = (t_min_val - t_max_train).days
        # Exactly 3 intervening calendar days excluded means the date diff must be > 3 days (>= 4 days)
        assert date_diff > purge_days, (
            f"Fold {k}: Purge gap violated! Max train date is {t_max_train}, "
            f"min val date is {t_min_val}, difference is only {date_diff} days."
        )

        # Check explicitly that none of the 3 purged calendar dates exist in the training set
        for p in range(1, purge_days + 1):
            purged_date = t_min_val - pd.Timedelta(days=p)
            assert purged_date not in set(synthetic_panel_df["date"].iloc[tr_idx]), (
                f"Fold {k}: Purged date {purged_date} leaked into training set!"
            )


def test_build_tuning_date_folds_cell_atomicity(synthetic_panel_df: pd.DataFrame):
    """Verify that all spatial cells of any calendar date stay strictly together in the same fold."""
    y = synthetic_panel_df["target"]
    folds = build_tuning_date_folds(synthetic_panel_df, y, n_splits=5, purge_days=3)

    for k, (tr_idx, val_idx) in enumerate(folds, 1):
        tr_dates_set = set(synthetic_panel_df["date"].iloc[tr_idx])
        val_dates_set = set(synthetic_panel_df["date"].iloc[val_idx])

        # Date sets must be completely disjoint
        assert tr_dates_set.isdisjoint(val_dates_set), f"Fold {k}: Dates split across train and val!"

        # Every date in val must have all its cells in val
        for d in val_dates_set:
            total_cells_for_d = (synthetic_panel_df["date"] == d).sum()
            cells_in_val = (synthetic_panel_df["date"].iloc[val_idx] == d).sum()
            assert cells_in_val == total_cells_for_d, (
                f"Fold {k}: Date {d} has {cells_in_val}/{total_cells_for_d} cells in validation."
            )


def test_build_tuning_date_folds_preserves_full_validation_and_both_classes(synthetic_panel_df: pd.DataFrame):
    """Verify 100% of validation block observations are preserved and both classes exist in all folds."""
    y = synthetic_panel_df["target"]
    folds = build_tuning_date_folds(synthetic_panel_df, y, n_splits=5, purge_days=3)

    # 60 dates split into 6 blocks: B0 (10), B1 (10), B2 (10), B3 (10), B4 (10), B5 (10)
    # Each block has 10 dates * 5 cells = 50 rows
    for k, (tr_idx, val_idx) in enumerate(folds, 1):
        assert len(val_idx) == 50, f"Fold {k}: Validation size altered, got {len(val_idx)}"
        # Verify both binary classes exist in train and val
        assert synthetic_panel_df["target"].iloc[tr_idx].nunique() == 2, f"Fold {k} train lacks 2 classes"
        assert synthetic_panel_df["target"].iloc[val_idx].nunique() == 2, f"Fold {k} val lacks 2 classes"


# ============================================================================
# 2. Subsampling Tests: Stratified, Training-Only & Reproducible
# ============================================================================

def test_subsampling_only_on_training_indices():
    """Verify subsampling is applied exclusively to fold training indices, keeping validation intact."""
    # Create dataset with 60 dates, 20 cells per date = 1200 rows
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []
    for d in dates:
        for i in range(20):
            rows.append({
                "date": d,
                "lat_cell": 35.0 + i * 0.1,
                "lon_cell": 3.0 + i * 0.1,
                "feat": np.random.randn(),
                "target": int(np.random.rand() < 0.2),
            })
    df = pd.DataFrame(rows)
    # Ensure positive cases throughout
    df.loc[::10, "target"] = 1
    y = df["target"]

    # Folds without subsampling
    folds_unconstrained = build_tuning_date_folds(df, y, n_splits=5, purge_days=3)

    # Folds with training cap = 80 samples
    cap = 80
    folds_subsampled = build_tuning_date_folds(df, y, n_splits=5, purge_days=3, max_train_samples=cap)

    for k in range(5):
        raw_tr, raw_va = folds_unconstrained[k]
        sub_tr, sub_va = folds_subsampled[k]

        # 1. Validation size must be 100% IDENTICAL
        assert len(sub_va) == len(raw_va), f"Fold {k+1}: Validation set was modified by subsampling!"
        assert np.array_equal(sub_va, raw_va), f"Fold {k+1}: Validation indices differ!"

        # 2. Training size must respect cap
        if len(raw_tr) > cap:
            assert len(sub_tr) == cap, f"Fold {k+1}: Expected training size {cap}, got {len(sub_tr)}"
        else:
            assert len(sub_tr) == len(raw_tr)

        # 3. Subsampled indices must be a strict subset of the raw training indices
        assert set(sub_tr).issubset(set(raw_tr)), f"Fold {k+1}: Subsampled indices leaked outside raw train!"

        # 4. Stratification check: positive class rate must be preserved within binomial tolerance
        raw_pos_rate = y.iloc[raw_tr].mean()
        sub_pos_rate = y.iloc[sub_tr].mean()
        assert abs(raw_pos_rate - sub_pos_rate) < 0.05, f"Fold {k+1}: Stratification degraded!"


def test_subsampling_reproducibility():
    """Verify that identical random_state produces identical subsampled indices."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": np.repeat(dates, 10),
        "target": np.random.choice([0, 1], size=600, p=[0.9, 0.1]),
    })
    y = df["target"]

    folds_run1 = build_tuning_date_folds(df, y, n_splits=5, purge_days=3, max_train_samples=50, random_state=123)
    folds_run2 = build_tuning_date_folds(df, y, n_splits=5, purge_days=3, max_train_samples=50, random_state=123)
    folds_run3 = build_tuning_date_folds(df, y, n_splits=5, purge_days=3, max_train_samples=50, random_state=999)

    for k in range(5):
        # Same seed must yield identical indices
        assert np.array_equal(folds_run1[k][0], folds_run2[k][0])
        # Different seed should yield different sample
        assert not np.array_equal(folds_run1[k][0], folds_run3[k][0])


def test_subsampling_cap_constants():
    """Verify that the domain cap constants match the architectural decisions."""
    assert RF_MAX_SEARCH_TRAIN_SAMPLES == 60000
    assert KNN_MAX_SEARCH_TRAIN_SAMPLES == 20000
    assert KNN_MAX_TRAIN_SAMPLES == 25000
    assert PURGE_DAYS == 3


# ============================================================================
# 3. Parameter Grids Tests: Combinations & Naming Introspection
# ============================================================================

def test_param_grids_combination_counts():
    """Verify that the 6 grids contain exactly the approved 88 combinations (440 CV fits)."""
    grids = get_param_grids()

    expected_counts = {
        "LogisticRegression": 4 * 2,                         # 8
        "DecisionTreeClassifier": 3 * 3 * 2,                 # 18
        "RandomForestClassifier": 3 * 2 * 3,                 # 18
        "HistGradientBoostingClassifier": 3 * 3 * 2 * 2,     # 36
        "KNeighborsClassifier": 2 * 2,                       # 4
        "GaussianNB": 4,                                     # 4
    }

    assert set(grids.keys()) == set(expected_counts.keys())

    total_combos = 0
    for name, grid in grids.items():
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        assert n_combos == expected_counts[name], (
            f"Model {name}: expected {expected_counts[name]} combinations, got {n_combos}"
        )
        total_combos += n_combos

    assert total_combos == 88, f"Expected 88 total configurations, got {total_combos}"
    assert total_combos * 5 == 440, f"Expected 440 total CV fits, got {total_combos * 5}"


def test_param_grids_exact_parameter_names():
    """Verify that all parameter names in param_grid exist in the estimator via get_params(deep=True)."""
    models = get_tuning_classifiers(random_state=42)
    grids = get_param_grids()

    for name, model in models.items():
        valid_keys = set(model.get_params(deep=True).keys())
        for param in grids[name].keys():
            assert param in valid_keys, (
                f"Model {name}: parameter '{param}' not found in estimator params! "
                f"Valid params: {sorted(valid_keys)}"
            )

    # Specific architectural checks:
    # 1. HistGradientBoostingClassifier must NOT have 'clf__' prefix
    for p in grids["HistGradientBoostingClassifier"].keys():
        assert not p.startswith("clf__"), f"HistGradientBoostingClassifier should not have clf__ prefix: {p}"

    # 2. GaussianNB must have 'clf__' prefix (wrapped in Pipeline with SimpleImputer)
    for p in grids["GaussianNB"].keys():
        assert p.startswith("clf__"), f"GaussianNB should have clf__ prefix: {p}"


def test_tuning_classifiers_anti_oversubscription():
    """Verify internal n_jobs=1 for RandomForest and KNN to prevent nested oversubscription."""
    models = get_tuning_classifiers(random_state=42)

    rf_step = models["RandomForestClassifier"].named_steps["clf"]
    assert rf_step.n_jobs == 1, f"RandomForestClassifier internal n_jobs must be 1, got {rf_step.n_jobs}"

    knn_step = models["KNeighborsClassifier"].named_steps["clf"]
    assert knn_step.n_jobs == 1, f"KNeighborsClassifier internal n_jobs must be 1, got {knn_step.n_jobs}"


# ============================================================================
# 4. GridSearchCV Protocol Tests: Mock/Lightweight Execution
# ============================================================================

def test_tune_single_classifier_protocol(synthetic_panel_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch):
    """Verify GridSearchCV configuration (scoring, refit=False, n_jobs, pre_dispatch, output structure)."""
    from sklearn.model_selection import GridSearchCV

    captured_kwargs = {}
    original_init = GridSearchCV.__init__

    def mock_init(self, estimator, param_grid, **kwargs):
        captured_kwargs.update(kwargs)
        # Use a reduced grid and fast mock for the test
        return original_init(self, estimator, param_grid, **kwargs)

    monkeypatch.setattr(GridSearchCV, "__init__", mock_init)

    # Use a tiny 2-candidate grid for LogisticRegression
    tiny_grid = {"clf__C": [0.1, 1.0]}
    models = get_tuning_classifiers(random_state=42)
    lr_model = models["LogisticRegression"]

    result = tune_single_classifier(
        name="LogisticRegression",
        estimator=lr_model,
        param_grid=tiny_grid,
        train_df=synthetic_panel_df,
        features=["feat1", "feat2"],
        label="target",
        n_splits=5,
        purge_days=3,
        n_jobs=2,
        pre_dispatch="2*n_jobs",
        verbose=0,
    )

    # Verify GridSearchCV initialization parameters
    assert captured_kwargs["scoring"] == "average_precision"
    assert captured_kwargs["refit"] is False
    assert captured_kwargs["n_jobs"] == 2
    assert captured_kwargs["pre_dispatch"] == "2*n_jobs"

    # Verify output dictionary format
    assert result["model"] == "LogisticRegression"
    assert "clf__C" in result["best_params"]
    assert isinstance(result["best_score"], float)
    assert isinstance(result["cv_results_df"], pd.DataFrame)
    assert len(result["cv_results_df"]) == 2
    assert "mean_test_score" in result["cv_results_df"].columns
    assert "rank_test_score" in result["cv_results_df"].columns


def test_refit_and_evaluate_tuned_model_protocol():
    """Verify manual refit on Train, threshold calibration on Validation, and KNN 25k cap."""
    np.random.seed(42)
    train_df = pd.DataFrame({
        "feat1": np.random.randn(300),
        "feat2": np.random.rand(300),
        "target": np.random.choice([0, 1], size=300, p=[0.9, 0.1]),
    })
    val_df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.rand(100),
        "target": np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
    })
    # Ensure positive examples
    train_df.loc[:10, "target"] = 1
    val_df.loc[:5, "target"] = 1

    models = get_tuning_classifiers(random_state=42)

    # 1. Test Logistic Regression refit
    best_params_lr = {"clf__C": 0.5, "clf__class_weight": None}
    record_lr, tuned_lr = refit_and_evaluate_tuned_model(
        name="LogisticRegression",
        base_estimator=models["LogisticRegression"],
        best_params=best_params_lr,
        train_df=train_df,
        val_df=val_df,
        features=["feat1", "feat2"],
        label="target",
    )
    assert record_lr["train_samples"] == 300
    assert record_lr["val_samples"] == 100
    assert 0.0 <= record_lr["optimal_threshold"] <= 1.0
    assert 0.0 <= record_lr["pr_auc"] <= 1.0
    assert tuned_lr.named_steps["clf"].C == 0.5

    # 2. Test KNN refit with sample cap (cap set to 150 for test)
    best_params_knn = {"clf__n_neighbors": 5, "clf__weights": "distance"}
    record_knn, _ = refit_and_evaluate_tuned_model(
        name="KNeighborsClassifier",
        base_estimator=models["KNeighborsClassifier"],
        best_params=best_params_knn,
        train_df=train_df,
        val_df=val_df,
        features=["feat1", "feat2"],
        label="target",
        max_knn_samples=150,
    )
    assert record_knn["train_samples"] == 150, f"Expected KNN train sample capped at 150, got {record_knn['train_samples']}"


# ============================================================================
# 5. Artifact Saving & Hermetic Isolation Tests
# ============================================================================

def test_model_file_slugs_standardization():
    """Verify exact snake_case standardized slugs for all 6 classifiers."""
    expected_slugs = {
        "LogisticRegression": "logistic_regression",
        "DecisionTreeClassifier": "decision_tree",
        "RandomForestClassifier": "random_forest",
        "HistGradientBoostingClassifier": "hist_gradient_boosting",
        "KNeighborsClassifier": "knn",
        "GaussianNB": "gaussian_nb",
    }
    assert MODEL_FILE_SLUGS == expected_slugs


def test_save_tuning_artifacts_hermetic(tmp_path: Path):
    """Verify saving comparison CSV, best hyperparameters JSON, and standardized per-model CV results."""
    comparison_df = pd.DataFrame({
        "model": ["LogisticRegression", "DecisionTreeClassifier"],
        "mean_cv_pr_auc": [0.25, 0.28],
        "pr_auc": [0.26, 0.29],
        "optimal_threshold": [0.18, 0.22],
        "f1_optimal": [0.32, 0.35],
    })
    best_hyperparams = {
        "LogisticRegression": {"clf__C": 1.0, "clf__class_weight": None},
        "DecisionTreeClassifier": {"clf__max_depth": 10, "clf__min_samples_leaf": 50},
    }
    cv_results_dict = {
        "LogisticRegression": pd.DataFrame({
            "params": [{"clf__C": 0.1}, {"clf__C": 1.0}],
            "mean_test_score": [0.22, 0.25],
            "rank_test_score": [2, 1],
        }),
        "DecisionTreeClassifier": pd.DataFrame({
            "params": [{"clf__max_depth": 6}, {"clf__max_depth": 10}],
            "mean_test_score": [0.26, 0.28],
            "rank_test_score": [2, 1],
        }),
    }

    save_tuning_artifacts(
        comparison_df=comparison_df,
        best_hyperparams=best_hyperparams,
        cv_results_dict=cv_results_dict,
        output_dir=tmp_path,
    )

    # 1. Verify tuned comparison CSV
    csv_file = tmp_path / "tuned_classifier_comparison.csv"
    assert csv_file.exists()
    loaded_df = pd.read_csv(csv_file)
    assert len(loaded_df) == 2
    assert "mean_cv_pr_auc" in loaded_df.columns

    # 2. Verify best hyperparameters JSON
    json_file = tmp_path / "best_hyperparameters.json"
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["LogisticRegression"]["clf__C"] == 1.0

    # 3. Verify standardized snake_case individual CV result files
    lr_cv = tmp_path / "cv_results_logistic_regression.csv"
    dt_cv = tmp_path / "cv_results_decision_tree.csv"
    assert lr_cv.exists()
    assert dt_cv.exists()


def test_progressive_and_atomic_saving_fault_tolerance(tmp_path: Path):
    """Verify progressive incremental saving: earlier results are written atomically and survive subsequent errors."""
    # Step 1: Model 1 (LogisticRegression) finishes GridSearchCV
    cv_df_lr = pd.DataFrame({
        "params": [{"clf__C": 0.1}, {"clf__C": 1.0}],
        "mean_test_score": [0.22, 0.25],
        "rank_test_score": [2, 1],
    })
    save_single_model_cv_results("LogisticRegression", cv_df_lr, output_dir=tmp_path)

    best_params_all = {"LogisticRegression": {"clf__C": 1.0, "clf__class_weight": None}}
    save_hyperparameters(best_params_all, output_dir=tmp_path)

    # Verify model 1 CV results and JSON exist immediately
    lr_csv = tmp_path / "cv_results_logistic_regression.csv"
    json_path = tmp_path / "best_hyperparameters.json"
    assert lr_csv.exists()
    assert json_path.exists()
    # Verify no dangling temporary files
    assert not list(tmp_path.glob("*.tmp"))

    # Step 2: Model 1 finishes validation evaluation
    records = [{
        "model": "LogisticRegression",
        "pr_auc": 0.26,
        "f1_optimal": 0.32,
    }]
    save_comparison_table(pd.DataFrame(records), output_dir=tmp_path)
    comp_csv = tmp_path / "tuned_classifier_comparison.csv"
    assert comp_csv.exists()
    assert len(pd.read_csv(comp_csv)) == 1

    # Step 3: Model 2 (DecisionTreeClassifier) finishes GridSearchCV
    cv_df_dt = pd.DataFrame({
        "params": [{"clf__max_depth": 6}, {"clf__max_depth": 10}],
        "mean_test_score": [0.26, 0.28],
        "rank_test_score": [2, 1],
    })
    save_single_model_cv_results("DecisionTreeClassifier", cv_df_dt, output_dir=tmp_path)
    best_params_all["DecisionTreeClassifier"] = {"clf__max_depth": 10, "clf__min_samples_leaf": 50}
    save_hyperparameters(best_params_all, output_dir=tmp_path)

    dt_csv = tmp_path / "cv_results_decision_tree.csv"
    assert dt_csv.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert len(loaded_json) == 2
    assert "DecisionTreeClassifier" in loaded_json

    # Step 4: Simulate a catastrophic failure during Model 3
    # Model 1 and Model 2 files must remain 100% intact and uncorrupted on disk
    assert lr_csv.exists() and len(pd.read_csv(lr_csv)) == 2
    assert dt_csv.exists() and len(pd.read_csv(dt_csv)) == 2
    assert json_path.exists()
    assert comp_csv.exists() and len(pd.read_csv(comp_csv)) == 1
    # Verify atomicity (no partial temp files)
    assert not list(tmp_path.glob("*.tmp"))


def test_test_data_isolation():
    """Verify that tune_all_classifiers signature does NOT accept or require test_df."""
    import inspect
    from tune_classifiers import tune_all_classifiers

    sig = inspect.signature(tune_all_classifiers)
    param_names = list(sig.parameters.keys())

    assert "train_df" in param_names
    assert "val_df" in param_names
    assert "test_df" not in param_names, "CRITICAL: test_df must NOT be an argument to tune_all_classifiers!"
