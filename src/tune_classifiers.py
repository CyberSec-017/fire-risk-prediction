"""Hyperparameter optimization for wildfire risk prediction using temporal GridSearchCV (K=5).

Methodological guarantees:
  1. Temporal 5-fold cross-validation constructed chronologically from unique dates.
  2. Strict 3-day purge gap (H=3, date < t_min - 3 days) between train and validation fold.
  3. Cell atomicity: all 209 spatial cells for any given calendar date remain strictly together.
  4. Validation fold integrity: 100% of validation dates and observations are preserved in every fold.
  5. Training subsampling: applied strictly to the training indices of each fold (RF capped at 60k, KNN at 20k)
     using stratified sampling preserving natural class balance (~5.56% fire rate).
  6. Metric: average_precision (PR-AUC) as primary optimization objective for rare-event imbalance.
  7. Refit protocol: refit=False in GridSearchCV; best estimators are manually refitted on full Train
     (with KNN capped at 25k samples).
  8. Anti-oversubscription: internal estimators set to n_jobs=1; GridSearchCV parallelized with n_jobs=2.
  9. Decision threshold: optimal F1 threshold theta* calibrated on Validation (2022) post-tuning.
  10. Test set (2023-2024) isolation: strictly untouched during tuning, refit, and threshold selection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, train_test_split

# Ensure src/ is in sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from compare_classifiers import (
    KNN_MAX_TRAIN_SAMPLES,
    PURGE_DAYS,
    RESULTS_DIR,
    evaluate_predictions,
    find_optimal_threshold_oof,
    get_classifiers,
    predict_proba_batched,
    subsample_if_needed,
)
from train_baseline import EXCLUDE_COLS, LABEL, load_data, time_split

# Subsampling caps for tuning folds (applied strictly to fold training indices)
RF_MAX_SEARCH_TRAIN_SAMPLES = 60000
KNN_MAX_SEARCH_TRAIN_SAMPLES = 20000

# Centralized standardized file slugs for detailed CSV artifacts
MODEL_FILE_SLUGS: dict[str, str] = {
    "LogisticRegression": "logistic_regression",
    "DecisionTreeClassifier": "decision_tree",
    "RandomForestClassifier": "random_forest",
    "HistGradientBoostingClassifier": "hist_gradient_boosting",
    "KNeighborsClassifier": "knn",
    "GaussianNB": "gaussian_nb",
}


def get_tuning_classifiers(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Instantiate the 6 classifiers configured for GridSearchCV without nested parallelism.

    Sets internal n_jobs=1 for RandomForest and KNN to avoid thread oversubscription
    when GridSearchCV coordinates worker processes.
    """
    models = get_classifiers(random_state=random_state)
    models["RandomForestClassifier"].set_params(clf__n_jobs=1)
    models["KNeighborsClassifier"].set_params(clf__n_jobs=1)
    return models


def get_param_grids() -> dict[str, dict[str, list[Any]]]:
    """Define targeted hyperparameter search grids for each of the 6 classifiers.

    Parameter names strictly match estimator introspection via get_params(deep=True):
      - Pipeline-wrapped models use the 'clf__' prefix.
      - HistGradientBoostingClassifier has no prefix (direct estimator).
    """
    return {
        "LogisticRegression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        "DecisionTreeClassifier": {
            "clf__max_depth": [6, 10, 15],
            "clf__min_samples_leaf": [20, 50, 100],
            "clf__class_weight": [None, "balanced"],
        },
        "RandomForestClassifier": {
            "clf__max_depth": [8, 12, 16],
            "clf__min_samples_leaf": [20, 50],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "HistGradientBoostingClassifier": {
            "learning_rate": [0.03, 0.08, 0.15],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [50, 100],
            "l2_regularization": [0.1, 1.0],
        },
        "KNeighborsClassifier": {
            "clf__n_neighbors": [7, 15],
            "clf__weights": ["uniform", "distance"],
        },
        "GaussianNB": {
            "clf__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
        },
    }


def build_tuning_date_folds(
    train_df: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    max_train_samples: int | None = None,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build chronological expanding-window folds grouped by unique dates for GridSearchCV.

    Guarantees:
      1. Every calendar date's cells remain atomic within the same fold.
      2. Strict temporal purge: training dates strictly precede (t_min - purge_days).
      3. 100% of validation dates and observations are preserved in every fold.
      4. Subsampling (if max_train_samples is set) is applied strictly to training indices,
         stratified on class labels, preserving the natural positive fire rate.
      5. Both classes (0 and 1) are verified to be present in both train and val folds.
      6. Output indices are integer row positions in [0, len(train_df) - 1], exactly aligned
         with the full training dataset passed to GridSearchCV.
    """
    unique_dates = np.sort(train_df["date"].drop_duplicates().to_numpy())
    if len(unique_dates) < (n_splits + 1) * (purge_days + 2):
        raise ValueError("Insufficient unique dates to create expanding folds with purge.")

    date_blocks = np.array_split(unique_dates, n_splits + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    y_series = pd.Series(y_train).reset_index(drop=True)

    date_to_indices = train_df.groupby("date").indices

    for k in range(1, n_splits + 1):
        val_block_dates = date_blocks[k]
        t_min = pd.Timestamp(val_block_dates.min())
        purge_cutoff = t_min - pd.Timedelta(days=purge_days)

        train_block_dates = unique_dates[unique_dates < purge_cutoff]
        if len(train_block_dates) == 0:
            raise ValueError(
                f"Fold {k}: No training dates remain after applying {purge_days}-day purge."
            )

        val_idx_list = [date_to_indices[d] for d in val_block_dates if d in date_to_indices]
        raw_train_idx_list = [date_to_indices[d] for d in train_block_dates if d in date_to_indices]

        val_idx = np.concatenate(val_idx_list) if val_idx_list else np.array([], dtype=int)
        raw_train_idx = np.concatenate(raw_train_idx_list) if raw_train_idx_list else np.array([], dtype=int)

        # Verify class distribution in validation fold
        y_val_fold = y_series.iloc[val_idx]
        if y_val_fold.nunique() < 2:
            raise ValueError(
                f"Fold {k}: Validation fold lacks one of the binary classes. "
                f"Unique values: {y_val_fold.unique()}"
            )

        # Apply stratified subsampling strictly to training indices if requested
        if max_train_samples is not None and len(raw_train_idx) > max_train_samples:
            y_train_raw = y_series.iloc[raw_train_idx]
            sub_train_idx, _ = train_test_split(
                raw_train_idx,
                train_size=max_train_samples,
                stratify=y_train_raw,
                random_state=random_state + k,
            )
            fold_train_idx = np.sort(sub_train_idx)
        else:
            fold_train_idx = np.sort(raw_train_idx)

        # Verify class distribution in training fold
        y_train_fold = y_series.iloc[fold_train_idx]
        if y_train_fold.nunique() < 2:
            raise ValueError(
                f"Fold {k}: Training fold lacks one of the binary classes. "
                f"Unique values: {y_train_fold.unique()}"
            )

        folds.append((fold_train_idx, val_idx))

    return folds


def tune_single_classifier(
    name: str,
    estimator: BaseEstimator,
    param_grid: dict[str, list[Any]],
    train_df: pd.DataFrame,
    features: list[str],
    label: str = LABEL,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    n_jobs: int = 2,
    pre_dispatch: str = "2*n_jobs",
    random_state: int = 42,
    verbose: int = 1,
) -> dict[str, Any]:
    """Execute temporal GridSearchCV for a single classifier using refit=False.

    Returns dictionary containing best_params, best_score (mean CV PR-AUC),
    cv_results DataFrame, and search runtime.
    """
    X_train = train_df[features]
    y_train = train_df[label]

    # Select appropriate training subsample cap for the model family
    if name == "RandomForestClassifier":
        max_train_samples = RF_MAX_SEARCH_TRAIN_SAMPLES
    elif name == "KNeighborsClassifier":
        max_train_samples = KNN_MAX_SEARCH_TRAIN_SAMPLES
    else:
        max_train_samples = None

    folds = build_tuning_date_folds(
        train_df=train_df,
        y_train=y_train,
        n_splits=n_splits,
        purge_days=purge_days,
        max_train_samples=max_train_samples,
        random_state=random_state,
    )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="average_precision",
        cv=folds,
        refit=False,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    tuning_time = time.perf_counter() - t0

    cv_results_df = pd.DataFrame(grid.cv_results_)

    return {
        "model": name,
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "cv_results_df": cv_results_df,
        "tuning_time_sec": round(tuning_time, 2),
        "n_candidates": len(grid.cv_results_["params"]),
        "n_folds": n_splits,
        "max_train_samples": max_train_samples if max_train_samples else len(train_df),
    }


def refit_and_evaluate_tuned_model(
    name: str,
    base_estimator: BaseEstimator,
    best_params: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    label: str = LABEL,
    max_knn_samples: int = KNN_MAX_TRAIN_SAMPLES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Refit the optimized estimator on the Train set and evaluate on the Validation set.

    Strict protocol:
      1. Instantiates tuned estimator with best_params.
      2. Refits on full Train (or 25k stratified sample for KNN).
      3. Predicts positive-class probabilities on Validation set (2022).
      4. Calibrates optimal decision threshold theta* on Validation predictions to maximize F1.
      5. Freezes theta* and computes comprehensive performance metrics.
    """
    X_train = train_df[features]
    y_train = train_df[label]
    X_val = val_df[features]
    y_val = val_df[label]

    # Instantiate and configure model with optimal hyperparameters
    tuned_model = clone(base_estimator)
    tuned_model.set_params(**best_params)

    # Re-enable multi-threading for the standalone final fit if applicable
    if name == "RandomForestClassifier" and "clf" in getattr(tuned_model, "named_steps", {}):
        tuned_model.set_params(clf__n_jobs=-1)
    elif name == "KNeighborsClassifier" and "clf" in getattr(tuned_model, "named_steps", {}):
        tuned_model.set_params(clf__n_jobs=-1)

    # Apply final training cap for KNN
    if name == "KNeighborsClassifier":
        X_tr_fit, y_tr_fit = subsample_if_needed(
            X_train, y_train,
            max_samples=max_knn_samples,
            random_state=random_state,
        )
    else:
        X_tr_fit, y_tr_fit = X_train, y_train

    # Final fit
    t0_fit = time.perf_counter()
    tuned_model.fit(X_tr_fit, y_tr_fit)
    fit_time = time.perf_counter() - t0_fit

    # Validation inference
    t0_eval = time.perf_counter()
    p_val = predict_proba_batched(tuned_model, X_val, batch_size=15000)
    eval_time = time.perf_counter() - t0_eval

    # Calibrate optimal decision threshold on Validation (F1-maximizing)
    optimal_thresh = find_optimal_threshold_oof(y_val, p_val)

    # Evaluate comprehensive metrics on Validation
    metrics = evaluate_predictions(y_val, p_val, optimal_threshold=optimal_thresh)

    record: dict[str, Any] = {
        "model": name,
        "train_samples": len(X_tr_fit),
        "val_samples": len(X_val),
        "fit_time_sec": round(fit_time, 2),
        "eval_time_sec": round(eval_time, 2),
        **metrics,
    }
    return record, tuned_model


def _atomic_write_df(df: pd.DataFrame, target_path: Path) -> None:
    """Write DataFrame atomically using a temporary file in the same directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, target_path)


def _atomic_write_json(data: dict[str, Any], target_path: Path) -> None:
    """Write dictionary to JSON atomically using a temporary file in the same directory."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, target_path)


def save_single_model_cv_results(
    model_name: str,
    cv_df: pd.DataFrame,
    output_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Save individual model CV results atomically with standardized slug filename."""
    out_path = Path(output_dir)
    slug = MODEL_FILE_SLUGS.get(model_name, model_name.lower())
    target_csv = out_path / f"cv_results_{slug}.csv"
    sorted_df = cv_df.sort_values(by="rank_test_score")
    _atomic_write_df(sorted_df, target_csv)
    return target_csv


def save_hyperparameters(
    best_hyperparams: dict[str, dict[str, Any]],
    output_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Save best hyperparameters dictionary atomically to JSON."""
    out_path = Path(output_dir)
    target_json = out_path / "best_hyperparameters.json"
    _atomic_write_json(best_hyperparams, target_json)
    return target_json


def save_comparison_table(
    comparison_df: pd.DataFrame,
    output_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Save cumulative comparison DataFrame atomically to CSV."""
    out_path = Path(output_dir)
    target_csv = out_path / "tuned_classifier_comparison.csv"
    _atomic_write_df(comparison_df, target_csv)
    return target_csv


def save_tuning_artifacts(
    comparison_df: pd.DataFrame,
    best_hyperparams: dict[str, dict[str, Any]],
    cv_results_dict: dict[str, pd.DataFrame],
    output_dir: Path | str = RESULTS_DIR,
) -> None:
    """Save tuning outputs (comparison CSV, hyperparameters JSON, CV detail CSVs) cleanly and atomically."""
    save_comparison_table(comparison_df, output_dir=output_dir)
    save_hyperparameters(best_hyperparams, output_dir=output_dir)
    for model_name, cv_df in cv_results_dict.items():
        save_single_model_cv_results(model_name, cv_df, output_dir=output_dir)


def tune_all_classifiers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    label: str = LABEL,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    n_jobs: int = 2,
    pre_dispatch: str = "2*n_jobs",
    random_state: int = 42,
    output_dir: Path | str = RESULTS_DIR,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Orchestrate hyperparameter optimization with progressive fault-tolerant saving."""
    classifiers = get_tuning_classifiers(random_state=random_state)
    param_grids = get_param_grids()

    comparison_records: list[dict[str, Any]] = []
    best_params_all: dict[str, dict[str, Any]] = {}
    cv_results_all: dict[str, pd.DataFrame] = {}

    print(f"Starting 5-fold temporal GridSearchCV tuning for {len(classifiers)} classifiers...")
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Features: {len(features)}")
    print(f"  Parallel configuration: n_jobs={n_jobs}, pre_dispatch='{pre_dispatch}'")

    for name, model in classifiers.items():
        grid = param_grids[name]
        n_combos = np.prod([len(v) for v in grid.values()])
        print(f"\n--- Tuning {name} ({n_combos} combinations x {n_splits} folds = {n_combos * n_splits} fits) ---")

        # 1. Run GridSearchCV
        try:
            tune_res = tune_single_classifier(
                name=name,
                estimator=model,
                param_grid=grid,
                train_df=train_df,
                features=features,
                label=label,
                n_splits=n_splits,
                purge_days=purge_days,
                n_jobs=n_jobs,
                pre_dispatch=pre_dispatch,
                random_state=random_state,
            )
        except Exception as exc:
            print(f"  ERROR during GridSearchCV tuning of {name}: {exc}")
            raise

        best_params = tune_res["best_params"]
        best_cv_score = tune_res["best_score"]
        tune_time = tune_res["tuning_time_sec"]
        cv_df = tune_res["cv_results_df"]

        cv_results_all[name] = cv_df
        best_params_all[name] = best_params

        # Step 2A: Progressive save immediately after GridSearchCV finishes
        save_single_model_cv_results(name, cv_df, output_dir=output_dir)
        save_hyperparameters(best_params_all, output_dir=output_dir)

        print(f"  Optimal parameters found in {tune_time:.1f}s (Mean CV PR-AUC: {best_cv_score:.4f}):")
        for p_name, p_val in best_params.items():
            print(f"    - {p_name}: {p_val}")

        # 2. Refit on full Train and evaluate on Validation
        print(f"  Refitting {name} on Train and evaluating on Validation 2022...", end=" ", flush=True)
        try:
            eval_record, _ = refit_and_evaluate_tuned_model(
                name=name,
                base_estimator=model,
                best_params=best_params,
                train_df=train_df,
                val_df=val_df,
                features=features,
                label=label,
                max_knn_samples=KNN_MAX_TRAIN_SAMPLES,
                random_state=random_state,
            )
        except Exception as exc:
            print(f"  ERROR during refit/evaluation of {name}: {exc}")
            raise

        eval_record["tuning_time_sec"] = tune_time
        eval_record["mean_cv_pr_auc"] = round(best_cv_score, 4)
        comparison_records.append(eval_record)

        # Step 2B: Progressive save of comparison table immediately after refit & evaluation
        save_comparison_table(pd.DataFrame(comparison_records), output_dir=output_dir)

        print(f"Done (Val PR-AUC: {eval_record['pr_auc']:.4f}, F1*: {eval_record['f1_optimal']:.4f})")

    comparison_df = pd.DataFrame(comparison_records)
    return comparison_df, best_params_all


def main():
    """Main execution point for hyperparameter optimization."""
    print("Loading fire panel dataset...")
    df, features = load_data()
    train, val, _ = time_split(df)

    res_df, _ = tune_all_classifiers(
        train_df=train,
        val_df=val,
        features=features,
        label=LABEL,
        n_splits=5,
        purge_days=PURGE_DAYS,
        n_jobs=2,
        pre_dispatch="2*n_jobs",
        random_state=42,
    )

    print("\n" + "=" * 105)
    print("TUNED CLASSIFIER COMPARISON RESULTS (VALIDATION SET 2022)")
    print("=" * 105)
    cols_display = [
        "model", "mean_cv_pr_auc", "pr_auc", "roc_auc", "brier_score",
        "optimal_threshold", "f1_optimal", "recall_optimal", "precision_optimal",
        "tuning_time_sec", "fit_time_sec",
    ]
    print(res_df[cols_display].to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    main()
