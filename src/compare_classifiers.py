"""Benchmark and comparison of 6 classical Machine Learning classifiers for wildfire risk prediction.

Evaluates 6 classical algorithms without calibration:
  1. LogisticRegression (with StandardScaler)
  2. DecisionTreeClassifier
  3. RandomForestClassifier
  4. HistGradientBoostingClassifier
  5. KNeighborsClassifier (with StandardScaler and stratified subsampling)
  6. GaussianNB

Methodological guarantees:
  - Folds constructed chronologically by unique dates on TRAIN (expanding window).
  - All 209 cells of a calendar day remain indivisible in the same fold.
  - Strict 3-day temporal purge (H=3, d < t_min - 3 days) between train and validation fold.
  - Initial Block 0 acts as honest warm-up without OOF prediction (NaN).
  - Decision threshold theta* learned strictly from Train OOF predictions
    via precision_recall_curve (Validation never participates).
  - Out-of-sample evaluation performed strictly on VALIDATION (2022).
  - TEST set (2023-2024) remains completely untouched.
  - Production model artifacts (models/fire_model.joblib, model_card.json) are never modified.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from train_baseline import EXCLUDE_COLS, LABEL, load_data, time_split

RESULTS_DIR = ROOT / "results"
KNN_MAX_TRAIN_SAMPLES = 25000
PURGE_DAYS = config.H  # 3 days


def get_classifiers(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Instantiate the 6 uncalibrated classical classifiers with standard pipelines.

    Scalers are included where mathematically required (Logistic Regression and KNN).
    Median imputation is included for estimators that do not handle NaNs natively (GWETROOT).
    HistGradientBoostingClassifier natively supports NaNs and requires no imputer.
    """
    return {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "DecisionTreeClassifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=50,
                random_state=random_state,
            )),
        ]),
        "RandomForestClassifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=30,
                max_samples=0.5,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            min_samples_leaf=100,
            learning_rate=0.08,
            random_state=random_state,
        ),
        "KNeighborsClassifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                algorithm="ball_tree",
                n_jobs=-1,
            )),
        ]),
        "GaussianNB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GaussianNB(
                var_smoothing=1e-9,
            )),
        ]),
    }


def build_date_folds(
    train_df: pd.DataFrame,
    n_splits: int = 3,
    purge_days: int = PURGE_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build chronological expanding-window folds grouped by unique dates.

    Guarantees:
      1. Every calendar date's cells are strictly within the same fold.
      2. Strict purge: train_dates < t_min - purge_days (exactly 3 full days excluded).
      3. Block 0 serves as warm-up; folds 1..n_splits generate OOF predictions.
    """
    unique_dates = np.sort(train_df["date"].drop_duplicates().to_numpy())
    if len(unique_dates) < (n_splits + 1) * (purge_days + 2):
        raise ValueError("Insufficient unique dates to create expanding folds with purge.")

    date_blocks = np.array_split(unique_dates, n_splits + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    # Map dates to row indices in train_df
    date_to_indices = train_df.groupby("date").indices

    for k in range(1, n_splits + 1):
        val_block_dates = date_blocks[k]
        t_min = pd.Timestamp(val_block_dates.min())
        purge_cutoff = t_min - pd.Timedelta(days=purge_days)

        train_block_dates = unique_dates[unique_dates < purge_cutoff]
        if len(train_block_dates) == 0:
            raise ValueError(f"Fold {k}: No training dates remain after applying {purge_days}-day purge.")

        train_idx_list = [date_to_indices[d] for d in train_block_dates if d in date_to_indices]
        val_idx_list = [date_to_indices[d] for d in val_block_dates if d in date_to_indices]

        train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=int)
        val_idx = np.concatenate(val_idx_list) if val_idx_list else np.array([], dtype=int)

        folds.append((train_idx, val_idx))

    return folds


def subsample_if_needed(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    max_samples: int | None = None,
    random_state: int = 42,
) -> tuple[Any, Any]:
    """Perform reproducible stratified subsampling if the dataset exceeds max_samples."""
    n_obs = len(X)
    if max_samples is None or n_obs <= max_samples:
        return X, y

    # Perform stratified split to maintain positive class base rate
    if isinstance(X, pd.DataFrame):
        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y,
            random_state=random_state,
        )
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y,
            random_state=random_state,
        )
    return X_sub, y_sub


def predict_proba_batched(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    batch_size: int = 15000,
) -> np.ndarray:
    """Predict positive-class probability in memory-friendly batches."""
    n_samples = len(X)
    if n_samples <= batch_size:
        return model.predict_proba(X)[:, 1]

    probas = np.zeros(n_samples, dtype=float)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X.iloc[start:end] if isinstance(X, pd.DataFrame) else X[start:end]
        probas[start:end] = model.predict_proba(batch)[:, 1]
    return probas


def compute_oof_predictions(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    is_knn: bool = False,
    max_knn_samples: int = KNN_MAX_TRAIN_SAMPLES,
    random_state: int = 42,
) -> np.ndarray:
    """Compute out-of-fold predicted probabilities strictly on the Train set.

    Block 0 observations remain NaN (honest warm-up accounting).
    """
    p_oof = np.full(len(X_train), fill_value=np.nan, dtype=float)

    for fold_i, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr_f = X_train.iloc[tr_idx]
        y_tr_f = y_train.iloc[tr_idx]
        X_va_f = X_train.iloc[val_idx]

        if is_knn:
            X_tr_f, y_tr_f = subsample_if_needed(
                X_tr_f, y_tr_f,
                max_samples=max_knn_samples,
                random_state=random_state,
            )

        fold_model = clone(model)
        fold_model.fit(X_tr_f, y_tr_f)
        p_oof[val_idx] = predict_proba_batched(fold_model, X_va_f)

    return p_oof


def find_optimal_threshold_oof(
    y_train: np.ndarray | pd.Series,
    p_oof: np.ndarray,
) -> float:
    """Derive the optimal F1 threshold from continuous OOF predictions using precision_recall_curve.

    Excludes unpredicted Block 0 observations (~np.isnan(p_oof)).
    Applies median rule in case of tied maximum F1 scores.
    """
    y_arr = np.asarray(y_train)
    valid_mask = ~np.isnan(p_oof)

    if not np.any(valid_mask):
        return 0.5

    y_valid = y_arr[valid_mask]
    p_valid = p_oof[valid_mask]

    precisions, recalls, thresholds = precision_recall_curve(y_valid, p_valid)
    if len(thresholds) == 0:
        return 0.5

    p_cand = precisions[:-1]
    r_cand = recalls[:-1]

    denom = p_cand + r_cand
    f1_scores = np.zeros_like(denom, dtype=float)
    nonzero_mask = denom > 0
    f1_scores[nonzero_mask] = (
        2 * (p_cand[nonzero_mask] * r_cand[nonzero_mask]) / denom[nonzero_mask]
    )

    max_f1 = np.max(f1_scores)
    if max_f1 <= 0:
        return 0.5

    # 1. Identify all indices achieving the maximal F1 (robust to floating point precision)
    best_indices = np.where(np.isclose(f1_scores, max_f1, rtol=1e-7, atol=1e-9))[0]

    # 2. Sort candidate indices (thresholds are monotonically increasing in precision_recall_curve)
    best_indices = np.sort(best_indices)

    # 3. In case of ties, select the candidate located at the middle index of the sorted list
    # (avoid numerical median interpolation which could produce a non-candidate with sub-optimal F1)
    selected_index = int(best_indices[len(best_indices) // 2])

    # 4. Guarantee that the returned threshold strictly belongs to candidate thresholds
    optimal_threshold = float(thresholds[selected_index])

    return optimal_threshold


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    probas: np.ndarray,
    optimal_threshold: float,
) -> dict[str, float]:
    """Compute threshold-free and thresholded performance metrics on evaluation set."""
    y_arr = np.asarray(y_true).astype(int)
    probas = np.clip(np.asarray(probas, dtype=float), 0.0, 1.0)

    # Threshold-independent metrics
    roc_auc = float(roc_auc_score(y_arr, probas))
    pr_auc = float(average_precision_score(y_arr, probas))
    brier = float(brier_score_loss(y_arr, probas))

    # Evaluation at optimal threshold (derived from OOF)
    y_pred_opt = (probas >= optimal_threshold).astype(int)
    f1_opt = float(f1_score(y_arr, y_pred_opt, zero_division=0))
    prec_opt = float(precision_score(y_arr, y_pred_opt, zero_division=0))
    rec_opt = float(recall_score(y_arr, y_pred_opt, zero_division=0))

    # Evaluation at standard default threshold 0.5
    y_pred_05 = (probas >= 0.5).astype(int)
    f1_05 = float(f1_score(y_arr, y_pred_05, zero_division=0))
    prec_05 = float(precision_score(y_arr, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_arr, y_pred_05, zero_division=0))
    acc_05 = float(accuracy_score(y_arr, y_pred_05))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "f1_optimal": round(f1_opt, 4),
        "precision_optimal": round(prec_opt, 4),
        "recall_optimal": round(rec_opt, 4),
        "f1_default_05": round(f1_05, 4),
        "precision_default_05": round(prec_05, 4),
        "recall_default_05": round(rec_05, 4),
        "accuracy_default_05": round(acc_05, 4),
    }


def fit_and_evaluate_classifier(
    name: str,
    model: BaseEstimator,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    label: str = LABEL,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    is_knn: bool = False,
    max_knn_samples: int = KNN_MAX_TRAIN_SAMPLES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a classifier, learn its threshold on OOF, and score on Validation."""
    X_train = train_df[features]
    y_train = train_df[label]
    X_val = val_df[features]
    y_val = val_df[label]

    if folds is None:
        folds = build_date_folds(train_df, n_splits=3, purge_days=PURGE_DAYS)

    # 1. Compute Train Out-Of-Fold predictions
    t0_oof = time.perf_counter()
    p_oof = compute_oof_predictions(
        model,
        X_train,
        y_train,
        folds,
        is_knn=is_knn,
        max_knn_samples=max_knn_samples,
        random_state=random_state,
    )
    oof_time = time.perf_counter() - t0_oof

    # 2. Select optimal threshold theta* strictly from OOF
    optimal_thresh = find_optimal_threshold_oof(y_train, p_oof)

    # 3. Fit final model on full train set (or subsampled train for KNN)
    final_model = clone(model)
    X_tr_final, y_tr_final = X_train, y_train
    if is_knn:
        X_tr_final, y_tr_final = subsample_if_needed(
            X_train, y_train,
            max_samples=max_knn_samples,
            random_state=random_state,
        )

    t0_fit = time.perf_counter()
    final_model.fit(X_tr_final, y_tr_final)
    fit_time = time.perf_counter() - t0_fit

    # 4. Predict on Validation set
    t0_eval = time.perf_counter()
    p_val = predict_proba_batched(final_model, X_val)
    eval_time = time.perf_counter() - t0_eval

    # 5. Evaluate all metrics
    metrics = evaluate_predictions(y_val, p_val, optimal_threshold=optimal_thresh)

    record: dict[str, Any] = {
        "model": name,
        "train_samples": len(X_tr_final),
        "val_samples": len(X_val),
        "oof_time_sec": round(oof_time, 2),
        "fit_time_sec": round(fit_time, 2),
        "eval_time_sec": round(eval_time, 2),
        **metrics,
    }
    return record


def save_comparison_results(
    df: pd.DataFrame,
    output_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Save the classifier comparison DataFrame to CSV in the target directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_csv = out_path / "classifier_comparison.csv"
    df.to_csv(target_csv, index=False)
    return target_csv


def compare_all_classifiers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    label: str = LABEL,
    random_state: int = 42,
    max_knn_samples: int = KNN_MAX_TRAIN_SAMPLES,
    output_dir: Path | str = RESULTS_DIR,
) -> pd.DataFrame:
    """Orchestrate evaluation across all 6 classical classifiers."""
    classifiers = get_classifiers(random_state=random_state)
    folds = build_date_folds(train_df, n_splits=3, purge_days=PURGE_DAYS)

    records = []
    print(f"Comparing {len(classifiers)} classical classifiers...")
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | {len(features)} features")

    for name, model in classifiers.items():
        print(f"  Evaluating {name} ...", end=" ", flush=True)
        is_knn = (name == "KNeighborsClassifier")
        rec = fit_and_evaluate_classifier(
            name=name,
            model=model,
            train_df=train_df,
            val_df=val_df,
            features=features,
            label=label,
            folds=folds,
            is_knn=is_knn,
            max_knn_samples=max_knn_samples,
            random_state=random_state,
        )
        print(f"Done (PR-AUC: {rec['pr_auc']:.4f}, ROC-AUC: {rec['roc_auc']:.4f}, F1*: {rec['f1_optimal']:.4f})")
        records.append(rec)

    comparison_df = pd.DataFrame(records)
    save_comparison_results(comparison_df, output_dir=output_dir)
    return comparison_df


def main():
    """Main entry point for running the classifier comparison benchmark."""
    print("Loading fire panel dataset...")
    df, features = load_data()
    train, val, _ = time_split(df)

    res_df = compare_all_classifiers(
        train_df=train,
        val_df=val,
        features=features,
        label=LABEL,
        random_state=42,
        max_knn_samples=KNN_MAX_TRAIN_SAMPLES,
    )

    print("\n" + "=" * 95)
    print("CLASSIFIER BENCHMARK COMPARISON RESULTS (VALIDATION SET 2022)")
    print("=" * 95)
    cols_display = [
        "model", "pr_auc", "roc_auc", "brier_score",
        "optimal_threshold", "f1_optimal", "recall_optimal", "precision_optimal",
        "f1_default_05", "fit_time_sec", "eval_time_sec",
    ]
    print(res_df[cols_display].to_string(index=False))
    print("=" * 95)
    print(f"Results exported to {RESULTS_DIR / 'classifier_comparison.csv'}")


if __name__ == "__main__":
    main()
