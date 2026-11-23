"""Experimental comparison of Feature Engineering V2 configurations.

Compares 6 feature configurations using HistGradientBoostingClassifier with optimal
hyperparameters from Commit 2 (results/best_hyperparameters.json):
  A. Baseline (20 features): Production baseline feature set.
  B. Baseline + VPD (22 features): Adds 'vpd', 'vpd_tmax_proxy'.
  C. Baseline + Anomalies (23 features): Adds 'tmax_anomaly_30d', 't2m_anomaly_hist', 't2m_zscore_hist'.
  D. Baseline + Indices (24 features): Adds 'angstrom_proxy', 'cbi', 'hdw_proxy', 'nesterov_proxy'.
  E. Baseline + Neighbors (23 features): Adds 'neighbor_fires_7d', 'neighbor_fires_14d', 'neighbor_fires_30d'.
  F. All Families Combined (32 features): Baseline + all 12 new features.

Methodological guarantees:
  - Dataset partitions: Continuous pre-test panel (2018-03-02 to 2022-12-28) used to build features,
    preserving historical continuity across the 3-day purge gap (Dec 29-31, 2021).
  - Train (2018-03-02 to 2021-12-28, 292,182 rows) and Validation (2022-01-01 to 2022-12-28, 75,658 rows)
    remain 100% identical across all 6 configurations.
  - Test set (2023-2024) is strictly isolated and never accessed.
  - Decision threshold theta* is determined strictly from Train Out-of-Fold (OOF) predictions
    using chronological expanding folds with a strict 3-day purge gap. Never optimized on Validation.
  - Evaluates PR-AUC (primary metric), ROC-AUC, Brier score, Precision, Recall, and F1 on Validation.
  - Results saved to results/feature_engineering_v2_comparison.csv via atomic file writing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from compare_classifiers import (
    PURGE_DAYS,
    RESULTS_DIR,
    find_optimal_threshold_oof,
)
from experimental_pipeline import build_experimental_pipeline
from features_v2 import ALL_NEW_FEATURES, FEATURE_FAMILIES, build_features_v2
from train_baseline import EXCLUDE_COLS, LABEL, VAL_YEAR
from tune_classifiers import build_tuning_date_folds, get_tuning_classifiers

BEST_HYPERPARAMS_PATH = RESULTS_DIR / "best_hyperparameters.json"
COMPARISON_RESULTS_PATH = RESULTS_DIR / "feature_engineering_v2_comparison.csv"

# The 20 baseline features defined in model_card.json and train_baseline.py
BASELINE_FEATURES: list[str] = [
    "lat_cell",
    "lon_cell",
    "T2M",
    "T2M_MAX",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "GWETROOT",
    "rain_7d",
    "tmax_7d",
    "rhmin_7d",
    "rain_14d",
    "tmax_14d",
    "rhmin_14d",
    "rain_30d",
    "tmax_30d",
    "rhmin_30d",
    "dry_days",
    "doy_sin",
    "doy_cos",
]

EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "A_baseline": {
        "slug": "baseline",
        "name": "A. Baseline (20 features)",
        "families": [],
        "n_features": 20,
    },
    "B_vpd": {
        "slug": "baseline_plus_vpd",
        "name": "B. Baseline + VPD (22 features)",
        "families": ["vpd"],
        "n_features": 22,
    },
    "C_anomalies": {
        "slug": "baseline_plus_anomalies",
        "name": "C. Baseline + Anomalies (23 features)",
        "families": ["anomalies"],
        "n_features": 23,
    },
    "D_indices": {
        "slug": "baseline_plus_indices",
        "name": "D. Baseline + Indices (24 features)",
        "families": ["indices"],
        "n_features": 24,
    },
    "E_neighbors": {
        "slug": "baseline_plus_neighbors",
        "name": "E. Baseline + Voisins (23 features)",
        "families": ["neighbors"],
        "n_features": 23,
    },
    "F_all": {
        "slug": "baseline_plus_all",
        "name": "F. Baseline + Toutes familles (32 features)",
        "families": ["vpd", "anomalies", "indices", "neighbors"],
        "n_features": 32,
    },
}


def get_feature_names_for_config(config_key: str) -> list[str]:
    """Retrieve the exact list of feature names for a given experiment configuration."""
    if config_key not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown config_key '{config_key}'. Allowed: {list(EXPERIMENT_CONFIGS.keys())}")

    cfg = EXPERIMENT_CONFIGS[config_key]
    features = list(BASELINE_FEATURES)
    for fam in cfg["families"]:
        features.extend(FEATURE_FAMILIES[fam])

    if len(features) != cfg["n_features"]:
        raise ValueError(
            f"Configuration {config_key} expected {cfg['n_features']} features, but got {len(features)}"
        )
    return features


def get_tuned_hgb_model(
    best_params_path: Path | str | None = None,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """Instantiate HistGradientBoostingClassifier with optimal tuned hyperparameters.

    Reuses the base estimator from get_tuning_classifiers() to preserve fixed baseline
    defaults (max_iter=150, random_state=42) and applies the optimal hyperparameters
    loaded from results/best_hyperparameters.json.
    Verifies parameter application via get_params(deep=True).
    """
    path = Path(best_params_path) if best_params_path is not None else BEST_HYPERPARAMS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Best hyperparameters file not found at {path}. Run tuning first."
        )

    with open(path, "r", encoding="utf-8") as f:
        all_best = json.load(f)

    if "HistGradientBoostingClassifier" not in all_best:
        raise KeyError(f"HistGradientBoostingClassifier not found in {path}")

    best_params = all_best["HistGradientBoostingClassifier"]

    # Re-use base estimator from get_tuning_classifiers
    base_models = get_tuning_classifiers(random_state=random_state)
    clf = base_models["HistGradientBoostingClassifier"]
    clf.set_params(**best_params)

    # Introspect to confirm exact application
    actual_params = clf.get_params(deep=True)
    for param_name, expected_val in best_params.items():
        if actual_params.get(param_name) != expected_val:
            raise ValueError(
                f"Parameter mismatch for {param_name}: expected {expected_val}, got {actual_params.get(param_name)}"
            )

    return clf


def load_and_prepare_continuous_panel(
    panel_path: Path | str | None = None,
    val_year: int = VAL_YEAR,
    h_days: int = config.H,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load continuous pre-test panel, enrich features, and slice Train and Validation.

    Guarantees:
      - Test set (years > val_year) is never loaded or accessed.
      - Continuous pre-test panel preserves rolling continuity across the 3-day purge gap.
      - Train (2018-03-02 to 2021-12-28, 292,182 rows) and Validation (2022-01-01 to 2022-12-28, 75,658 rows)
        have identical row counts and indices across all experiments.
      - No rows are dropped due to NaNs (leveraging HistGradientBoosting native NaN handling).
    """
    path = Path(panel_path) if panel_path is not None else config.FIRE_PANEL
    df = pd.read_parquet(path)

    val_start = pd.Timestamp(f"{val_year}-01-01")
    test_start = pd.Timestamp(f"{val_year + 1}-01-01")

    # Filter pre-test panel strictly: dates strictly before the test set purge horizon
    # Pre-test data ends at 2022-12-28 (since 2022-12-29 to 2022-12-31 is Purge 2)
    pre_test_mask = df["date"] < test_start - pd.Timedelta(days=h_days)
    pre_test_panel = df[pre_test_mask].copy()

    # Load historical FIRMS detections prior to panel start date if available
    history_df = None
    if config.FIRES_DAILY.exists():
        fires_daily = pd.read_parquet(config.FIRES_DAILY)
        min_date = pre_test_panel["date"].min()
        hist_mask = (fires_daily["date"] >= "2018-01-01") & (fires_daily["date"] < min_date)
        history_df = fires_daily[hist_mask].copy()

    # Enrich with all 12 new features continuously over the pre-test timeline
    enriched_panel = build_features_v2(pre_test_panel, history_df=history_df)

    # Slice Train and Validation using exact chronological conditions
    train_mask = enriched_panel["date"] < val_start - pd.Timedelta(days=h_days)
    val_mask = (enriched_panel["date"] >= val_start) & (enriched_panel["date"] < test_start - pd.Timedelta(days=h_days))

    train_df = enriched_panel[train_mask].copy()
    val_df = enriched_panel[val_mask].copy()

    return train_df, val_df


def compute_oof_threshold(
    model: HistGradientBoostingClassifier,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    train_dates: pd.Series | np.ndarray,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    random_state: int = 42,
) -> float:
    """Compute optimal decision threshold theta* strictly from Train Out-of-Fold predictions.

    Uses chronological expanding-window folds with a strict 3-day purge gap.
    Validation data is never accessed during threshold determination.
    """
    temp_df = pd.DataFrame({"date": pd.to_datetime(train_dates)})
    y_arr = np.asarray(y_train)
    folds = build_tuning_date_folds(temp_df, y_arr, n_splits=n_splits, purge_days=purge_days)

    oof_probs = np.full(len(y_arr), np.nan, dtype=np.float64)

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        fold_clf = clone(model)
        X_tr = X_train.iloc[tr_idx].copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train[tr_idx])
        y_tr = y_arr[tr_idx]
        X_va = X_train.iloc[val_idx].copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train[val_idx])

        # Safeguard: scikit-learn's HistGradientBoosting binning fails if a feature is 100% NaN
        # in early expanding folds (e.g. 2018 where historical anomalies have no prior reference).
        # We replace any 100% NaN column with 0.0 for that specific fold without altering real data.
        all_nan_cols = [c for c in X_tr.columns if X_tr[c].isna().all()]
        for c in all_nan_cols:
            X_tr[c] = 0.0
            X_va[c] = 0.0

        fold_clf.fit(X_tr, y_tr)
        probs = fold_clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = probs

    theta_star = find_optimal_threshold_oof(y_arr, oof_probs)
    return float(theta_star)


def run_single_experiment(
    config_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model: HistGradientBoostingClassifier | None = None,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    random_state: int = 42,
    use_pipeline: bool = False,
) -> dict[str, Any]:
    """Run an isolated experiment for one feature configuration.

    Protocol:
      1. Extract X_train, y_train, X_val, y_val according to config feature whitelist.
      2. Verify absence of protected columns in X.
      3. Compute OOF decision threshold theta* on Train.
      4. Freeze theta*.
      5. Fit model on full Train.
      6. Predict probabilities on Validation.
      7. Evaluate PR-AUC, ROC-AUC, Brier score, Precision, Recall, F1 on Validation.
    ``use_pipeline=True`` wraps the estimator in the experimental
    ColumnTransformer/passthrough pipeline; the default keeps the historical raw
    DataFrame behaviour and never touches the deployed model.
    """
    if config_key not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown config_key '{config_key}'")

    cfg = EXPERIMENT_CONFIGS[config_key]
    features = get_feature_names_for_config(config_key)

    # Integrity assertion: ensure no target/leak columns
    for col in (LABEL, "burned", "n_detections", "total_frp", "date"):
        if col in features:
            raise ValueError(f"CRITICAL: Protected column '{col}' detected in feature set!")

    X_train = train_df[features]
    y_train = train_df[LABEL].to_numpy(dtype=int)
    X_val = val_df[features]
    y_val = val_df[LABEL].to_numpy(dtype=int)

    clf = model if model is not None else get_tuned_hgb_model(random_state=random_state)
    if use_pipeline and not isinstance(clf, Pipeline):
        # Opt-in only: explicit numeric schema, passthrough preprocessing for
        # HistGradientBoosting. Production inference never uses this wrapper.
        clf = build_experimental_pipeline(clf, features=features)

    t0 = time.time()
    # Step 1: Compute OOF threshold on Train
    theta_star = compute_oof_threshold(
        clf,
        X_train,
        y_train,
        train_df["date"],
        n_splits=n_splits,
        purge_days=purge_days,
        random_state=random_state,
    )

    # Step 2: Refit on full Train
    full_clf = clone(clf)
    all_nan_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_nan_cols:
        X_train = X_train.copy()
        X_val = X_val.copy()
        for c in all_nan_cols:
            X_train[c] = 0.0
            X_val[c] = 0.0

    full_clf.fit(X_train, y_train)

    # Step 3: Evaluate on Validation
    val_probs = full_clf.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0

    pr_auc = float(average_precision_score(y_val, val_probs))
    roc_auc = float(roc_auc_score(y_val, val_probs))
    brier = float(brier_score_loss(y_val, val_probs))

    val_preds = (val_probs >= theta_star).astype(int)
    prec = float(precision_score(y_val, val_preds, zero_division=0))
    rec = float(recall_score(y_val, val_preds, zero_division=0))
    f1 = float(f1_score(y_val, val_preds, zero_division=0))

    return {
        "config_key": config_key,
        "name": cfg["name"],
        "n_features": len(features),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "threshold_oof": theta_star,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "fit_time_seconds": round(elapsed, 2),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }


def save_metadata(
    configs: dict[str, dict[str, Any]],
    model: HistGradientBoostingClassifier,
    results_records: list[dict[str, Any]],
    output_path: Path | str | None = None,
) -> None:
    """Atomically save experimental metadata and feature specifications for reproducibility."""
    path = Path(output_path) if output_path is not None else RESULTS_DIR / "feature_engineering_v2_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": "HistGradientBoostingClassifier",
        "hyperparameters": model.get_params(),
        "cv_protocol": {
            "n_splits": 5,
            "purge_days": 3,
            "horizon_h": 3,
            "threshold_selection": "Train OOF F1 maximization",
        },
        "configurations": {
            k: {
                "name": v["name"],
                "n_features": v["n_features"],
                "features": get_feature_names_for_config(k),
            }
            for k, v in configs.items()
        },
        "completed_experiments": [r["config_key"] for r in results_records],
    }

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
        encoding="utf-8",
    )
    temp_path = Path(temp_file.name)
    try:
        json.dump(meta, temp_file, indent=2)
        temp_file.close()
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def run_all_feature_comparisons(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model: HistGradientBoostingClassifier | None = None,
    output_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    n_splits: int = 5,
    purge_days: int = PURGE_DAYS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute all 6 feature configurations and compile comparative metrics table.

    Progressively and atomically saves results after each individual experiment to ensure
    fault tolerance and full reproducibility.
    """
    out_csv = Path(output_path) if output_path is not None else COMPARISON_RESULTS_PATH
    meta_json = Path(metadata_path) if metadata_path is not None else RESULTS_DIR / "feature_engineering_v2_metadata.json"

    records: list[dict[str, Any]] = []
    base_clf = model if model is not None else get_tuned_hgb_model(random_state=random_state)

    for idx, (config_key, cfg) in enumerate(EXPERIMENT_CONFIGS.items(), start=1):
        print(f"\n[{idx}/6] Evaluating {cfg['name']} ({cfg['n_features']} features) ...")
        res = run_single_experiment(
            config_key=config_key,
            train_df=train_df,
            val_df=val_df,
            model=base_clf,
            n_splits=n_splits,
            purge_days=purge_days,
            random_state=random_state,
        )
        records.append(res)
        results_df = pd.DataFrame(records)

        # Compute delta PR-AUC relative to baseline if baseline has been computed
        baseline_rows = results_df.loc[results_df["config_key"] == "A_baseline", "pr_auc"]
        if not baseline_rows.empty:
            baseline_pr_auc = baseline_rows.values[0]
            results_df["delta_pr_auc_vs_baseline"] = results_df["pr_auc"] - baseline_pr_auc
        else:
            results_df["delta_pr_auc_vs_baseline"] = 0.0

        print(
            f"  PR-AUC: {res['pr_auc']:.4f} | ROC-AUC: {res['roc_auc']:.4f} | "
            f"F1: {res['f1_score']:.4f} (theta*={res['threshold_oof']:.4f}) | "
            f"Time: {res['fit_time_seconds']}s"
        )

        # Progressive atomic save after every experiment
        save_comparison_results(results_df, output_path=out_csv)
        save_metadata(
            configs=EXPERIMENT_CONFIGS,
            model=base_clf,
            results_records=records,
            output_path=meta_json,
        )

    return results_df


def save_comparison_results(
    results_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> None:
    """Atomically save the comparison table to CSV."""
    path = Path(output_path) if output_path is not None else COMPARISON_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
        encoding="utf-8",
        newline="",
    )
    temp_path = Path(temp_file.name)
    try:
        results_df.to_csv(temp_file, index=False, lineterminator="\n")
        temp_file.close()
        os.replace(temp_path, path)
        print(f"Saved feature comparison table to {path}")
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def main():
    print("=" * 80)
    print("FEATURE ENGINEERING V2 — EXPERIMENTAL COMPARISON")
    print("=" * 80)

    print("1. Loading continuous pre-test panel and preparing features ...")
    train_df, val_df = load_and_prepare_continuous_panel()
    print(f"   Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    print("2. Instantiating tuned HistGradientBoostingClassifier ...")
    clf = get_tuned_hgb_model()
    print("   Parameters:", clf.get_params())

    print("3. Executing comparison of 6 configurations ...")
    results = run_all_feature_comparisons(train_df, val_df, model=clf)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS SUMMARY")
    print("=" * 80)
    print(results[["name", "n_features", "pr_auc", "delta_pr_auc_vs_baseline", "roc_auc", "f1_score", "threshold_oof"]].to_string(index=False))

    save_comparison_results(results)


if __name__ == "__main__":
    main()
