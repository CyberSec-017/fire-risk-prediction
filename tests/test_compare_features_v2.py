"""Unit tests for src/compare_features_v2.py (Comparison Protocol of Feature Engineering V2).

Tests configuration integrity, exact feature counts, absence of protected columns,
exact model parameters, temporal fold generation, OOF threshold determination,
Test set isolation, identical train/val splits, and fast synthetic pipeline execution.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from compare_features_v2 import (
    BASELINE_FEATURES,
    BEST_HYPERPARAMS_PATH,
    EXPERIMENT_CONFIGS,
    compute_oof_threshold,
    get_feature_names_for_config,
    get_tuned_hgb_model,
    load_and_prepare_continuous_panel,
    run_all_feature_comparisons,
    run_single_experiment,
)
from features_v2 import build_features_v2
from train_baseline import EXCLUDE_COLS, LABEL


class TestExperimentConfigurations(unittest.TestCase):
    """Tests for the 6 experimental configurations."""

    def test_six_configurations_exist_with_exact_counts(self):
        """Verify the 6 configurations have exact feature counts (20, 22, 23, 24, 23, 32)."""
        expected_counts = {
            "A_baseline": 20,
            "B_vpd": 22,
            "C_anomalies": 23,
            "D_indices": 24,
            "E_neighbors": 23,
            "F_all": 32,
        }
        self.assertCountEqual(list(EXPERIMENT_CONFIGS.keys()), list(expected_counts.keys()))

        for config_key, expected_n in expected_counts.items():
            features = get_feature_names_for_config(config_key)
            self.assertEqual(len(features), expected_n, f"Failed for {config_key}")
            # Ensure features are unique
            self.assertEqual(len(features), len(set(features)), f"Duplicates in {config_key}")

    def test_no_protected_columns_in_feature_sets(self):
        """None of the feature sets may contain target, burned, n_detections, total_frp, or date."""
        protected = {"target", "burned", "n_detections", "total_frp", "date"}
        for config_key in EXPERIMENT_CONFIGS:
            features = set(get_feature_names_for_config(config_key))
            leakage_cols = features & protected
            self.assertEqual(len(leakage_cols), 0, f"Protected columns found in {config_key}: {leakage_cols}")

    def test_baseline_features_match_original_20(self):
        """Baseline features must match the 20 variables from production model card."""
        self.assertEqual(len(BASELINE_FEATURES), 20)
        self.assertIn("GWETROOT", BASELINE_FEATURES)
        self.assertIn("rain_30d", BASELINE_FEATURES)
        self.assertIn("doy_sin", BASELINE_FEATURES)


class TestTunedModelInstantiation(unittest.TestCase):
    """Tests for HistGradientBoostingClassifier instantiation with tuned hyperparams."""

    def test_tuned_hgb_model_parameters(self):
        """Model must load best_hyperparameters.json and preserve fixed defaults."""
        clf = get_tuned_hgb_model()
        params = clf.get_params(deep=True)

        # Check tuned hyperparameters from Commit 2
        self.assertEqual(params["l2_regularization"], 0.1)
        self.assertEqual(params["learning_rate"], 0.03)
        self.assertEqual(params["max_depth"], 8)
        self.assertEqual(params["min_samples_leaf"], 100)

        # Check fixed defaults preserved from baseline
        self.assertEqual(params["max_iter"], 150)
        self.assertEqual(params["random_state"], 42)


class TestTemporalProtocolAndThreshold(unittest.TestCase):
    """Tests for chronological folds, OOF threshold determination, and Test isolation."""

    def setUp(self):
        # Create a small synthetic continuous panel across 2 cells and 40 dates
        dates = pd.date_range("2018-03-01", periods=40, freq="D")
        rows = []
        for d in dates:
            for lat in (35.0, 35.5):
                for lon in (3.0, 3.625):
                    rows.append({
                        "lat_cell": lat,
                        "lon_cell": lon,
                        "date": d,
                        "T2M": 20.0 + np.sin(d.day),
                        "T2M_MAX": 25.0 + np.sin(d.day),
                        "RH2M": 50.0 + 10.0 * np.cos(d.day),
                        "WS2M": 3.0,
                        "PRECTOTCORR": 1.0 if d.day % 5 == 0 else 0.0,
                        "GWETROOT": 0.5,
                        "rain_7d": 1.0, "tmax_7d": 25.0, "rhmin_7d": 40.0,
                        "rain_14d": 2.0, "tmax_14d": 25.0, "rhmin_14d": 35.0,
                        "rain_30d": 3.0, "tmax_30d": 25.0, "rhmin_30d": 30.0,
                        "dry_days": 2, "doy_sin": 0.5, "doy_cos": 0.8,
                        "n_detections": 1 if d.day % 7 == 0 else 0,
                        "target": 1 if (lat == 35.5 and d.day % 4 == 0) else 0,
                    })
        self.synthetic_df = pd.DataFrame(rows)

    def test_oof_threshold_selection_on_train_only(self):
        """OOF threshold must be derived exclusively from Train without touching Validation."""
        enriched = build_features_v2(self.synthetic_df, lat_step=0.5, lon_step=0.625)
        # Train: first 30 days, Val: last 10 days
        train_df = enriched[enriched["date"] <= "2018-03-30"].copy()

        features = get_feature_names_for_config("A_baseline")
        X_train = train_df[features]
        y_train = train_df["target"].to_numpy()

        fast_clf = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)
        theta_star = compute_oof_threshold(
            fast_clf,
            X_train,
            y_train,
            train_df["date"],
            n_splits=2,
            purge_days=1,
            random_state=42,
        )
        self.assertIsInstance(theta_star, float)
        self.assertGreaterEqual(theta_star, 0.0)
        self.assertLessEqual(theta_star, 1.0)

    def test_no_test_set_access_in_continuous_panel(self):
        """load_and_prepare_continuous_panel must exclude Test years 2023-2024."""
        import tempfile
        # Create a small multi-year synthetic panel spanning 2021 to 2024
        dates = (
            pd.date_range("2021-06-01", periods=10).tolist()
            + pd.date_range("2021-12-25", periods=7).tolist()   # includes purge 2021-12-29..31
            + pd.date_range("2022-06-01", periods=10).tolist()
            + pd.date_range("2022-12-25", periods=7).tolist()   # includes purge 2022-12-29..31
            + pd.date_range("2023-06-01", periods=10).tolist()  # Test
            + pd.date_range("2024-06-01", periods=10).tolist()  # Test
        )
        rows = []
        for d in dates:
            for lat, lon in [(35.0, 3.0), (35.5, 3.625)]:
                rows.append({
                    "lat_cell": lat, "lon_cell": lon, "date": d,
                    "T2M": 22.0, "T2M_MAX": 27.0, "RH2M": 50.0, "WS2M": 3.0,
                    "PRECTOTCORR": 0.0, "GWETROOT": 0.5,
                    "rain_7d": 0.0, "tmax_7d": 27.0, "rhmin_7d": 40.0,
                    "rain_14d": 0.0, "tmax_14d": 27.0, "rhmin_14d": 35.0,
                    "rain_30d": 0.0, "tmax_30d": 27.0, "rhmin_30d": 30.0,
                    "dry_days": 10, "doy_sin": 0.5, "doy_cos": 0.8,
                    "n_detections": 0, "target": 0,
                })
        dummy_df = pd.DataFrame(rows)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            dummy_df.to_parquet(tmp_path, index=False)
            train_df, val_df = load_and_prepare_continuous_panel(panel_path=tmp_path, val_year=2022)

            # Train must end strictly before 2021-12-29 (H=3 days purge)
            self.assertLess(train_df["date"].max(), pd.Timestamp("2021-12-29"))

            # Val must start >= 2022-01-01 and end < 2022-12-29 (H=3 days purge)
            self.assertGreaterEqual(val_df["date"].min(), pd.Timestamp("2022-01-01"))
            self.assertLess(val_df["date"].max(), pd.Timestamp("2022-12-29"))

            # Test set (2023, 2024) must be strictly absent
            self.assertFalse((train_df["date"] >= "2023-01-01").any())
            self.assertFalse((val_df["date"] >= "2023-01-01").any())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestSyntheticComparisonPipeline(unittest.TestCase):
    """Fast integration test of the 6 configurations on synthetic data."""

    def setUp(self):
        dates = pd.date_range("2018-03-01", periods=30, freq="D")
        rows = []
        for d in dates:
            for lat in (35.0, 35.5):
                for lon in (3.0, 3.625):
                    rows.append({
                        "lat_cell": lat,
                        "lon_cell": lon,
                        "date": d,
                        "T2M": 22.0 + np.sin(d.day),
                        "T2M_MAX": 27.0 + np.sin(d.day),
                        "RH2M": 45.0 + 10.0 * np.cos(d.day),
                        "WS2M": 3.5,
                        "PRECTOTCORR": 2.0 if d.day % 4 == 0 else 0.0,
                        "GWETROOT": 0.4,
                        "rain_7d": 1.0, "tmax_7d": 27.0, "rhmin_7d": 35.0,
                        "rain_14d": 2.0, "tmax_14d": 27.0, "rhmin_14d": 30.0,
                        "rain_30d": 3.0, "tmax_30d": 27.0, "rhmin_30d": 25.0,
                        "dry_days": 1, "doy_sin": 0.5, "doy_cos": 0.8,
                        "n_detections": 1 if d.day % 6 == 0 else 0,
                        "target": 1 if (lat == 35.5 and d.day % 3 == 0) else 0,
                    })
        self.df = pd.DataFrame(rows)
        self.enriched = build_features_v2(self.df, lat_step=0.5, lon_step=0.625)

        # Split: Train first 20 days, Val last 10 days
        self.train_df = self.enriched[self.enriched["date"] < "2018-03-21"].copy()
        self.val_df = self.enriched[self.enriched["date"] >= "2018-03-21"].copy()

    def test_identical_train_val_rows_across_configs(self):
        """Train and Validation rows must be strictly identical across all 6 configs."""
        for config_key in EXPERIMENT_CONFIGS:
            features = get_feature_names_for_config(config_key)
            X_tr = self.train_df[features]
            X_va = self.val_df[features]
            self.assertEqual(len(X_tr), len(self.train_df))
            self.assertEqual(len(X_va), len(self.val_df))
            self.assertListEqual(list(X_tr.index), list(self.train_df.index))
            self.assertListEqual(list(X_va.index), list(self.val_df.index))

    def test_fast_synthetic_comparison_run(self):
        """Verify run_all_feature_comparisons produces a complete, valid results table."""
        fast_clf = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_csv = Path(tmp_dir) / "test_comp.csv"
            tmp_json = Path(tmp_dir) / "test_meta.json"
            results = run_all_feature_comparisons(
                train_df=self.train_df,
                val_df=self.val_df,
                model=fast_clf,
                output_path=tmp_csv,
                metadata_path=tmp_json,
                n_splits=2,
                purge_days=1,
                random_state=42,
            )

            self.assertTrue(tmp_csv.exists())
            self.assertTrue(tmp_json.exists())

        self.assertEqual(len(results), 6)
        expected_cols = [
            "config_key", "name", "n_features", "pr_auc", "roc_auc",
            "brier_score", "threshold_oof", "f1_score", "precision",
            "recall", "delta_pr_auc_vs_baseline", "fit_time_seconds"
        ]
        for col in expected_cols:
            self.assertIn(col, results.columns)

        # Delta vs baseline for A_baseline must be exactly 0.0
        baseline_delta = results.loc[results["config_key"] == "A_baseline", "delta_pr_auc_vs_baseline"].iloc[0]
        self.assertAlmostEqual(baseline_delta, 0.0)

        # All PR-AUCs must be in [0, 1]
        self.assertTrue(((results["pr_auc"] >= 0.0) & (results["pr_auc"] <= 1.0)).all())

    def test_reproducibility(self):
        """Running the same config with the same random_state must yield identical metrics."""
        fast_clf1 = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)
        fast_clf2 = HistGradientBoostingClassifier(max_iter=5, max_depth=3, random_state=42)

        res1 = run_single_experiment(
            "A_baseline", self.train_df, self.val_df,
            model=fast_clf1, n_splits=2, purge_days=1, random_state=42
        )
        res2 = run_single_experiment(
            "A_baseline", self.train_df, self.val_df,
            model=fast_clf2, n_splits=2, purge_days=1, random_state=42
        )

        self.assertEqual(res1["pr_auc"], res2["pr_auc"])
        self.assertEqual(res1["threshold_oof"], res2["threshold_oof"])
        self.assertEqual(res1["f1_score"], res2["f1_score"])


if __name__ == "__main__":
    unittest.main()
