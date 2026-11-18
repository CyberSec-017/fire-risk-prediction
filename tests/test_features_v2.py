"""Unit tests for src/features_v2.py (Advanced Feature Engineering V2).

Tests mathematical validity, physical boundaries, causal isolation,
zero future leakage, zero target dependence, index preservation,
and correct feature counts across all 4 feature families.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features_v2 import (
    ALL_NEW_FEATURES,
    FEATURE_FAMILIES,
    build_features_v2,
    compute_fire_weather_indices,
    compute_neighbor_fire_history,
    compute_temperature_anomalies,
    compute_vpd,
)


class TestVPD(unittest.TestCase):
    """Tests for Vapor Pressure Deficit (VPD) features."""

    def setUp(self):
        self.df = pd.DataFrame({
            "lat_cell": [36.0, 36.0],
            "lon_cell": [3.0, 3.0],
            "date": pd.to_datetime(["2020-07-01", "2020-07-02"]),
            "T2M": [20.0, 30.0],
            "T2M_MAX": [25.0, 38.0],
            "RH2M": [50.0, 20.0],
        })

    def test_vpd_known_values(self):
        """Verify VPD matches Magnus-Tetens exact analytical values at 20°C and 50% RH."""
        res = compute_vpd(self.df)
        # es(20) = 0.61078 * exp(17.27 * 20 / 257.3) = 0.61078 * exp(1.3424) ~ 2.3377 kPa
        # at 50% RH: vpd = 2.3377 * (1 - 0.50) ~ 1.1688 kPa
        val = res.loc[0, "vpd"]
        self.assertAlmostEqual(val, 1.1688, places=3)
        # vpd_tmax_proxy uses T2M_MAX (25°C) and RH2M (50%)
        # es(25) = 0.61078 * exp(17.27 * 25 / 262.3) ~ 3.167 kPa -> vpd ~ 1.583 kPa
        val_max = res.loc[0, "vpd_tmax_proxy"]
        self.assertAlmostEqual(val_max, 1.583, places=2)
        self.assertGreater(val_max, val)

    def test_vpd_positivity_and_clipping(self):
        """VPD must be non-negative even with out-of-bound or saturated RH."""
        test_df = pd.DataFrame({
            "T2M": [15.0, 25.0, -5.0],
            "T2M_MAX": [20.0, 30.0, 0.0],
            "RH2M": [100.0, 110.0, -10.0],
        })
        res = compute_vpd(test_df)
        # At 100% and 110% RH, VPD should be exactly 0.0
        self.assertAlmostEqual(res.loc[0, "vpd"], 0.0, places=5)
        self.assertAlmostEqual(res.loc[1, "vpd"], 0.0, places=5)
        # At -10% RH, clipped to 0%, VPD is positive and equals es
        self.assertGreater(res.loc[2, "vpd"], 0.0)
        self.assertTrue((res["vpd"] >= 0.0).all())

    def test_vpd_nan_handling(self):
        """Missing temperature or humidity must cleanly evaluate to NaN."""
        test_df = pd.DataFrame({
            "T2M": [np.nan, 25.0],
            "T2M_MAX": [30.0, np.nan],
            "RH2M": [50.0, np.nan],
        })
        res = compute_vpd(test_df)
        self.assertTrue(np.isnan(res.loc[0, "vpd"]))
        self.assertTrue(np.isnan(res.loc[1, "vpd"]))
        self.assertTrue(np.isnan(res.loc[1, "vpd_tmax_proxy"]))

    def test_vpd_preserves_immutability_and_index(self):
        """Original DataFrame must remain unmodified and custom index preserved."""
        df_custom = self.df.copy()
        df_custom.index = [105, 204]
        res = compute_vpd(df_custom)
        self.assertListEqual(list(res.index), [105, 204])
        self.assertNotIn("vpd", self.df.columns)


class TestTemperatureAnomalies(unittest.TestCase):
    """Tests for causal temperature anomalies."""

    def setUp(self):
        # Construct multi-year synthetic dataset for a single cell across July
        dates = pd.date_range("2018-07-01", "2018-07-10").tolist() + \
                pd.date_range("2019-07-01", "2019-07-10").tolist() + \
                pd.date_range("2020-07-01", "2020-07-10").tolist() + \
                pd.date_range("2021-07-01", "2021-07-10").tolist()
        n = len(dates)
        # 2018: base 25°C, 2019: base 27°C, 2020: base 29°C, 2021: base 31°C
        temps = []
        for d in dates:
            if d.year == 2018:
                temps.append(25.0)
            elif d.year == 2019:
                temps.append(27.0)
            elif d.year == 2020:
                temps.append(29.0)
            else:
                temps.append(31.0)

        self.df = pd.DataFrame({
            "lat_cell": [36.0] * n,
            "lon_cell": [3.0] * n,
            "date": pd.to_datetime(dates),
            "T2M": temps,
            "T2M_MAX": [t + 5.0 for t in temps],
        })

    def test_tmax_anomaly_30d_uses_shift1(self):
        """tmax_anomaly_30d must strictly exclude current day t."""
        # On day 0, no prior history exists -> shift(1) is NaN -> anomaly is NaN
        res = compute_temperature_anomalies(self.df)
        self.assertTrue(np.isnan(res.loc[0, "tmax_anomaly_30d"]))

        # Check day 1: prior day is day 0 with T2M_MAX = 30.0. Current T2M_MAX = 30.0 -> anomaly = 0.0
        self.assertAlmostEqual(res.loc[1, "tmax_anomaly_30d"], 0.0)

        # Introduce a sudden spike on day 5 and verify day 5 does not leak into day 4
        df_spike = self.df.copy()
        df_spike.loc[5, "T2M_MAX"] = 50.0  # Spike of +20°C
        res_spike = compute_temperature_anomalies(df_spike)
        # Day 4 must NOT see the spike of day 5
        self.assertAlmostEqual(res_spike.loc[4, "tmax_anomaly_30d"], 0.0)
        # Day 5 sees the spike relative to past days (which averaged 30.0) -> ~20.0
        self.assertAlmostEqual(res_spike.loc[5, "tmax_anomaly_30d"], 20.0)

    def test_historical_anomaly_prior_years_only(self):
        """Anomalies must use strictly prior years (Y' < Y)."""
        res = compute_temperature_anomalies(self.df)

        # Year 2018 has 0 prior years -> t2m_anomaly_hist must be NaN
        sub_2018 = res[res["date"].dt.year == 2018]
        self.assertTrue(sub_2018["t2m_anomaly_hist"].isna().all())
        self.assertTrue(sub_2018["t2m_zscore_hist"].isna().all())

        # Year 2019 has 1 prior year (2018: mean 25.0). Current T2M = 27.0 -> anomaly = +2.0
        sub_2019 = res[res["date"].dt.year == 2019]
        self.assertTrue(np.allclose(sub_2019["t2m_anomaly_hist"], 2.0))
        # Z-score requires >= 2 prior years -> must be NaN in 2019
        self.assertTrue(sub_2019["t2m_zscore_hist"].isna().all())

        # Year 2020 has 2 prior years (2018: 25.0, 2019: 27.0). Prior mean = 26.0.
        # Current T2M = 29.0 -> anomaly = 29 - 26 = +3.0
        sub_2020 = res[res["date"].dt.year == 2020]
        self.assertTrue(np.allclose(sub_2020["t2m_anomaly_hist"], 3.0))
        # Prior std for [25, 27] is sqrt(2) ~ 1.4142. Z-score = 3.0 / 1.4142 ~ 2.1213
        self.assertTrue(sub_2020["t2m_zscore_hist"].notna().all())
        self.assertAlmostEqual(sub_2020["t2m_zscore_hist"].iloc[0], 3.0 / np.std([25.0, 27.0], ddof=1), places=4)

    def test_zero_future_leakage(self):
        """Altering future observations in 2021 must NOT alter 2018, 2019, or 2020 anomalies."""
        res_baseline = compute_temperature_anomalies(self.df)

        df_future_corrupted = self.df.copy()
        # Set extreme temperatures in 2021
        df_future_corrupted.loc[df_future_corrupted["date"].dt.year == 2021, "T2M"] = 100.0

        res_corrupted = compute_temperature_anomalies(df_future_corrupted)

        # Prior years (2018, 2019, 2020) must be 100% bitwise identical
        sub_base = res_baseline[res_baseline["date"].dt.year <= 2020]
        sub_corr = res_corrupted[res_corrupted["date"].dt.year <= 2020]

        pd.testing.assert_series_equal(sub_base["t2m_anomaly_hist"], sub_corr["t2m_anomaly_hist"])
        pd.testing.assert_series_equal(sub_base["t2m_zscore_hist"], sub_corr["t2m_zscore_hist"])

    def test_zero_variance_std_yields_nan_zscore(self):
        """If prior years have identical temperature (std = 0), Z-score must be NaN."""
        df_const = self.df.copy()
        # Make 2018 and 2019 exactly equal
        df_const.loc[df_const["date"].dt.year == 2019, "T2M"] = 25.0
        res = compute_temperature_anomalies(df_const)
        sub_2020 = res[res["date"].dt.year == 2020]
        # In 2020, prior values are [25.0, 25.0] -> std is 0.0 -> Z-score must be NaN
        self.assertTrue(sub_2020["t2m_zscore_hist"].isna().all())


class TestMeteorologicalIndices(unittest.TestCase):
    """Tests for simplified empirical fire weather indices."""

    def setUp(self):
        self.df = pd.DataFrame({
            "lat_cell": [36.0] * 5,
            "lon_cell": [3.0] * 5,
            "date": pd.date_range("2020-07-01", periods=5),
            "T2M": [25.0, 30.0, 35.0, 20.0, 15.0],
            "T2M_MAX": [30.0, 35.0, 40.0, 25.0, 20.0],
            "RH2M": [40.0, 30.0, 20.0, 60.0, 80.0],
            "WS2M": [3.0, 5.0, 8.0, 2.0, 1.0],
            "PRECTOTCORR": [5.0, 0.0, 0.0, 5.0, 0.0],  # Day 0 and Day 3 have 5mm rain resets
        })

    def test_angstrom_proxy(self):
        """Check Ångström formula: RH/20 + (27-T)/10."""
        res = compute_fire_weather_indices(self.df)
        # Row 0: T=25, RH=40 -> 40/20 + (27-25)/10 = 2.0 + 0.2 = 2.2
        self.assertAlmostEqual(res.loc[0, "angstrom_proxy"], 2.2)

    def test_cbi_metric_formula(self):
        """Check Chandler Burning Index metric formula and non-negativity."""
        res = compute_fire_weather_indices(self.df)
        # Row 0: T=25, RH=40
        # term = (104.5 - 1.373*40 + 0.54*25) = (104.5 - 54.92 + 13.5) = 63.08
        # factor = 124 * 10^(-0.0142 * 40) = 124 * 10^(-0.568) = 124 * 0.270396 = 33.529
        # CBI = (63.08 * 33.529) / 60 ~ 35.25
        self.assertAlmostEqual(res.loc[0, "cbi"], 35.25, places=1)
        self.assertTrue((res["cbi"] >= 0.0).all())

    def test_nesterov_proxy_initial_history_nan_until_first_rain(self):
        """Nesterov proxy must be NaN until the first valid rain reset >= 3.0 mm."""
        df_unobserved = pd.DataFrame({
            "lat_cell": [36.0] * 4,
            "lon_cell": [3.0] * 4,
            "date": pd.date_range("2020-07-01", periods=4),
            "T2M": [25.0, 26.0, 27.0, 28.0],
            "RH2M": [40.0, 40.0, 40.0, 40.0],
            "WS2M": [3.0, 3.0, 3.0, 3.0],
            "PRECTOTCORR": [0.0, 0.0, 4.0, 0.0],  # Rain >= 3mm on Day 2
        })
        res = compute_fire_weather_indices(df_unobserved)
        # Days 0 and 1 have not experienced any rain reset -> must be NaN
        self.assertTrue(np.isnan(res.loc[0, "nesterov_proxy"]))
        self.assertTrue(np.isnan(res.loc[1, "nesterov_proxy"]))
        # Day 2 experienced rain >= 3mm -> must reset to 0.0
        self.assertAlmostEqual(res.loc[2, "nesterov_proxy"], 0.0)
        # Day 3 accumulates from 0.0 -> valid positive value
        self.assertGreater(res.loc[3, "nesterov_proxy"], 0.0)

    def test_nesterov_proxy_cell_isolation(self):
        """Nesterov proxy state must be strictly isolated per geographic cell."""
        dates = pd.date_range("2020-07-01", periods=5)
        rows = []
        for d in dates:
            # Both cells have rain on Day 0 to initialize valid tracking from known state
            rain_a = 10.0 if (d == dates[0] or d == dates[2]) else 0.0
            rain_b = 10.0 if d == dates[0] else 0.0
            # Cell A: gets 10mm rain on day 0 and day 2
            rows.append({
                "lat_cell": 35.0, "lon_cell": 3.0, "date": d,
                "T2M": 28.0, "RH2M": 30.0, "WS2M": 3.0,
                "PRECTOTCORR": rain_a,
            })
            # Cell B: gets rain on day 0, then stays completely dry for days 1-4
            rows.append({
                "lat_cell": 36.0, "lon_cell": 4.0, "date": d,
                "T2M": 28.0, "RH2M": 30.0, "WS2M": 3.0,
                "PRECTOTCORR": rain_b,
            })
        df_two_cells = pd.DataFrame(rows)
        res = compute_fire_weather_indices(df_two_cells)

        res_a = res[(res["lat_cell"] == 35.0) & (res["lon_cell"] == 3.0)].sort_values("date")
        res_b = res[(res["lat_cell"] == 36.0) & (res["lon_cell"] == 4.0)].sort_values("date")

        # Cell A resets to 0.0 on day 2
        self.assertAlmostEqual(res_a.iloc[2]["nesterov_proxy"], 0.0)
        # Cell B is never reset on day 2: day 2 is strictly greater than day 1
        self.assertGreater(res_b.iloc[2]["nesterov_proxy"], res_b.iloc[1]["nesterov_proxy"])
        self.assertGreater(res_b.iloc[4]["nesterov_proxy"], res_a.iloc[4]["nesterov_proxy"])

    def test_hdw_proxy(self):
        """HDW proxy must equal vpd * WS2M."""
        res = compute_fire_weather_indices(self.df)
        # Compute expected VPD for row 0
        df_vpd = compute_vpd(self.df)
        expected_hdw = df_vpd.loc[0, "vpd"] * self.df.loc[0, "WS2M"]
        self.assertAlmostEqual(res.loc[0, "hdw_proxy"], expected_hdw)

    def test_nesterov_proxy_accumulation_and_rain_reset(self):
        """Nesterov proxy must accumulate on dry days and reset to 0 on days with rain >= 3mm."""
        res = compute_fire_weather_indices(self.df)
        # Day 0 has 5.0 mm rain (>= 3.0 mm) -> resets to 0.0
        n0 = res.loc[0, "nesterov_proxy"]
        self.assertAlmostEqual(n0, 0.0)

        # Days 1 and 2 have 0mm rain -> strictly accumulating
        n1 = res.loc[1, "nesterov_proxy"]
        n2 = res.loc[2, "nesterov_proxy"]
        self.assertGreater(n1, 0.0)
        self.assertGreater(n2, n1)

        # Day 3 has 5.0 mm rain (>= 3.0 mm) -> must reset to 0.0
        n3 = res.loc[3, "nesterov_proxy"]
        self.assertAlmostEqual(n3, 0.0)

        # Day 4 has 0mm rain -> starts accumulating again from 0 + inc_day4
        n4 = res.loc[4, "nesterov_proxy"]
        self.assertGreater(n4, 0.0)
        self.assertLess(n4, n2)  # Reset occurred


class TestNeighborFireHistory(unittest.TestCase):
    """Tests for spatial-temporal neighbor fire history."""

    def setUp(self):
        # 3x3 spatial grid over 10 days
        dates = pd.date_range("2020-01-01", periods=10)
        rows = []
        for d in dates:
            for lat in (35.0, 35.5, 36.0):
                for lon in (3.0, 3.625, 4.25):
                    # Fire on day 4 (2020-01-05) at center cell (35.5, 3.625)
                    fires = 5 if (lat == 35.5 and lon == 3.625 and d == pd.Timestamp("2020-01-05")) else 0
                    rows.append({
                        "lat_cell": lat,
                        "lon_cell": lon,
                        "date": d,
                        "n_detections": fires,
                        "target": 1 if fires > 0 else 0,
                    })
        self.df = pd.DataFrame(rows)

    def test_shift1_zero_leakage_day_t(self):
        """Fires at day t must NOT appear in neighbor features on day t."""
        res = compute_neighbor_fire_history(self.df, lat_step=0.5, lon_step=0.625)
        center_cell = (self.df["lat_cell"] == 35.5) & (self.df["lon_cell"] == 3.625)

        # For corner cell (35.0, 3.0), which is a neighbor of the center:
        corner_cell = (self.df["lat_cell"] == 35.0) & (self.df["lon_cell"] == 3.0)

        # On 2020-01-05 (day of the fire), corner cell must NOT observe the fire
        # (It should be NaN because 7 prior days are not yet available)
        sub_d5 = res[corner_cell & (res["date"] == "2020-01-05")]
        self.assertTrue(np.isnan(sub_d5["neighbor_fires_7d"].iloc[0]))

        # On 2020-01-08 (day 7, 7 full days available: 2020-01-01 to 2020-01-07),
        # the fire from 2020-01-05 is now in the past window [t-7, t-1] -> observed as 5.0
        sub_d8 = res[corner_cell & (res["date"] == "2020-01-08")]
        self.assertAlmostEqual(sub_d8["neighbor_fires_7d"].iloc[0], 5.0)

    def test_target_insensitivity(self):
        """Output must be 100% identical regardless of what 'target' contains."""
        res1 = compute_neighbor_fire_history(self.df, lat_step=0.5, lon_step=0.625)

        # Corrupt target completely (invert, randomize, delete)
        df_corrupt = self.df.copy()
        df_corrupt["target"] = 1 - df_corrupt["target"]
        res2 = compute_neighbor_fire_history(df_corrupt, lat_step=0.5, lon_step=0.625)

        df_no_target = self.df.drop(columns=["target"])
        res3 = compute_neighbor_fire_history(df_no_target, lat_step=0.5, lon_step=0.625)

        pd.testing.assert_series_equal(res1["neighbor_fires_7d"], res2["neighbor_fires_7d"])
        pd.testing.assert_series_equal(res1["neighbor_fires_7d"], res3["neighbor_fires_7d"])

    def test_incomplete_history_produces_nan(self):
        """Incomplete rolling window must produce NaN, not 0."""
        res = compute_neighbor_fire_history(self.df, lat_step=0.5, lon_step=0.625)
        # First 6 days must have NaN for 7d window
        sub_early = res[res["date"] < "2020-01-08"]
        self.assertTrue(sub_early["neighbor_fires_7d"].isna().all())

    def test_history_df_populates_early_days(self):
        """Supplying prior historical observations enables early calculations without NaN."""
        # Create 7 prior days in December 2019
        prior_dates = pd.date_range("2019-12-25", "2019-12-31")
        prior_rows = []
        for d in prior_dates:
            for lat in (35.0, 35.5, 36.0):
                for lon in (3.0, 3.625, 4.25):
                    prior_rows.append({
                        "lat_cell": lat,
                        "lon_cell": lon,
                        "date": d,
                        "n_detections": 0,
                    })
        hist_df = pd.DataFrame(prior_rows)

        res = compute_neighbor_fire_history(
            self.df,
            history_df=hist_df,
            lat_step=0.5,
            lon_step=0.625,
        )
        # Now day 0 (2020-01-01) has full 7 prior days available from December 2019 -> not NaN
        sub_d0 = res[res["date"] == "2020-01-01"]
        self.assertTrue(sub_d0["neighbor_fires_7d"].notna().all())
        self.assertTrue(np.allclose(sub_d0["neighbor_fires_7d"], 0.0))


class TestBuildFeaturesV2(unittest.TestCase):
    """Tests for the master build_features_v2 dispatcher."""

    def setUp(self):
        dates = pd.date_range("2018-07-01", "2018-07-15")
        rows = []
        for d in dates:
            for lat in (35.0, 35.5):
                for lon in (3.0, 3.625):
                    rows.append({
                        "lat_cell": lat,
                        "lon_cell": lon,
                        "date": d,
                        "T2M": 25.0,
                        "T2M_MAX": 30.0,
                        "RH2M": 50.0,
                        "WS2M": 4.0,
                        "PRECTOTCORR": 0.0,
                        "GWETROOT": 0.5,
                        "n_detections": 0,
                        "target": 0,
                    })
        self.df = pd.DataFrame(rows)

    def test_all_12_features_added(self):
        """build_features_v2 with all families must add exactly 12 new features."""
        out = build_features_v2(self.df, lat_step=0.5, lon_step=0.625)
        new_cols = [c for c in out.columns if c not in self.df.columns]
        self.assertEqual(len(new_cols), 12)
        self.assertCountEqual(new_cols, ALL_NEW_FEATURES)
        self.assertEqual(len(out), len(self.df))

    def test_individual_families(self):
        """Check feature additions per family."""
        out_vpd = build_features_v2(self.df, feature_families=["vpd"])
        self.assertEqual(len([c for c in out_vpd.columns if c not in self.df.columns]), 2)

        out_anom = build_features_v2(self.df, feature_families=["anomalies"])
        self.assertEqual(len([c for c in out_anom.columns if c not in self.df.columns]), 3)

        out_idx = build_features_v2(self.df, feature_families=["indices"])
        self.assertEqual(len([c for c in out_idx.columns if c not in self.df.columns]), 4)

        out_nb = build_features_v2(self.df, feature_families=["neighbors"], lat_step=0.5, lon_step=0.625)
        self.assertEqual(len([c for c in out_nb.columns if c not in self.df.columns]), 3)

    def test_invalid_family_raises(self):
        """Unknown family name must raise ValueError."""
        with self.assertRaises(ValueError):
            build_features_v2(self.df, feature_families=["invalid_family"])


if __name__ == "__main__":
    unittest.main()
