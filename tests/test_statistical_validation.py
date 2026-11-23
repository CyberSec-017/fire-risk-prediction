"""Unit coverage for the four statistical hypothesis tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import statistical_validation as sv  # noqa: E402


def _season_detection_frame() -> pd.DataFrame:
    rows = []
    rates = {"DJF (hiver)": 0.05, "MAM (printemps)": 0.10,
             "JJA (été)": 0.35, "SON (automne)": 0.20}
    for season, rate in rates.items():
        n = 100
        positives = int(round(n * rate))
        rows.extend([{"season": season, "burned": 1}] * positives)
        rows.extend([{"season": season, "burned": 0}] * (n - positives))
    return pd.DataFrame(rows)


def test_chi_square_detects_association_and_checks_expected_counts():
    result = sv.chi_square_season_detection(_season_detection_frame())

    assert result["test_id"] == "chi2_season_detection"
    assert result["status"] == "ok"
    assert result["statistic"] > 0
    assert result["p_value"] is not None and result["p_value"] < 0.05
    assert result["significant"] is True
    assert result["effect_size"]["name"] == "V de Cramér"
    assert result["contingency"]["rows"] == list(sv.SEASON_ORDER)
    expected_check = result["assumptions"][0]
    assert expected_check["passed"] is True
    assert "Effectif attendu minimal" in expected_check["detail"]
    assert "causalité" in result["interpretation"]


def test_chi_square_handles_degenerate_tables_safely():
    frame = pd.DataFrame({"season": ["DJF (hiver)"] * 20, "burned": [0] * 20})
    result = sv.chi_square_season_detection(frame)

    assert result["status"] == "insufficient_data"
    assert result["p_value"] is None
    assert result["significant"] is None
    assert result["p_display"] == "—"


def test_kruskal_detects_group_shift_and_summarises_each_group():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "group": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
        "value": np.concatenate([
            rng.normal(10, 1, 40), rng.normal(20, 1, 40), rng.normal(30, 1, 40)
        ]),
    })
    result = sv.kruskal_groups(frame, group_col="group", value_col="value")

    assert result["status"] == "ok"
    assert result["p_value"] < 0.05
    assert result["effect_size"]["name"] == "epsilon²"
    assert [item["group"] for item in result["group_summaries"]] == ["A", "B", "C"]
    assert all(item["n"] == 40 for item in result["group_summaries"])
    assert "rangs" in result["interpretation"]


def test_kruskal_requires_at_least_two_groups():
    frame = pd.DataFrame({"season": ["DJF (hiver)"] * 20,
                          "T2M_MAX": np.arange(20, dtype=float)})
    result = sv.kruskal_groups(frame)

    assert result["status"] == "insufficient_data"
    assert result["statistic"] is None


def test_shapiro_uses_reproducible_bounded_sample():
    values = pd.Series(np.linspace(-10, 40, 20_000))
    first = sv.shapiro_normality(values, max_sample=500, seed=11)
    second = sv.shapiro_normality(values, max_sample=500, seed=11)

    assert first["status"] == "ok"
    assert first["sample"]["tested_rows"] == 500
    assert first["sample"]["cap"] == 500
    assert first["p_value"] == second["p_value"]
    assert first["histogram"]["normal_expected_counts"]
    assert 0.0 <= float(first["statistic"]) <= 1.0
    assert first["assumptions"][0]["passed"] is True


def test_shapiro_handles_non_numeric_or_tiny_inputs():
    empty = sv.shapiro_normality(["a", None, "b"])
    assert empty["status"] == "insufficient_data"
    assert empty["p_value"] is None

    tiny = sv.shapiro_normality([1, 2], max_sample=5000)
    assert tiny["status"] == "insufficient_data"


def test_mann_whitney_reports_direction_and_effect_size():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "burned": [0] * 60 + [1] * 25,
        "T2M_MAX": np.concatenate([rng.normal(20, 2, 60), rng.normal(32, 2, 25)]),
    })
    result = sv.mann_whitney_groups(frame)

    assert result["status"] == "ok"
    assert result["p_value"] < 0.05
    assert result["effect_size"]["name"] == "rang-bisérial"
    assert result["effect_size"]["value"] < 0
    assert len(result["group_summaries"]) == 2
    assert "températures plus élevées les jours avec détection" in result["interpretation"]


def test_mann_whitney_handles_small_groups():
    frame = pd.DataFrame({"burned": [0, 0, 1], "T2M_MAX": [10.0, 11.0, 30.0]})
    result = sv.mann_whitney_groups(frame)

    assert result["status"] == "insufficient_data"
    assert result["p_value"] is None


def test_build_report_bundles_four_contextual_tests():
    dates = pd.to_datetime(["2021-01-15", "2021-04-15", "2021-07-15", "2021-10-15"] * 25)
    frame = pd.DataFrame({
        "date": dates,
        "lat_cell": [36.0] * 100,
        "lon_cell": [3.0] * 100,
        "T2M_MAX": np.tile([8.0, 18.0, 34.0, 22.0], 25),
        "burned": np.tile([0, 1, 1, 0], 25),
    })
    report = sv.build_statistical_report(frame, alpha=0.05, seed=1)

    assert [test["test_id"] for test in report["tests"]] == list(sv.EXPECTED_TEST_IDS)
    assert report["dataset"]["rows"] == 100
    assert report["dataset"]["date_min"] == "2021-01-15"
    assert report["protocol"]["shapiro_sample_cap"] == sv.SHAPIRO_SAMPLE_CAP
    for test in report["tests"]:
        assert test["null_hypothesis"].startswith("H0")
        assert test["alternative_hypothesis"].startswith("H1")
        assert test["why_appropriate"]
        assert isinstance(test["assumptions"], list)
        assert "Décision" in sv.interpretation_text(test, 0.05)
    sv.validate_statistical_report(report)


def test_invalid_alpha_falls_back_to_documented_default():
    result = sv.chi_square_season_detection(_season_detection_frame(), alpha=2.0)

    assert result["alpha"] == sv.DEFAULT_ALPHA
    assert "alpha" in result["warning"]
    assert result["status"] == "ok"


def test_validate_report_rejects_missing_tests():
    with pytest.raises(ValueError, match="quatre tests"):
        sv.validate_statistical_report({"generated_at": "x", "protocol": {},
                                        "dataset": {}, "tests": [], "limitations": []})
