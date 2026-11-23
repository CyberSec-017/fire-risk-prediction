"""Unit tests for the reusable data-quality checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_quality import (  # noqa: E402
    column_profile,
    duplicate_counts,
    dtype_checks,
    invalid_value_counts,
    validate_dataset,
)


def _clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_cell": [32.0, 32.5, 33.0],
            "lon_cell": [3.0, 3.625, 4.25],
            "T2M": [20.0, 21.5, 22.0],
            "status": ["ok", "ok", "ok"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )


def _issue_codes(report) -> list[str]:
    return [issue.code for issue in report.issues]


def test_column_profile_reports_missing_percentages_and_dtypes():
    frame = _clean_frame()
    frame.loc[1, "T2M"] = np.nan

    profile = column_profile(frame)
    by_column = profile.set_index("column")

    assert by_column.loc["T2M", "missing_count"] == 1
    assert by_column.loc["T2M", "missing_percent"] == pytest.approx(100.0 / 3.0, abs=1e-4)
    assert by_column.loc["date", "dtype"].startswith("datetime64")
    assert by_column["missing_count"].sum() == 1


def test_duplicate_counts_full_rows_and_declared_keys():
    frame = pd.DataFrame(
        {
            "lat_cell": [32.0, 32.0, 32.5],
            "lon_cell": [3.0, 3.0, 3.625],
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "T2M": [20.0, 21.0, 22.0],
        }
    )

    counts = duplicate_counts(
        frame,
        {"cell_date": ["lat_cell", "lon_cell", "date"]},
    )

    assert counts["all_columns"] == 0
    assert counts["cell_date"] == 1


def test_duplicate_counts_reject_unknown_key_columns():
    with pytest.raises(KeyError, match="missing column"):
        duplicate_counts(_clean_frame(), {"bad_key": ["not_a_column"]})


def test_dtype_checks_support_aliases_and_concrete_dtypes():
    frame = _clean_frame()

    results = dtype_checks(
        frame,
        {
            "T2M": "number",
            "date": "datetime",
            "status": "string",
            "lat_cell": "int64",
        },
    ).set_index("column")

    assert bool(results.loc["T2M", "matches"])
    assert bool(results.loc["date", "matches"])
    assert bool(results.loc["status", "matches"])
    assert not bool(results.loc["lat_cell", "matches"])


def test_invalid_value_counts_flags_ranges_sets_and_non_finite_numbers():
    frame = pd.DataFrame(
        {
            "GWETROOT": [0.2, 1.4, np.nan, np.inf],
            "status": ["ok", "bad", "ok", "also_bad"],
        }
    )

    invalid = invalid_value_counts(
        frame,
        numeric_ranges={"GWETROOT": (0.0, 1.0)},
        allowed_values={"status": {"ok"}},
    ).set_index("column")

    assert int(invalid.loc["GWETROOT", "invalid_count"]) == 2
    assert int(invalid.loc["status", "invalid_count"]) == 2
    assert invalid.loc["status", "invalid_percent"] == pytest.approx(50.0)


def test_validate_dataset_reports_all_check_categories():
    frame = _clean_frame()
    frame["T2M"] = [20.0, np.nan, 25.0]
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    report = validate_dataset(
        frame,
        name="fire_panel",
        required_columns=["lat_cell", "lon_cell", "T2M", "absent"],
        unique_keys={"cell_date": ["lat_cell", "lon_cell", "date"]},
        expected_dtypes={"lat_cell": "float64", "date": "datetime"},
        numeric_ranges={"T2M": (-50.0, 60.0)},
        allowed_values={"status": {"ok"}},
        expected_rows=3,
        min_rows=1,
        max_missing_percent=0.0,
        max_duplicate_rows=0,
    )

    codes = _issue_codes(report)
    assert "missing_columns" in codes
    assert "missing_values" in codes
    assert "duplicate_rows" in codes
    assert "duplicate_key" in codes
    assert "expected_rows" in codes
    assert "incorrect_dtype" not in codes
    assert report.ok is False
    assert report.duplicate_rows == 1
    assert report.summary_frame().loc[0, "status"] == "FAIL"
    assert report.summary_frame().loc[0, "rows"] == 4

    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["name"] == "fire_panel"
    assert payload["dimensions"] == {"rows": 4, "columns": 5}


def test_validate_dataset_accepts_a_clean_frame():
    frame = _clean_frame()

    report = validate_dataset(
        frame,
        name="fires_daily",
        required_columns=["lat_cell", "lon_cell", "date"],
        unique_keys={"cell_date": ["lat_cell", "lon_cell", "date"]},
        expected_dtypes={"date": "datetime", "T2M": "number"},
        numeric_ranges={"T2M": (-50.0, 60.0)},
        allowed_values={"status": {"ok"}},
        expected_rows=3,
        expected_columns=5,
    )

    assert report.issues == ()
    assert report.ok is True
    assert report.summary_frame().loc[0, "status"] == "PASS"
    assert report.summary_frame().loc[0, "missing_percent"] == 0.0
    assert report.issues_frame().empty


def test_validate_dataset_dimension_gates_are_independent():
    frame = _clean_frame()

    too_wide = validate_dataset(frame, name="wide", max_columns=4)
    too_tall = validate_dataset(frame, name="tall", max_rows=2)
    empty = validate_dataset(frame.iloc[0:0], name="empty")

    assert _issue_codes(too_wide) == ["max_columns"]
    assert _issue_codes(too_tall) == ["max_rows"]
    assert "min_rows" in _issue_codes(empty)
