"""AppTest coverage for the reusable Streamlit data-quality summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.data_quality_view import render_data_quality_summary  # noqa: E402
from data_quality import validate_dataset  # noqa: E402


def _summary_app(root):
    import sys

    import pandas as pd

    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from dashboard.data_quality_view import render_data_quality_summary
    from data_quality import validate_dataset

    frame = pd.DataFrame(
        {
            "lat_cell": [32.0, 32.5],
            "lon_cell": [3.0, 3.625],
            "GWETROOT": [0.4, 1.6],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    report = validate_dataset(
        frame,
        name="sample_panel",
        expected_dtypes={"date": "datetime"},
        numeric_ranges={"GWETROOT": (0.0, 1.0)},
    )
    render_data_quality_summary(report, title="Quality report", expanded=True)


def test_data_quality_summary_renders_dimensions_types_and_issues():
    app = AppTest.from_function(_summary_app, default_timeout=30, args=(ROOT,))
    app.run()

    assert not app.exception, app.exception
    tables = [frame.value for frame in app.dataframe]
    column_sets = [set(table.columns) for table in tables]

    expected_column_sets = (
        {"dataset", "rows", "columns", "missing_percent", "status"},
        {"column", "dtype", "missing_count", "missing_percent"},
        {"column", "expected", "actual", "matches"},
        {"column", "invalid_count", "invalid_percent", "rule"},
        {"code", "level", "message", "column", "count"},
    )
    for expected in expected_column_sets:
        assert any(expected <= actual for actual in column_sets), expected
