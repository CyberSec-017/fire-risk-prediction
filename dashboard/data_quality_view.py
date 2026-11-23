"""Streamlit display helpers for reusable data-quality reports.

The profiling logic lives in ``src/data_quality.py`` and is pure and testable.
This module only renders those reports with native Streamlit widgets.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_quality import DatasetQualityReport, validate_dataset  # noqa: E402


def build_pipeline_quality_reports() -> tuple[DatasetQualityReport, ...]:
    """Profile the pipeline's derived parquet datasets when they are available."""
    import config  # noqa: PLC0415 - imported lazily so the display module stays importable

    panel_specs: dict[str, object] = {
        "required_columns": [
            "lat_cell",
            "lon_cell",
            "date",
            "T2M",
            "T2M_MAX",
            "RH2M",
            "WS2M",
            "PRECTOTCORR",
            "GWETROOT",
            "target",
        ],
        "unique_keys": {"cell_date": ["lat_cell", "lon_cell", "date"]},
        "expected_dtypes": {"date": "datetime", "target": "number"},
        "numeric_ranges": {
            "lat_cell": (32.0, 37.5),
            "lon_cell": (-2.0, 9.0),
            "T2M": (-90.0, 70.0),
            "T2M_MAX": (-90.0, 70.0),
            "RH2M": (0.0, 100.0),
            "WS2M": (0.0, 150.0),
            "PRECTOTCORR": (0.0, 1000.0),
            "GWETROOT": (0.0, 1.0),
            "dry_days": (0.0, 365.0),
            "doy_sin": (-1.0, 1.0),
            "doy_cos": (-1.0, 1.0),
            "burned": (0.0, 1.0),
            "target": (0.0, 1.0),
            "n_detections": (0.0, None),
            "total_frp": (0.0, None),
        },
        "min_columns": 20,
        # GWETROOT is legitimately missing for ~7% of cells in the shipped panel.
        "max_missing_percent": 10.0,
        "max_duplicate_rows": 0,
    }
    fires_specs: dict[str, object] = {
        "required_columns": ["lat_cell", "lon_cell", "date", "n_detections", "total_frp"],
        "unique_keys": {"cell_date": ["lat_cell", "lon_cell", "date"]},
        "expected_dtypes": {"date": "datetime", "n_detections": "number"},
        "numeric_ranges": {
            "lat_cell": (32.0, 37.5),
            "lon_cell": (-2.0, 9.0),
            "n_detections": (0.0, None),
            "total_frp": (0.0, None),
        },
        "min_columns": 5,
        "max_missing_percent": 0.0,
        "max_duplicate_rows": 0,
    }

    reports: list[DatasetQualityReport] = []
    for name, path, specs in (
        ("fire_panel", config.FIRE_PANEL, panel_specs),
        ("fires_daily", config.FIRES_DAILY, fires_specs),
    ):
        if Path(path).exists():
            frame = pd.read_parquet(path)
            reports.append(validate_dataset(frame, name=name, **specs))
    return tuple(reports)


def render_data_quality_summary(
    reports: DatasetQualityReport | Iterable[DatasetQualityReport],
    *,
    title: str = "Data quality",
    expanded: bool = False,
    container: st.delta_generator.DeltaGenerator | None = None,
) -> None:
    """Render one or more quality reports with a compact overview and details."""
    items: Sequence[DatasetQualityReport] = (
        [reports] if isinstance(reports, DatasetQualityReport) else list(reports)
    )
    target = container if container is not None else st

    with target.expander(title, expanded=expanded):
        if not items:
            st.info("No data-quality reports are available for this session.")
            return

        overview = pd.concat(
            [report.summary_frame() for report in items], ignore_index=True
        )
        st.dataframe(
            overview,
            hide_index=True,
            width="stretch",
            alt="Overview of dataset dimensions and data-quality check results",
        )

        for report in items:
            st.caption(f"**{report.name}**")
            col_rows, col_columns, col_duplicates = st.columns(3)
            col_rows.metric("Rows", f"{report.n_rows:,}")
            col_columns.metric("Columns", report.n_columns)
            col_duplicates.metric("Duplicate rows", report.duplicate_rows)

            st.dataframe(
                report.columns,
                hide_index=True,
                width="stretch",
                alt=f"Column types and missing-value percentages for {report.name}",
            )
            if not report.dtypes.empty:
                st.dataframe(
                    report.dtypes,
                    hide_index=True,
                    width="stretch",
                    alt=f"Expected versus observed column dtypes for {report.name}",
                )
            if not report.invalid_values.empty:
                st.dataframe(
                    report.invalid_values,
                    hide_index=True,
                    width="stretch",
                    alt=f"Invalid-value counts by column for {report.name}",
                )
            if not report.issues_frame().empty:
                st.dataframe(
                    report.issues_frame(),
                    hide_index=True,
                    width="stretch",
                    alt=f"Data-quality issues detected in {report.name}",
                )
