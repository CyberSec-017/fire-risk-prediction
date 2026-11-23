"""Reusable, dataset-agnostic data-quality checks.

The impact module already owns its domain-specific validator
(``scripts/validate_impact_layers.py``) and ``app_core.py`` already validates the
trained model/card contract.  This module deliberately covers the remaining,
generic layer: missing values, duplicates, column dtypes, invalid values and
dataset dimensions for any pandas DataFrame used by the pipeline.

All checks are pure functions. Nothing is repaired, dropped or coerced in the
source frame, and no model artifact or production prediction path is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

DtypeExpectation = str | Callable[[pd.Series], bool]


@dataclass(frozen=True)
class QualityIssue:
    """One explicit data-quality finding."""

    code: str
    message: str
    level: str = "error"
    column: str | None = None
    count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level,
            "message": self.message,
            "column": self.column,
            "count": self.count,
        }


@dataclass(frozen=True)
class DatasetQualityReport:
    """A reusable summary of one dataset's quality."""

    name: str
    dimensions: dict[str, int]
    columns: pd.DataFrame
    duplicates: dict[str, int]
    dtypes: pd.DataFrame
    invalid_values: pd.DataFrame
    issues: tuple[QualityIssue, ...]

    @property
    def n_rows(self) -> int:
        return self.dimensions["rows"]

    @property
    def n_columns(self) -> int:
        return self.dimensions["columns"]

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def duplicate_rows(self) -> int:
        return self.duplicates.get("all_columns", 0)

    def summary_frame(self) -> pd.DataFrame:
        """One-row overview suitable for ``st.dataframe``/``st.table``."""
        total_cells = self.n_rows * self.n_columns
        missing_count = (
            int(self.columns["missing_count"].sum()) if not self.columns.empty else 0
        )
        invalid_count = (
            int(self.invalid_values["invalid_count"].sum())
            if not self.invalid_values.empty
            else 0
        )
        row = {
            "dataset": self.name,
            "rows": self.n_rows,
            "columns": self.n_columns,
            "missing_values": missing_count,
            "missing_percent": round(100.0 * missing_count / total_cells, 4)
            if total_cells
            else 0.0,
            "duplicate_rows": self.duplicate_rows,
            "invalid_values": invalid_count,
            "issues": len(self.issues),
            "status": "PASS" if self.ok else "FAIL",
        }
        return pd.DataFrame([row])

    def issues_frame(self) -> pd.DataFrame:
        """Tabular issues with a stable schema, even when there are none."""
        columns = ["code", "level", "message", "column", "count"]
        if not self.issues:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([issue.as_dict() for issue in self.issues])[columns]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation for reports and APIs."""
        return {
            "name": self.name,
            "ok": self.ok,
            "dimensions": dict(self.dimensions),
            "columns": self.columns.to_dict(orient="records"),
            "duplicates": dict(self.duplicates),
            "dtypes": self.dtypes.to_dict(orient="records"),
            "invalid_values": self.invalid_values.to_dict(orient="records"),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _percent(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value percentages, dtypes and cardinality per column."""
    rows: list[dict[str, Any]] = []
    for position, label in enumerate(df.columns):
        series = df.iloc[:, position]
        missing = int(series.isna().sum())
        try:
            distinct = int(series.nunique(dropna=True))
        except TypeError:  # unhashable cell values (lists/dicts)
            distinct = int(series.astype(str).nunique())
        rows.append(
            {
                "column": str(label),
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_percent": _percent(missing, len(df)),
                "distinct_count": distinct,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["column", "dtype", "missing_count", "missing_percent", "distinct_count"],
    )


def duplicate_counts(
    df: pd.DataFrame,
    subsets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, int]:
    """Count fully duplicated rows and caller-declared unique-key duplicates."""
    counts: dict[str, int] = {"all_columns": int(df.duplicated().sum())}
    for name, columns in (subsets or {}).items():
        keys = list(columns)
        if not keys:
            continue
        missing = [column for column in keys if column not in df.columns]
        if missing:
            raise KeyError(f"Duplicate key '{name}' references missing column(s): {missing}")
        counts[str(name)] = int(df.duplicated(subset=keys).sum())
    return counts


def _matches_dtype(series: pd.Series, expectation: DtypeExpectation) -> bool:
    if callable(expectation):
        return bool(expectation(series))

    key = str(expectation).strip().lower()
    checkers: dict[str, Callable[[pd.Series], bool]] = {
        "number": is_numeric_dtype,
        "numeric": is_numeric_dtype,
        "int": is_integer_dtype,
        "integer": is_integer_dtype,
        "float": is_float_dtype,
        "bool": is_bool_dtype,
        "boolean": is_bool_dtype,
        "datetime": is_datetime64_any_dtype,
        "date": is_datetime64_any_dtype,
        "str": is_string_dtype,
        "string": is_string_dtype,
    }
    if key in checkers:
        return bool(checkers[key](series))
    return str(series.dtype).lower() == str(expectation).lower()


def dtype_checks(
    df: pd.DataFrame,
    expected_dtypes: Mapping[str, DtypeExpectation] | None = None,
) -> pd.DataFrame:
    """Compare declared dtypes with the observed pandas dtypes."""
    rows: list[dict[str, Any]] = []
    for column, expectation in (expected_dtypes or {}).items():
        if column not in df.columns:
            continue
        series = df[column]
        rows.append(
            {
                "column": str(column),
                "expected": str(expectation)
                if not callable(expectation)
                else getattr(expectation, "__name__", "callable"),
                "actual": str(series.dtype),
                "matches": _matches_dtype(series, expectation),
            }
        )
    return pd.DataFrame(rows, columns=["column", "expected", "actual", "matches"])


def invalid_value_counts(
    df: pd.DataFrame,
    numeric_ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
    allowed_values: Mapping[str, Collection[Any]] | None = None,
) -> pd.DataFrame:
    """Count non-finite/out-of-range numbers and values outside allowed sets.

    Missing values are excluded here: they are reported separately by
    :func:`column_profile` so the two problems can be diagnosed independently.
    """
    rules: dict[str, dict[str, Any]] = {}
    for column, bounds in (numeric_ranges or {}).items():
        rules.setdefault(str(column), {})["range"] = tuple(bounds)
    for column, values in (allowed_values or {}).items():
        rules.setdefault(str(column), {})["allowed"] = set(values)

    rows: list[dict[str, Any]] = []
    for column, rule in rules.items():
        if column not in df.columns:
            continue
        series = df[column]
        present = np.asarray(series.notna(), dtype=bool)
        invalid = np.zeros(len(df), dtype=bool)
        descriptions: list[str] = []

        if "range" in rule:
            lower, upper = rule["range"]
            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric.to_numpy(dtype="float64", na_value=np.nan)
            invalid |= present & ~np.isfinite(values)
            if lower is not None:
                invalid |= ~np.isnan(values) & (values < float(lower))
            if upper is not None:
                invalid |= ~np.isnan(values) & (values > float(upper))
            descriptions.append(f"range[{lower}, {upper}]")

        if "allowed" in rule:
            allowed = list(rule["allowed"])
            valid = np.asarray(series.isin(allowed).fillna(False), dtype=bool)
            invalid |= present & ~valid
            preview = ", ".join(str(value) for value in sorted(map(str, allowed))[:5])
            descriptions.append(f"allowed_values({preview})")

        count = int(invalid.sum())
        rows.append(
            {
                "column": column,
                "invalid_count": count,
                "invalid_percent": _percent(count, len(df)),
                "rule": "; ".join(descriptions),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["column", "invalid_count", "invalid_percent", "rule"],
    )


def _dimension_issues(
    name: str,
    n_rows: int,
    n_columns: int,
    *,
    expected_rows: int | None,
    expected_columns: int | None,
    min_rows: int | None,
    min_columns: int | None,
    max_rows: int | None,
    max_columns: int | None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    expectations = (
        ("expected_rows", expected_rows, n_rows, "rows"),
        ("expected_columns", expected_columns, n_columns, "columns"),
        ("min_rows", min_rows, n_rows, "rows"),
        ("min_columns", min_columns, n_columns, "columns"),
        ("max_rows", max_rows, n_rows, "rows"),
        ("max_columns", max_columns, n_columns, "columns"),
    )
    for code, limit, actual, axis in expectations:
        if limit is None:
            continue
        failed = (
            actual != limit
            if code.startswith("expected_")
            else actual < limit
            if code.startswith("min_")
            else actual > limit
        )
        if failed:
            operator = "==" if code.startswith("expected_") else ">=" if code.startswith("min_") else "<="
            issues.append(
                QualityIssue(
                    code=code,
                    message=(
                        f"{name} has {actual} {axis}, but the {code.replace('_', ' ')} "
                        f"requirement is {operator} {limit}."
                    ),
                    count=actual,
                )
            )
    return issues


def validate_dataset(
    df: pd.DataFrame,
    name: str = "dataset",
    *,
    required_columns: Sequence[str] = (),
    unique_keys: Mapping[str, Sequence[str]] | None = None,
    expected_dtypes: Mapping[str, DtypeExpectation] | None = None,
    numeric_ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
    allowed_values: Mapping[str, Collection[Any]] | None = None,
    expected_rows: int | None = None,
    expected_columns: int | None = None,
    min_rows: int | None = 1,
    min_columns: int | None = 1,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_missing_percent: float | None = 0.0,
    max_duplicate_rows: int | None = 0,
) -> DatasetQualityReport:
    """Profile a DataFrame and emit explicit issues for every failed check.

    ``None`` limits disable that gate.  By default an empty frame, any missing
    value, any duplicated row, dtype mismatches and invalid values are treated
    as errors.  Callers can relax individual gates for datasets where those
    states are legitimate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_dataset expects a pandas DataFrame.")

    issues: list[QualityIssue] = []
    missing_columns = [str(column) for column in required_columns if column not in df.columns]
    if missing_columns:
        issues.append(
            QualityIssue(
                code="missing_columns",
                message=f"{name} is missing required column(s): {missing_columns}.",
            )
        )

    dimensions = {"rows": int(len(df.index)), "columns": int(len(df.columns))}
    issues.extend(
        _dimension_issues(
            name,
            dimensions["rows"],
            dimensions["columns"],
            expected_rows=expected_rows,
            expected_columns=expected_columns,
            min_rows=min_rows,
            min_columns=min_columns,
            max_rows=max_rows,
            max_columns=max_columns,
        )
    )

    profile = column_profile(df)
    if max_missing_percent is not None:
        for row in profile.itertuples(index=False):
            if row.missing_percent > max_missing_percent:
                issues.append(
                    QualityIssue(
                        code="missing_values",
                        message=(
                            f"{name}.{row.column} has {row.missing_count} missing value(s) "
                            f"({row.missing_percent:.4f}%), exceeding "
                            f"{max_missing_percent:.4f}%."
                        ),
                        column=str(row.column),
                        count=int(row.missing_count),
                    )
                )

    duplicates = duplicate_counts(df, unique_keys)
    if max_duplicate_rows is not None and duplicates["all_columns"] > max_duplicate_rows:
        issues.append(
            QualityIssue(
                code="duplicate_rows",
                message=(
                    f"{name} has {duplicates['all_columns']} duplicated row(s), exceeding "
                    f"the limit of {max_duplicate_rows}."
                ),
                count=duplicates["all_columns"],
            )
        )
    for key, count in duplicates.items():
        if key == "all_columns" or count == 0:
            continue
        issues.append(
            QualityIssue(
                code="duplicate_key",
                message=f"{name} has {count} duplicated row(s) for unique key '{key}'.",
                column=key,
                count=count,
            )
        )

    dtypes = dtype_checks(df, expected_dtypes)
    for row in dtypes.itertuples(index=False):
        if not row.matches:
            issues.append(
                QualityIssue(
                    code="incorrect_dtype",
                    message=(
                        f"{name}.{row.column} has dtype {row.actual}, expected {row.expected}."
                    ),
                    column=str(row.column),
                )
            )

    invalid = invalid_value_counts(df, numeric_ranges, allowed_values)
    for row in invalid.itertuples(index=False):
        if row.invalid_count:
            issues.append(
                QualityIssue(
                    code="invalid_values",
                    message=(
                        f"{name}.{row.column} has {row.invalid_count} invalid value(s) "
                        f"({row.invalid_percent:.4f}%): {row.rule}."
                    ),
                    column=str(row.column),
                    count=int(row.invalid_count),
                )
            )

    return DatasetQualityReport(
        name=name,
        dimensions=dimensions,
        columns=profile,
        duplicates=duplicates,
        dtypes=dtypes,
        invalid_values=invalid,
        issues=tuple(issues),
    )
