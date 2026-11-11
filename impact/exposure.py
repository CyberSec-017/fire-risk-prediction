"""Exposure calculations for the Wildfire Impact Assessment module.

All functions here are pure pandas/numpy transformations on an
already-built exposure table.  They perform no I/O and no Streamlit calls.

Data-integrity contract
-----------------------
* Missing required columns raise :class:`MissingColumnsError`; they are never
  replaced by zeros.
* Non-numeric, non-finite or negative values raise
  :class:`IncompleteDataError` / :class:`InconsistentDataError`.
* Land-cover totals that exceed the cell area by more than 5 % raise
  :class:`InconsistentDataError` instead of being clipped silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from impact import config


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ImpactDataError(ValueError):
    """Base class for explicit impact-data failures."""


class MissingColumnsError(ImpactDataError):
    """Raised when an exposure dimension lacks required columns."""

    def __init__(self, dimension: str, missing: Iterable[str]):
        self.dimension = dimension
        self.missing = tuple(missing)
        super().__init__(
            f"{dimension} exposure is unavailable: missing required column(s) "
            f"{list(self.missing)}. Missing data is never replaced with zeros."
        )


class IncompleteDataError(ImpactDataError):
    """Raised when required values are non-numeric, missing or infinite."""

    def __init__(self, dimension: str, columns: Iterable[str]):
        self.dimension = dimension
        self.columns = tuple(columns)
        super().__init__(
            f"{dimension} exposure is unavailable: required values are missing, "
            f"non-numeric or non-finite in {list(self.columns)}."
        )


class InconsistentDataError(ImpactDataError):
    """Raised when exposure values violate physical or schema constraints."""


@dataclass(frozen=True)
class ExposureResult:
    """Prepared exposure table plus per-dimension availability."""

    status: str
    frame: pd.DataFrame | None
    message: str
    dimensions: Mapping[str, Mapping[str, object]]

    @property
    def available(self) -> bool:
        return self.status == config.STATUS_AVAILABLE

    @property
    def unavailable(self) -> bool:
        return not self.available


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def cell_bounds(
    lat_cell: float,
    lon_cell: float,
) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lat_max, lon_min, lon_max)`` for one grid cell.

    Panel coordinates are NASA POWER cell centres, so each cell extends half
    a grid step around its centre.
    """
    half_lat = config.MODEL_LAT_STEP / 2.0
    half_lon = config.MODEL_LON_STEP / 2.0
    return (
        float(lat_cell) - half_lat,
        float(lat_cell) + half_lat,
        float(lon_cell) - half_lon,
        float(lon_cell) + half_lon,
    )


def cell_area_km2(lat_cell: float, lon_cell: float) -> float:
    """Exact spherical area of a grid cell in km²."""
    lat_min, lat_max, lon_min, lon_max = cell_bounds(lat_cell, lon_cell)
    delta_lon = np.radians(lon_max - lon_min)
    delta_sin = np.sin(np.radians(lat_max)) - np.sin(np.radians(lat_min))
    return float(config.EARTH_RADIUS_KM**2 * delta_lon * delta_sin)


def is_on_model_grid(
    lat: float,
    lon: float,
    atol: float = 1e-6,
) -> bool:
    """True when a coordinate lies on the model's training grid."""
    lat = float(lat)
    lon = float(lon)
    lat_grid = round(lat / config.MODEL_LAT_STEP) * config.MODEL_LAT_STEP
    lon_grid = round(lon / config.MODEL_LON_STEP) * config.MODEL_LON_STEP
    on_grid = abs(lat - lat_grid) <= atol and abs(lon - lon_grid) <= atol
    inside = (
        config.MODEL_LAT_MIN - atol <= lat <= config.MODEL_LAT_MAX + atol
        and config.MODEL_LON_MIN - atol <= lon <= config.MODEL_LON_MAX + atol
    )
    return bool(on_grid and inside)


def grid_alignment_issues(frame: pd.DataFrame) -> list[str]:
    """Return human-readable problems with a frame's cell coordinates."""
    missing = [c for c in ("lat_cell", "lon_cell") if c not in frame.columns]
    if missing:
        return [f"missing coordinate column(s): {missing}"]
    issues: list[str] = []
    lat = pd.to_numeric(frame["lat_cell"], errors="coerce")
    lon = pd.to_numeric(frame["lon_cell"], errors="coerce")
    if lat.isna().any() or lon.isna().any():
        issues.append("cell coordinates contain missing or non-numeric values")
    duplicates = frame.duplicated(subset=["lat_cell", "lon_cell"]).sum()
    if duplicates:
        issues.append(f"{int(duplicates)} duplicate cell coordinate pair(s)")
    inside = (
        lat.between(config.MODEL_LAT_MIN, config.MODEL_LAT_MAX)
        & lon.between(config.MODEL_LON_MIN, config.MODEL_LON_MAX)
    )
    if (~inside.fillna(False)).any():
        issues.append(
            f"{int((~inside.fillna(False)).sum())} cell(s) fall outside the "
            "model's geographic bounds"
        )
    return issues


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    dimension: str,
) -> None:
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise ImpactDataError(f"{dimension} exposure requires a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise MissingColumnsError(dimension, missing)


def _numeric_block(
    frame: pd.DataFrame,
    columns: Iterable[str],
    dimension: str,
) -> pd.DataFrame:
    columns = list(columns)
    try:
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise IncompleteDataError(dimension, columns) from exc
    bad = numeric.isna() | np.isinf(numeric)
    if bad.to_numpy().any():
        offending = [column for column in columns if bad[column].any()]
        raise IncompleteDataError(dimension, offending)
    return numeric


def _require_positive_area(frame: pd.DataFrame, dimension: str) -> pd.Series:
    area = pd.to_numeric(frame[config.CELL_AREA_COLUMN], errors="coerce")
    if area.isna().any() or np.isinf(area).any():
        raise IncompleteDataError(dimension, [config.CELL_AREA_COLUMN])
    if (area <= 0).any():
        raise InconsistentDataError(
            f"{dimension} exposure is unavailable: {config.CELL_AREA_COLUMN} "
            "must be strictly positive."
        )
    return area


def _require_non_negative(
    numeric: pd.DataFrame,
    dimension: str,
) -> None:
    negative = (numeric < 0).any()
    offending = [column for column in numeric.columns if bool(negative[column])]
    if offending:
        raise InconsistentDataError(
            f"{dimension} exposure is unavailable: negative values in "
            f"{offending}."
        )


# ---------------------------------------------------------------------------
# Dimension calculations
# ---------------------------------------------------------------------------
def environmental_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fuel/land-cover metrics to a cell exposure frame.

    Adds: ``fuel_area_km2``, ``non_fuel_area_km2``, ``landcover_total_km2``,
    ``landcover_coverage_fraction``, ``fuel_fraction``, ``forest_fraction`` and
    ``landcover_complete``.
    """
    _require_columns(frame, config.ENV_REQUIRED_COLUMNS, "Environmental")
    out = frame.copy()
    numeric = _numeric_block(out, config.ENV_REQUIRED_COLUMNS, "Environmental")
    _require_non_negative(numeric, "Environmental")
    area = _require_positive_area(out, "Environmental")

    flammable_columns = [
        config.LC_AREA_COLUMNS[key] for key in config.FLAMMABLE_LC_KEYS
    ]
    non_fuel_columns = [
        column
        for key, column in config.LC_AREA_COLUMNS.items()
        if key not in config.FLAMMABLE_LC_KEYS
    ]

    fuel_area = numeric[flammable_columns].sum(axis=1)
    non_fuel_area = numeric[non_fuel_columns].sum(axis=1)
    total_area = fuel_area + non_fuel_area

    if (total_area > area * 1.05).any():
        raise InconsistentDataError(
            "Environmental exposure is unavailable: land-cover class areas "
            "exceed the cell area by more than 5 %."
        )

    for column in config.LC_AREA_COLUMNS.values():
        out[column] = numeric[column]

    coverage = (total_area / area).clip(lower=0.0, upper=1.0)
    out["fuel_area_km2"] = fuel_area.astype(float)
    out["non_fuel_area_km2"] = non_fuel_area.astype(float)
    out["landcover_total_km2"] = total_area.astype(float)
    out["landcover_coverage_fraction"] = coverage.astype(float)
    out["fuel_fraction"] = (fuel_area / area).clip(lower=0.0, upper=1.0)
    out["forest_fraction"] = (
        numeric[config.LC_AREA_COLUMNS["tree"]] / area
    ).clip(lower=0.0, upper=1.0)
    out["landcover_complete"] = coverage >= 0.95
    return out


def population_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Add population-density and settlement availability metrics."""
    _require_columns(frame, config.POP_REQUIRED_COLUMNS, "Population")
    out = frame.copy()
    numeric = _numeric_block(out, config.POP_REQUIRED_COLUMNS, "Population")
    _require_non_negative(numeric, "Population")
    area = _require_positive_area(out, "Population")

    for column in config.POP_REQUIRED_COLUMNS:
        out[column] = numeric[column]

    out["pop_density_per_km2"] = numeric["pop_cell"] / area

    settlement_columns = [
        column
        for column in config.OPTIONAL_POP_COLUMNS
        if column in out.columns
    ]
    if settlement_columns:
        settlement_numeric = out[settlement_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        known = settlement_numeric.notna()
        if ((settlement_numeric < 0) & known).to_numpy().any():
            raise InconsistentDataError(
                "Population (settlements) exposure is unavailable: negative "
                f"values in {settlement_columns}."
            )
        for column in settlement_columns:
            out[column] = settlement_numeric[column]
        settlement_complete = bool(known.to_numpy().all())
    else:
        settlement_complete = False
    out["settlement_data_available"] = bool(settlement_complete)
    return out


def infrastructure_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    """Add road/building/facility density metrics."""
    _require_columns(frame, config.INFRA_REQUIRED_COLUMNS, "Infrastructure")
    out = frame.copy()
    numeric = _numeric_block(out, config.INFRA_REQUIRED_COLUMNS, "Infrastructure")
    _require_non_negative(numeric, "Infrastructure")
    area = _require_positive_area(out, "Infrastructure")

    for column in config.INFRA_REQUIRED_COLUMNS:
        out[column] = numeric[column]

    built_fraction = (numeric["building_footprint_km2"] / area).clip(
        lower=0.0, upper=1.0
    )
    if (numeric["building_footprint_km2"] > area * 1.05).any():
        raise InconsistentDataError(
            "Infrastructure exposure is unavailable: building footprint area "
            "exceeds the cell area by more than 5 %."
        )

    out["road_density_km_per_km2"] = numeric["road_km_total"] / area
    out["building_density_per_km2"] = numeric["building_count"] / area
    out["built_fraction"] = built_fraction
    out["critical_density_per_km2"] = numeric["critical_count_total"] / area

    detail_columns = [
        column
        for column in config.OPTIONAL_INFRA_COLUMNS
        if column in out.columns
    ]
    if detail_columns:
        detail_numeric = out[detail_columns].apply(pd.to_numeric, errors="coerce")
        known = detail_numeric.notna()
        if ((detail_numeric < 0) & known).to_numpy().any():
            raise InconsistentDataError(
                "Infrastructure (detail) exposure is unavailable: negative "
                f"values in {detail_columns}."
            )
        for column in detail_columns:
            out[column] = detail_numeric[column]
        detail_complete = bool(known.to_numpy().all())
    else:
        detail_complete = False
    out["infrastructure_detail_available"] = bool(detail_complete)
    return out


# ---------------------------------------------------------------------------
# Dimension availability / preparation
# ---------------------------------------------------------------------------
_DIMENSION_COLUMNS = {
    "environmental": config.ENV_REQUIRED_COLUMNS,
    "population": config.POP_REQUIRED_COLUMNS,
    "infrastructure": config.INFRA_REQUIRED_COLUMNS,
}


def dimension_availability(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    """Report availability per dimension without computing or filling values."""
    report: dict[str, dict[str, object]] = {}
    for dimension, columns in _DIMENSION_COLUMNS.items():
        if frame is None or not isinstance(frame, pd.DataFrame):
            report[dimension] = {
                "status": config.STATUS_UNAVAILABLE,
                "missing": tuple(columns),
                "message": "No exposure table was supplied.",
            }
            continue
        missing = tuple(column for column in columns if column not in frame.columns)
        if missing:
            report[dimension] = {
                "status": config.STATUS_UNAVAILABLE,
                "missing": missing,
                "message": (
                    f"{dimension.title()} exposure is unavailable: missing "
                    f"column(s) {list(missing)}."
                ),
            }
        else:
            report[dimension] = {
                "status": config.STATUS_AVAILABLE,
                "missing": (),
                "message": f"{dimension.title()} exposure columns are present.",
            }
    return report


def prepare_exposure(frame: pd.DataFrame | None) -> ExposureResult:
    """Validate and enrich an exposure table for severity scoring.

    Returns ``status=UNAVAILABLE`` with a reason when any dimension is missing
    or incomplete.  No partial table with zero-filled dimensions is returned.
    """
    dimensions = dimension_availability(frame)
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return ExposureResult(
            status=config.STATUS_UNAVAILABLE,
            frame=None,
            message="Exposure is unavailable: no exposure rows were supplied.",
            dimensions=dimensions,
        )

    unavailable = [
        dimension
        for dimension, info in dimensions.items()
        if info["status"] != config.STATUS_AVAILABLE
    ]
    if unavailable:
        messages = " ".join(
            str(dimensions[dimension]["message"]) for dimension in unavailable
        )
        return ExposureResult(
            status=config.STATUS_UNAVAILABLE,
            frame=None,
            message=messages,
            dimensions=dimensions,
        )

    try:
        out = environmental_exposure(frame)
        out = population_exposure(out)
        out = infrastructure_exposure(out)
    except ImpactDataError as exc:
        return ExposureResult(
            status=config.STATUS_UNAVAILABLE,
            frame=None,
            message=str(exc),
            dimensions=dimensions,
        )

    return ExposureResult(
        status=config.STATUS_AVAILABLE,
        frame=out,
        message="Environmental, population and infrastructure exposure are available.",
        dimensions=dimensions,
    )


# ---------------------------------------------------------------------------
# Risk / exposure join
# ---------------------------------------------------------------------------
def merge_risk_and_exposure(
    exposure: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join per-cell risk scores onto the exposure table.

    Both frames must describe the same cell grid.  Rows whose risk is outside
    the model area remain present and are flagged ``outside_model_area``; their
    risk values are *not* filled or reinterpreted.
    """
    _require_columns(exposure, ("lat_cell", "lon_cell"), "Exposure join")
    _require_columns(
        scores,
        ("lat_cell", "lon_cell", "probability", "risk", "in_model_area"),
        "Risk scores",
    )
    if scores.duplicated(subset=["lat_cell", "lon_cell"]).any():
        raise InconsistentDataError(
            "Risk scores contain duplicate cell coordinates; expected one row "
            "per cell and date."
        )

    merged = exposure.merge(
        scores[
            [
                "lat_cell",
                "lon_cell",
                "probability",
                "risk",
                "in_model_area",
            ]
        ],
        on=["lat_cell", "lon_cell"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(exposure):
        raise InconsistentDataError(
            "Risk/exposure join changed the number of exposure rows."
        )

    risk_text = merged["risk"].astype("string").str.upper()
    merged["outside_model_area"] = (
        (
            ~merged["in_model_area"].fillna(False).astype(bool)
            | risk_text.eq("OUTSIDE").fillna(False)
            | merged["probability"].isna()
        )
        .astype(bool)
    )
    return merged


__all__ = [
    "ImpactDataError",
    "MissingColumnsError",
    "IncompleteDataError",
    "InconsistentDataError",
    "ExposureResult",
    "cell_bounds",
    "cell_area_km2",
    "is_on_model_grid",
    "grid_alignment_issues",
    "environmental_exposure",
    "population_exposure",
    "infrastructure_exposure",
    "dimension_availability",
    "prepare_exposure",
    "merge_risk_and_exposure",
]
