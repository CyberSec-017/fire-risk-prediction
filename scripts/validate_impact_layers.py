"""Offline validation checks for the impact-assessment artifacts.

Run after ``scripts/build_impact_layers.py``:

    python scripts/validate_impact_layers.py --write-report

The checks are deliberately conservative and never repair data:

* cell-area sums match the spherical area of the model grid;
* land-cover class areas sum to the cell area;
* population counts are finite, non-negative, ring-monotonic and plausible;
* OSM building footprints are compared with the optional GHSL built surface.

Geospatial libraries are not imported here at all: the validator reads the
small derived parquet artifacts, so it runs in the runtime environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from impact import config, exposure  # noqa: E402

REPORT_FILENAME = "validation_report.json"


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValidationIssue:
    """One explicit validation finding."""

    code: str
    level: str  # "error", "warning" or "info"
    message: str
    cell_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level,
            "message": self.message,
            "cell_id": self.cell_id,
        }


def _issue(code: str, level: str, message: str, cell_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, level=level, message=message, cell_id=cell_id)


def _error(code: str, message: str, cell_id: str | None = None) -> ValidationIssue:
    return _issue(code, "error", message, cell_id)


def _warning(code: str, message: str, cell_id: str | None = None) -> ValidationIssue:
    return _issue(code, "warning", message, cell_id)


def _missing_columns(
    frame: pd.DataFrame, required: Iterable[str], dataset: str
) -> list[ValidationIssue]:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [
            _error(
                "missing_columns",
                f"{dataset} is missing required column(s): {missing}.",
            )
        ]
    return []


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------------------
# Cell-area checks
# ---------------------------------------------------------------------------
def validate_cell_areas(
    frame: pd.DataFrame,
    tolerance_km2: float = 1.0,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Check each stored cell area against the spherical formula and the sum."""
    required = ("lat_cell", "lon_cell", config.CELL_AREA_COLUMN)
    issues = _missing_columns(frame, required, "Cell exposure")
    if issues:
        return issues, {}

    numeric = _numeric(frame, required)
    bad = numeric.isna().any(axis=1) | np.isinf(numeric).any(axis=1)
    if bad.any():
        cell_ids = frame.loc[bad, "cell_id"].tolist() if "cell_id" in frame else []
        issues.append(
            _error(
                "invalid_cell_coordinates",
                f"{int(bad.sum())} cell(s) have missing/non-finite coordinates or area.",
                cell_ids[0] if cell_ids else None,
            )
        )
        return issues, {}

    expected = np.array(
        [
            exposure.cell_area_km2(lat, lon)
            for lat, lon in zip(numeric["lat_cell"], numeric["lon_cell"])
        ]
    )
    stored = numeric[config.CELL_AREA_COLUMN].to_numpy(dtype=float)
    differences = np.abs(stored - expected)
    offenders = np.where(differences > tolerance_km2)[0]
    for index in offenders[:20]:
        cell_id = frame["cell_id"].iloc[index] if "cell_id" in frame else None
        issues.append(
            _error(
                "cell_area_mismatch",
                (
                    f"Stored area {stored[index]:.3f} km² differs from the "
                    f"spherical area {expected[index]:.3f} km² by more than "
                    f"{tolerance_km2} km²."
                ),
                str(cell_id) if cell_id is not None else None,
            )
        )
    if len(offenders) > 20:
        issues.append(
            _error(
                "cell_area_mismatch_many",
                f"{len(offenders) - 20} further cell-area mismatches were suppressed.",
            )
        )

    summary = {
        "cells": int(len(frame)),
        "total_area_km2": float(stored.sum()),
        "expected_total_area_km2": float(expected.sum()),
        "max_abs_error_km2": float(differences.max()) if differences.size else 0.0,
        "tolerance_km2": float(tolerance_km2),
    }
    return issues, summary


# ---------------------------------------------------------------------------
# Land-cover checks
# ---------------------------------------------------------------------------
def validate_landcover_fractions(
    frame: pd.DataFrame,
    tolerance: float = 0.01,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Check that land-cover class areas sum to the cell area."""
    lc_columns = list(config.LC_AREA_COLUMNS.values())
    required = (config.CELL_AREA_COLUMN, *lc_columns)
    issues = _missing_columns(frame, required, "Cell exposure")
    if issues:
        return issues, {}

    numeric = _numeric(frame, required)
    if numeric[lc_columns].isna().all().all():
        return [
            _warning(
                "landcover_unavailable",
                "Land-cover data is unavailable (WorldCover was not ingested).",
            )
        ], {"cells": int(len(frame)), "status": "not ingested"}

    bad = numeric.isna().any(axis=1) | np.isinf(numeric).any(axis=1)
    if bad.any():
        cell_ids = frame.loc[bad, "cell_id"].tolist() if "cell_id" in frame else []
        issues.append(
            _error(
                "invalid_landcover_values",
                f"{int(bad.sum())} cell(s) have missing/non-finite land-cover areas.",
                cell_ids[0] if cell_ids else None,
            )
        )
        return issues, {}

    negative = (numeric[lc_columns] < 0).any(axis=1)
    if negative.any():
        cell_ids = frame.loc[negative, "cell_id"].tolist() if "cell_id" in frame else []
        issues.append(
            _error(
                "negative_landcover_area",
                f"{int(negative.sum())} cell(s) have negative land-cover area.",
                cell_ids[0] if cell_ids else None,
            )
        )

    area = numeric[config.CELL_AREA_COLUMN].to_numpy(dtype=float)
    if (area <= 0).any():
        issues.append(_error("nonpositive_cell_area", "Cell area must be strictly positive."))
        return issues, {}

    fractions = numeric[lc_columns].to_numpy(dtype=float) / area[:, None]
    totals = fractions.sum(axis=1)
    over = totals > 1.0 + tolerance
    under = totals < 1.0 - tolerance
    for index in np.where(over)[0][:20]:
        cell_id = frame["cell_id"].iloc[index] if "cell_id" in frame else None
        issues.append(
            _error(
                "landcover_fraction_above_one",
                f"Land-cover fractions sum to {totals[index]:.4f} (expected 1.0).",
                str(cell_id) if cell_id is not None else None,
            )
        )
    for index in np.where(under)[0][:20]:
        cell_id = frame["cell_id"].iloc[index] if "cell_id" in frame else None
        issues.append(
            _error(
                "landcover_fraction_below_one",
                f"Land-cover fractions sum to {totals[index]:.4f} (expected 1.0); "
                "coverage is incomplete and was not zero-filled.",
                str(cell_id) if cell_id is not None else None,
            )
        )

    class_over_one = (fractions > 1.0 + tolerance).any(axis=0)
    for column, is_over in zip(lc_columns, class_over_one):
        if is_over:
            issues.append(
                _error(
                    "landcover_class_above_one",
                    f"Class fraction for {column} exceeds 1.0.",
                )
            )

    summary = {
        "cells": int(len(frame)),
        "min_fraction_sum": float(totals.min()) if totals.size else None,
        "max_fraction_sum": float(totals.max()) if totals.size else None,
        "mean_fraction_sum": float(totals.mean()) if totals.size else None,
        "tolerance": float(tolerance),
    }
    return issues, summary


# ---------------------------------------------------------------------------
# Population checks
# ---------------------------------------------------------------------------
def validate_population(
    frame: pd.DataFrame,
    max_density_per_km2: float = 200_000.0,
    monotonic_tolerance: float = 1e-6,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Check population finiteness, signs, densities and ring monotonicity."""
    required = (config.CELL_AREA_COLUMN, *config.POP_REQUIRED_COLUMNS[1:])
    issues = _missing_columns(frame, required, "Cell exposure")
    if issues:
        return issues, {}

    numeric = _numeric(frame, required)
    bad = numeric.isna().any(axis=1) | np.isinf(numeric).any(axis=1)
    if bad.any():
        cell_ids = frame.loc[bad, "cell_id"].tolist() if "cell_id" in frame else []
        issues.append(
            _error(
                "invalid_population_values",
                f"{int(bad.sum())} cell(s) have missing/non-finite population.",
                cell_ids[0] if cell_ids else None,
            )
        )
        return issues, {}

    negative = (numeric < 0).any(axis=1)
    if negative.any():
        cell_ids = frame.loc[negative, "cell_id"].tolist() if "cell_id" in frame else []
        issues.append(
            _error(
                "negative_population",
                f"{int(negative.sum())} cell(s) have negative population.",
                cell_ids[0] if cell_ids else None,
            )
        )

    area = numeric[config.CELL_AREA_COLUMN].to_numpy(dtype=float)
    density = numeric["pop_cell"].to_numpy(dtype=float) / area
    implausible = density > max_density_per_km2
    for index in np.where(implausible)[0][:20]:
        cell_id = frame["cell_id"].iloc[index] if "cell_id" in frame else None
        issues.append(
            _warning(
                "implausible_population_density",
                f"Population density {density[index]:,.0f} people/km² exceeds "
                f"{max_density_per_km2:,.0f} people/km².",
                str(cell_id) if cell_id is not None else None,
            )
        )

    ring_5 = numeric["pop_5km"].to_numpy(dtype=float)
    ring_10 = numeric["pop_10km"].to_numpy(dtype=float)
    ring_25 = numeric["pop_25km"].to_numpy(dtype=float)
    non_monotonic = (
        (ring_10 + monotonic_tolerance < ring_5)
        | (ring_25 + monotonic_tolerance < ring_10)
    )
    for index in np.where(non_monotonic)[0][:20]:
        cell_id = frame["cell_id"].iloc[index] if "cell_id" in frame else None
        issues.append(
            _warning(
                "non_monotonic_rings",
                "Population rings are not non-decreasing: "
                f"5 km={ring_5[index]:,.0f}, 10 km={ring_10[index]:,.0f}, "
                f"25 km={ring_25[index]:,.0f}.",
                str(cell_id) if cell_id is not None else None,
            )
        )

    summary = {
        "cells": int(len(frame)),
        "total_population": float(numeric["pop_cell"].sum()),
        "max_density_per_km2": float(np.nanmax(density)) if density.size else None,
        "non_monotonic_cells": int(non_monotonic.sum()),
        "implausible_density_cells": int(implausible.sum()),
    }
    return issues, summary


# ---------------------------------------------------------------------------
# OSM vs GHSL diagnostic
# ---------------------------------------------------------------------------
def ghsl_osm_diagnostics(
    frame: pd.DataFrame,
    low_ratio: float = 0.20,
    high_ratio: float = 2.0,
) -> tuple[list[ValidationIssue], pd.DataFrame, dict[str, Any]]:
    """Compare OSM building footprints with the optional GHSL built surface.

    The ratio is diagnostic only: OSM is not corrected or scaled to GHSL.
    """
    if "built_surface_km2" not in frame.columns:
        return (
            [
                _warning(
                    "ghsl_unavailable",
                    "GHSL built-surface column is absent, so the OSM-vs-GHSL "
                    "completeness diagnostic cannot run.",
                )
            ],
            pd.DataFrame(),
            {"available": False},
        )
    required = ("built_surface_km2", "building_footprint_km2")
    issues = _missing_columns(frame, required, "Cell exposure")
    if issues:
        return issues, pd.DataFrame(), {"available": False}

    numeric = _numeric(frame, required)
    bad = numeric.isna().any(axis=1) | np.isinf(numeric).any(axis=1)
    if bad.any():
        return (
            [
                _error(
                    "invalid_ghsl_values",
                    f"{int(bad.sum())} cell(s) have missing/non-finite GHSL or "
                    "OSM footprint values.",
                )
            ],
            pd.DataFrame(),
            {"available": True},
        )

    osm = numeric["building_footprint_km2"].to_numpy(dtype=float)
    ghsl = numeric["built_surface_km2"].to_numpy(dtype=float)
    ratio = np.full(len(frame), np.nan, dtype=float)
    positive = ghsl > 0
    ratio[positive] = osm[positive] / ghsl[positive]

    diagnostic = pd.DataFrame(
        {
            "cell_id": frame["cell_id"].to_numpy() if "cell_id" in frame else np.arange(len(frame)),
            "osm_footprint_km2": osm,
            "ghsl_built_surface_km2": ghsl,
            "osm_to_ghsl_ratio": ratio,
        }
    )
    low = positive & (ratio < low_ratio)
    high = positive & (ratio > high_ratio)
    ghsl_zero_osm_positive = (~positive) & (osm > 0)
    diagnostic["diagnostic"] = "ok"
    diagnostic.loc[low, "diagnostic"] = "osm_low_coverage"
    diagnostic.loc[high, "diagnostic"] = "osm_above_ghsl"
    diagnostic.loc[ghsl_zero_osm_positive, "diagnostic"] = "ghsl_zero_osm_positive"

    for index in np.where(low)[0][:20]:
        issues.append(
            _warning(
                "osm_low_coverage",
                f"OSM footprint is {ratio[index]:.1%} of GHSL built surface "
                f"(< {low_ratio:.0%}); OSM may be incomplete here.",
                str(diagnostic.loc[index, "cell_id"]),
            )
        )
    for index in np.where(high)[0][:20]:
        issues.append(
            _warning(
                "osm_above_ghsl",
                f"OSM footprint is {ratio[index]:.1f}× GHSL built surface "
                f"(> {high_ratio:.1f}×); check building geometries.",
                str(diagnostic.loc[index, "cell_id"]),
            )
        )
    for index in np.where(ghsl_zero_osm_positive)[0][:20]:
        issues.append(
            _warning(
                "ghsl_zero_osm_positive",
                "GHSL reports zero built surface but OSM has mapped buildings.",
                str(diagnostic.loc[index, "cell_id"]),
            )
        )

    summary = {
        "available": True,
        "cells_with_ghsl": int(positive.sum()),
        "low_coverage_cells": int(low.sum()),
        "above_ghsl_cells": int(high.sum()),
        "median_ratio": float(np.nanmedian(ratio)) if positive.any() else None,
        "low_ratio_threshold": float(low_ratio),
        "high_ratio_threshold": float(high_ratio),
    }
    return issues, diagnostic, summary


# ---------------------------------------------------------------------------
# Cross-file checks
# ---------------------------------------------------------------------------
def validate_cross_file(
    cells: pd.DataFrame,
    rings: pd.DataFrame | None = None,
    settlements: pd.DataFrame | None = None,
    infrastructure: pd.DataFrame | None = None,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Check coordinates, rings and non-negative derived values."""
    issues: list[ValidationIssue] = []
    coordinate_problems = _missing_columns(
        cells, ("lat_cell", "lon_cell"), "Cell exposure"
    )
    if coordinate_problems:
        return coordinate_problems, {}
    cell_keys = set(
        zip(
            pd.to_numeric(cells.get("lat_cell"), errors="coerce"),
            pd.to_numeric(cells.get("lon_cell"), errors="coerce"),
        )
    )

    def check_points(frame: pd.DataFrame, dataset: str) -> None:
        if frame is None or frame.empty:
            return
        problems = _missing_columns(frame, ("lat_cell", "lon_cell"), dataset)
        issues.extend(problems)
        if problems:
            return
        lat = pd.to_numeric(frame["lat_cell"], errors="coerce")
        lon = pd.to_numeric(frame["lon_cell"], errors="coerce")
        outside = (
            ~lat.between(config.MODEL_LAT_MIN, config.MODEL_LAT_MAX)
            | ~lon.between(config.MODEL_LON_MIN, config.MODEL_LON_MAX)
            | lat.isna()
            | lon.isna()
        )
        if outside.any():
            issues.append(
                _error(
                    "points_outside_study_area",
                    f"{int(outside.sum())} row(s) in {dataset} fall outside the "
                    "model study area.",
                )
            )
        unknown = [
            (float(a), float(b))
            for a, b in zip(lat.dropna(), lon.dropna())
            if (float(a), float(b)) not in cell_keys
        ]
        if unknown:
            issues.append(
                _error(
                    "points_not_on_cells",
                    f"{len(unknown)} row(s) in {dataset} are not on a known cell "
                    "centre.",
                )
            )

    check_points(settlements, "Settlements")
    check_points(infrastructure, "Infrastructure")

    ring_summary: dict[str, Any] = {}
    if rings is not None and not rings.empty:
        problems = _missing_columns(rings, config.RING_REQUIRED_COLUMNS, "Ring exposure")
        issues.extend(problems)
        if not problems:
            known_rings = {float(value) for value in config.RING_DISTANCES_KM}
            invalid_rings = ~pd.to_numeric(rings["ring_km"], errors="coerce").isin(
                known_rings
            )
            if invalid_rings.any():
                issues.append(
                    _error(
                        "invalid_ring_distance",
                        f"{int(invalid_rings.sum())} ring row(s) have an unexpected "
                        f"ring_km value; expected {sorted(known_rings)}.",
                    )
                )
            numeric = _numeric(
                rings,
                (
                    "population",
                    "road_km",
                    "building_count",
                    "building_footprint_km2",
                    "critical_count",
                ),
            )
            negative = (numeric < 0).any(axis=1)
            if negative.any():
                issues.append(
                    _error(
                        "negative_ring_values",
                        f"{int(negative.sum())} ring row(s) contain negative values.",
                    )
                )
            check_points(rings, "Ring exposure")
            ring_summary = {
                "rows": int(len(rings)),
                "cells": int(rings[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]),
                "distances_km": sorted(known_rings),
                "note": "Distance rings overlap and must not be summed across cells.",
            }

    if settlements is not None and "distance_km" in settlements.columns:
        distance = pd.to_numeric(settlements["distance_km"], errors="coerce")
        if (distance < 0).any() or distance.isna().any():
            issues.append(
                _error(
                    "invalid_settlement_distance",
                    "Settlement distance_km contains negative or missing values.",
                )
            )

    return issues, ring_summary


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _read_parquet(path: Path, dataset: str) -> tuple[pd.DataFrame | None, list[ValidationIssue]]:
    if not path.exists():
        return None, [
            _error(
                "artifact_missing",
                f"{dataset} is missing at {path}. Run scripts/build_impact_layers.py first.",
            )
        ]
    try:
        return pd.read_parquet(path), []
    except Exception as exc:  # noqa: BLE001
        return None, [_error("artifact_unreadable", f"{dataset} at {path} cannot be read: {exc}")]


def _clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_clean_json(record) for record in frame.to_dict(orient="records")]


def run_validation(impact_dir: Path, write_report: bool = False) -> dict[str, Any]:
    """Run every check against the derived artifacts and optionally write a report."""
    impact_dir = Path(impact_dir)
    issues: list[ValidationIssue] = []
    summaries: dict[str, Any] = {}
    diagnostics_records: list[dict[str, Any]] = []

    cells, cell_issues = _read_parquet(
        impact_dir / config.CELL_EXPOSURE_PATH.name, "Cell exposure"
    )
    issues.extend(cell_issues)
    if cells is not None:
        for function, key in (
            (validate_cell_areas, "cell_areas"),
            (validate_landcover_fractions, "landcover"),
            (validate_population, "population"),
        ):
            found, summary = function(cells)
            issues.extend(found)
            summaries[key] = summary

        ghsl_issues, diagnostic, ghsl_summary = ghsl_osm_diagnostics(cells)
        issues.extend(ghsl_issues)
        summaries["osm_vs_ghsl"] = ghsl_summary
        diagnostics_records = _records(diagnostic)

    rings, ring_issues = _read_parquet(
        impact_dir / config.RING_EXPOSURE_PATH.name, "Ring exposure"
    )
    issues.extend(ring_issues)
    settlements, settlement_issues = _read_parquet(
        impact_dir / config.SETTLEMENTS_PATH.name, "Settlements"
    )
    issues.extend(settlement_issues)
    infrastructure, infrastructure_issues = _read_parquet(
        impact_dir / config.INFRASTRUCTURE_PATH.name, "Infrastructure"
    )
    issues.extend(infrastructure_issues)
    admin_path = impact_dir / config.ADMIN_UNITS_PATH.name
    if admin_path.exists():
        admin_units, admin_issues = _read_parquet(
            admin_path, "Administrative units"
        )
        issues.extend(admin_issues)
    else:
        # Optional OCHA input: absence is informational, not a build error.
        admin_units = None
        issues.append(
            _warning(
                "admin_units_unavailable",
                "Administrative-unit assignment is unavailable (optional OCHA input not built).",
            )
        )

    if cells is not None:
        cross_issues, ring_summary = validate_cross_file(
            cells, rings, settlements, infrastructure
        )
        issues.extend(cross_issues)
        summaries["cross_file"] = ring_summary

    counts = {
        "error": sum(1 for issue in issues if issue.level == "error"),
        "warning": sum(1 for issue in issues if issue.level == "warning"),
        "info": sum(1 for issue in issues if issue.level == "info"),
    }
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "impact_dir": str(impact_dir),
        "ok": counts["error"] == 0,
        "counts": counts,
        "issues": [issue.as_dict() for issue in issues],
        "summaries": summaries,
        "diagnostics": diagnostics_records,
    }
    report = _clean_json(report)
    if write_report:
        impact_dir.mkdir(parents=True, exist_ok=True)
        (impact_dir / REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--impact-dir",
        type=Path,
        default=config.IMPACT_DATA_DIR,
        help="Directory containing derived impact artifacts.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"Write {REPORT_FILENAME} into the impact directory.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return a non-zero exit code when warnings are present.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_validation(args.impact_dir, write_report=args.write_report)
    print(
        "Impact validation: "
        f"{report['counts']['error']} error(s), "
        f"{report['counts']['warning']} warning(s), "
        f"{report['counts']['info']} info."
    )
    for issue in report["issues"]:
        location = f" [{issue['cell_id']}]" if issue.get("cell_id") else ""
        print(f"  {issue['level'].upper()}: {issue['code']}{location} — {issue['message']}")
    if args.write_report:
        print(f"Report written to {Path(args.impact_dir) / REPORT_FILENAME}")
    if report["counts"]["error"]:
        return 1
    if args.fail_on_warning and report["counts"]["warning"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
