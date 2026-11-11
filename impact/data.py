"""Read-only loaders for derived impact artifacts.

Every loader returns a :class:`DataResult` instead of a bare ``None`` or an
empty frame, so the UI can show an explicit *unavailable* state.  Missing
data is never replaced with zeros.

No Streamlit, model or pipeline imports are allowed here; see
``tests/test_impact_integrity.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from impact import config


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataResult:
    """Outcome of loading one derived dataset."""

    status: str
    frame: pd.DataFrame | None = None
    message: str = ""
    path: Path | None = None
    manifest: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return self.status == config.STATUS_AVAILABLE

    @property
    def unavailable(self) -> bool:
        return not self.available


@dataclass(frozen=True)
class DateRangeResult:
    """Outcome of reading the risk window from the fire panel."""

    status: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    message: str = ""
    path: Path | None = None
    dates: tuple[pd.Timestamp, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.status == config.STATUS_AVAILABLE


def _resolve(path: Path | str | None, default: Path) -> Path:
    return Path(path) if path is not None else default


def _missing(dataset: str, path: Path) -> DataResult:
    return DataResult(
        status=config.STATUS_MISSING,
        frame=None,
        message=(
            f"{dataset} is unavailable: {path} does not exist. "
            "Run scripts/build_impact_layers.py to generate the impact artifacts."
        ),
        path=path,
    )


def _invalid(dataset: str, path: Path, detail: str) -> DataResult:
    return DataResult(
        status=config.STATUS_INVALID,
        frame=None,
        message=f"{dataset} at {path} is invalid: {detail}",
        path=path,
    )


def _check_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    dataset: str,
    path: Path,
) -> DataResult | None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return _invalid(dataset, path, f"missing required column(s): {missing}.")
    return None


def _load_frame(
    dataset: str,
    path: Path,
    required: Iterable[str],
) -> DataResult:
    if not path.exists():
        return _missing(dataset, path)
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        return _invalid(dataset, path, f"could not be read as parquet ({exc}).")
    if frame is None or frame.empty:
        return _invalid(dataset, path, "the file contains no rows.")
    problem = _check_columns(frame, required, dataset, path)
    if problem is not None:
        return problem
    return DataResult(
        status=config.STATUS_AVAILABLE,
        frame=frame,
        message=f"{dataset} loaded from {path}.",
        path=path,
    )


# ---------------------------------------------------------------------------
# Derived exposure artifacts
# ---------------------------------------------------------------------------
def load_cell_exposure(path: Path | str | None = None) -> DataResult:
    """Load the per-cell exposure table (WorldCover/WorldPop/OSM joins)."""
    return _load_frame(
        "Cell exposure table",
        _resolve(path, config.CELL_EXPOSURE_PATH),
        required=tuple(
            dict.fromkeys(
                (config.CELL_ID_COLUMN, "lat_cell", "lon_cell")
                + config.ENV_REQUIRED_COLUMNS
                + config.POP_REQUIRED_COLUMNS
                + config.INFRA_REQUIRED_COLUMNS
            )
        ),
    )


def load_ring_exposure(path: Path | str | None = None) -> DataResult:
    """Load the non-additive per-ring (5/10/25 km) exposure table."""
    return _load_frame(
        "Ring exposure table",
        _resolve(path, config.RING_EXPOSURE_PATH),
        required=config.RING_REQUIRED_COLUMNS,
    )


def load_settlements(path: Path | str | None = None) -> DataResult:
    """Load settlement points and their population estimates."""
    return _load_frame(
        "Settlement table",
        _resolve(path, config.SETTLEMENTS_PATH),
        required=config.SETTLEMENTS_REQUIRED_COLUMNS,
    )


def load_infrastructure(path: Path | str | None = None) -> DataResult:
    """Load the mapped critical-infrastructure asset table."""
    return _load_frame(
        "Infrastructure asset table",
        _resolve(path, config.INFRASTRUCTURE_PATH),
        required=config.INFRASTRUCTURE_REQUIRED_COLUMNS,
    )


def load_admin_units(path: Path | str | None = None) -> DataResult:
    """Load the cell-to-admin-unit assignment used for regional comparisons."""
    return _load_frame(
        "Administrative-unit table",
        _resolve(path, config.ADMIN_UNITS_PATH),
        required=config.ADMIN_UNITS_REQUIRED_COLUMNS,
    )


def load_burned_area(path: Path | str | None = None) -> DataResult:
    """Load confirmed burned area.

    Burned area is a phase-2 dataset.  Absence is reported as unavailable /
    missing; it is never inferred from detections.
    """
    resolved = _resolve(path, config.BURNED_AREA_PATH)
    if path is None and not resolved.exists():
        return DataResult(
            status=config.STATUS_UNAVAILABLE,
            frame=None,
            message=(
                "Confirmed burned area is unavailable: no independent "
                "burned-area product has been ingested (phase 2). It is never "
                "derived from FIRMS detections or model probabilities."
            ),
            path=resolved,
        )
    return _load_frame(
        "Confirmed burned-area table",
        resolved,
        required=config.BURNED_AREA_REQUIRED_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------
def load_manifest(path: Path | str | None = None) -> DataResult:
    """Load and validate ``data/impact/manifest.json``."""
    resolved = _resolve(path, config.MANIFEST_PATH)
    if not resolved.exists():
        return DataResult(
            status=config.STATUS_MISSING,
            frame=None,
            manifest=None,
            message=(
                f"Impact data manifest is unavailable: {resolved} does not "
                "exist, so source provenance cannot be verified."
            ),
            path=resolved,
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return DataResult(
            status=config.STATUS_INVALID,
            frame=None,
            manifest=None,
            message=f"Impact data manifest at {resolved} is invalid: {exc}",
            path=resolved,
        )

    if not isinstance(payload, dict):
        return DataResult(
            status=config.STATUS_INVALID,
            frame=None,
            manifest=None,
            message="Impact data manifest must be a JSON object.",
            path=resolved,
        )
    problems = _manifest_problems(payload)
    if problems:
        return DataResult(
            status=config.STATUS_INVALID,
            frame=None,
            manifest=payload,
            message="Impact data manifest is incomplete: " + "; ".join(problems),
            path=resolved,
        )
    return DataResult(
        status=config.STATUS_AVAILABLE,
        frame=None,
        manifest=payload,
        message=f"Impact data manifest loaded from {resolved}.",
        path=resolved,
    )


def _manifest_problems(payload: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not payload.get("schema_version"):
        problems.append("missing schema_version")
    if not payload.get("generated_at"):
        problems.append("missing generated_at")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        problems.append("missing non-empty sources list")
    else:
        for index, source in enumerate(sources):
            if not isinstance(source, Mapping):
                problems.append(f"sources[{index}] is not an object")
                continue
            for key in ("name", "version", "license"):
                if not source.get(key):
                    problems.append(f"sources[{index}] missing {key}")
    return problems


def severity_thresholds_from_manifest(
    manifest: Mapping[str, Any] | None,
) -> tuple[float, ...] | None:
    """Extract frozen IEI tier thresholds from a manifest, if present."""
    if not manifest:
        return None
    severity = manifest.get("severity")
    if not isinstance(severity, Mapping):
        return None
    thresholds = severity.get("tier_thresholds")
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 4:
        return None
    try:
        values = tuple(float(value) for value in thresholds)
    except (TypeError, ValueError):
        return None
    if any(not (0.0 <= value <= 1.0) for value in values):
        return None
    if list(values) != sorted(values):
        return None
    return values


# ---------------------------------------------------------------------------
# Fire-panel date range (the data's dates, never dashboard config years)
# ---------------------------------------------------------------------------
def date_range_from_panel(
    frame: pd.DataFrame | None,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the min/max observation dates present in a panel frame."""
    if frame is None or frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min(), dates.max()


def load_panel_date_range(path: Path | str | None = None) -> DateRangeResult:
    """Read the risk window from ``fire_panel.parquet``.

    Only the ``date`` column is read.  The result reflects the data itself,
    not the dashboard configuration's nominal year constants.
    """
    resolved = _resolve(path, config.FIRE_PANEL_PATH)
    if not resolved.exists():
        return DateRangeResult(
            status=config.STATUS_MISSING,
            message=f"Fire panel is unavailable: {resolved} does not exist.",
            path=resolved,
        )
    try:
        frame = pd.read_parquet(resolved, columns=["date"])
    except Exception as exc:  # noqa: BLE001
        return DateRangeResult(
            status=config.STATUS_INVALID,
            message=f"Fire panel dates could not be read from {resolved}: {exc}",
            path=resolved,
        )
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return DateRangeResult(
            status=config.STATUS_INVALID,
            message=f"Fire panel at {resolved} contains no parseable dates.",
            path=resolved,
        )
    unique_dates = tuple(sorted(pd.Timestamp(value) for value in dates.unique()))
    return DateRangeResult(
        status=config.STATUS_AVAILABLE,
        start=unique_dates[0],
        end=unique_dates[-1],
        dates=unique_dates,
        message=(
            f"Risk window read from {resolved.name}: "
            f"{unique_dates[0].date()} to {unique_dates[-1].date()} "
            f"({len(unique_dates):,} dates)."
        ),
        path=resolved,
    )


# ---------------------------------------------------------------------------
# Availability summary for the UI
# ---------------------------------------------------------------------------
def availability_report() -> list[dict[str, Any]]:
    """Return an explicit status row per impact dataset.

    Rows never carry a numeric fallback; consumers must check ``status``.
    """
    loaders = [
        ("cell_exposure", "Cell exposure", load_cell_exposure),
        ("ring_exposure", "Distance-ring exposure", load_ring_exposure),
        ("settlements", "Settlements", load_settlements),
        ("infrastructure", "Critical infrastructure", load_infrastructure),
        ("admin_units", "Administrative units", load_admin_units),
        ("burned_area", "Confirmed burned area", load_burned_area),
    ]
    rows = []
    for key, label, loader in loaders:
        result = loader()
        rows.append(
            {
                "key": key,
                "label": label,
                "status": result.status,
                "available": result.available,
                "message": result.message,
                "path": str(result.path) if result.path is not None else "",
            }
        )
    manifest = load_manifest()
    rows.append(
        {
            "key": "manifest",
            "label": "Data manifest",
            "status": manifest.status,
            "available": manifest.available,
            "message": manifest.message,
            "path": str(manifest.path) if manifest.path is not None else "",
        }
    )
    return rows


__all__ = [
    "DataResult",
    "DateRangeResult",
    "load_cell_exposure",
    "load_ring_exposure",
    "load_settlements",
    "load_infrastructure",
    "load_admin_units",
    "load_burned_area",
    "load_manifest",
    "severity_thresholds_from_manifest",
    "date_range_from_panel",
    "load_panel_date_range",
    "availability_report",
]
