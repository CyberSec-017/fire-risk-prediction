"""Impact Exposure Index (IEI), severity tiers and priority matrix.

The index is a transparent prioritisation heuristic:

    IEI = 0.35 * fuel_norm + 0.35 * pop_norm + 0.30 * infra_norm

with

    infra_norm = 0.45 * road_norm + 0.35 * building_norm
               + 0.20 * critical_norm

Each component is normalized to [0, 1].  Fuel uses cell area as its
denominator; population and infrastructure use their study-area 95th
percentile.  Missing or degenerate inputs raise an explicit error instead of
being treated as zero.

The resulting tier is *not* a damage probability.  The risk-band × tier
matrix answers "where should limited attention go first?", not "what will
burn?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from impact import config
from impact.data import severity_thresholds_from_manifest
from impact.exposure import ImpactDataError

# ---------------------------------------------------------------------------
# Index definition
# ---------------------------------------------------------------------------
IEI_WEIGHTS = {"fuel": 0.35, "population": 0.35, "infrastructure": 0.30}
INFRA_SUBWEIGHTS = {"road": 0.45, "building": 0.35, "critical": 0.20}

# Renormalized weights when fuel component is absent (0.35/0.65 and 0.30/0.65)
IEI_RENORMALIZED_DIVISOR = IEI_WEIGHTS["population"] + IEI_WEIGHTS["infrastructure"]
IEI_WEIGHTS_NO_FUEL = {
    "population": IEI_WEIGHTS["population"] / IEI_RENORMALIZED_DIVISOR,
    "infrastructure": IEI_WEIGHTS["infrastructure"] / IEI_RENORMALIZED_DIVISOR,
}

TIER_LABELS = ("Very Low", "Low", "Moderate", "High", "Very High")

# Provisional defaults.  The build stage freezes the final values in
# data/impact/manifest.json under severity.tier_thresholds.
DEFAULT_TIER_THRESHOLDS = (0.10, 0.20, 0.35, 0.55)

RISK_BANDS = ("LOW", "MODERATE", "HIGH", "VERY HIGH")

# Rows: model risk band.  Columns: exposure tier.
# 1 = minimal priority, 5 = critical priority.
PRIORITY_MATRIX: dict[str, dict[str, int]] = {
    "LOW": {
        "Very Low": 1,
        "Low": 1,
        "Moderate": 2,
        "High": 2,
        "Very High": 3,
    },
    "MODERATE": {
        "Very Low": 1,
        "Low": 2,
        "Moderate": 2,
        "High": 3,
        "Very High": 4,
    },
    "HIGH": {
        "Very Low": 2,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very High": 5,
    },
    "VERY HIGH": {
        "Very Low": 2,
        "Low": 3,
        "Moderate": 4,
        "High": 5,
        "Very High": 5,
    },
}

PRIORITY_LABELS = {
    1: "Minimal priority",
    2: "Low priority",
    3: "Moderate priority",
    4: "High priority",
    5: "Critical priority",
}

IEI_REQUIRED_COLUMNS = (
    config.CELL_AREA_COLUMN,
    "fuel_area_km2",
    "pop_cell",
    "road_km_total",
    "building_footprint_km2",
    "critical_count_total",
)

IEI_NON_FUEL_COLUMNS = (
    config.CELL_AREA_COLUMN,
    "pop_cell",
    "road_km_total",
    "building_footprint_km2",
    "critical_count_total",
)

IEI_COMPONENT_COLUMNS = ("fuel_norm", "pop_norm", "infra_norm")

NOT_APPLICABLE = "Not applicable"
TIER_UNAVAILABLE = "Unavailable"


class SeverityInputError(ImpactDataError):
    """Raised when IEI inputs are missing or cannot be normalized."""


@dataclass(frozen=True)
class SeverityResult:
    """Frame enriched with IEI/tier/priority plus an explicit status."""

    status: str
    frame: pd.DataFrame | None
    message: str
    thresholds: tuple[float, ...]
    missing_components: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status == config.STATUS_AVAILABLE

    @property
    def unavailable(self) -> bool:
        return not self.available


# ---------------------------------------------------------------------------
# Thresholds and tiers
# ---------------------------------------------------------------------------
def _validate_weights(weights: Mapping[str, float], expected: Sequence[str]) -> None:
    if set(weights) != set(expected):
        raise SeverityInputError(
            f"Severity weights must define exactly {list(expected)}; got "
            f"{list(weights)}."
        )
    values = tuple(float(value) for value in weights.values())
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise SeverityInputError("Severity weights must be finite and non-negative.")
    if abs(sum(values) - 1.0) > 1e-9:
        raise SeverityInputError(
            f"Severity weights must sum to 1.0; got {sum(values):.6f}."
        )


def resolve_thresholds(
    thresholds: Sequence[float] | None = None,
    manifest: Mapping[str, object] | None = None,
) -> tuple[float, ...]:
    """Resolve IEI tier thresholds with explicit precedence.

    Order: caller-supplied thresholds → manifest-frozen thresholds → defaults.
    Exactly four strictly increasing values in (0, 1) are required.
    """
    if thresholds is None:
        thresholds = severity_thresholds_from_manifest(manifest)
    if thresholds is None:
        thresholds = DEFAULT_TIER_THRESHOLDS
    values = tuple(float(value) for value in thresholds)
    if len(values) != len(TIER_LABELS) - 1:
        raise SeverityInputError(
            f"Expected {len(TIER_LABELS) - 1} tier thresholds, got {len(values)}."
        )
    if any(not np.isfinite(value) or not (0.0 < value < 1.0) for value in values):
        raise SeverityInputError("Tier thresholds must lie strictly between 0 and 1.")
    if list(values) != sorted(values) or len(set(values)) != len(values):
        raise SeverityInputError("Tier thresholds must be strictly increasing.")
    return values


def classify_iei(
    values: pd.Series | Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> pd.Series:
    """Map IEI values to tier labels; NaN becomes ``Unavailable``."""
    resolved = resolve_thresholds(thresholds)
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    bins = [-np.inf, *resolved, np.inf]
    labels = pd.cut(series, bins=bins, labels=TIER_LABELS, right=False)
    result = pd.Series(labels.astype("object"), index=series.index)
    result = result.where(series.notna(), TIER_UNAVAILABLE)
    return result.astype(str)


def tier_for_iei(
    value: float,
    thresholds: Sequence[float] | None = None,
) -> str:
    """Scalar tier lookup (NaN → ``Unavailable``)."""
    return str(classify_iei([value], thresholds=thresholds).iloc[0])


# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------
def _numeric_required(
    frame: pd.DataFrame,
    required: Sequence[str] = IEI_REQUIRED_COLUMNS,
) -> pd.DataFrame:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SeverityInputError(
            "Impact severity is unavailable: missing required column(s) "
            f"{missing}. Run impact.exposure.prepare_exposure first; missing "
            "data is never treated as zero."
        )
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna() | np.isinf(numeric)
    if bad.to_numpy().any():
        offending = [column for column in required if bad[column].any()]
        raise SeverityInputError(
            "Impact severity is unavailable: missing, non-numeric or "
            f"non-finite values in {offending}."
        )
    negative = [column for column in required if (numeric[column] < 0).any()]
    if negative:
        raise SeverityInputError(
            f"Impact severity is unavailable: negative values in {negative}."
        )
    if (numeric[config.CELL_AREA_COLUMN] <= 0).any():
        raise SeverityInputError(
            f"Impact severity is unavailable: {config.CELL_AREA_COLUMN} must be "
            "strictly positive."
        )
    return numeric


def _p95(series: pd.Series, component: str) -> float:
    p95 = float(pd.to_numeric(series, errors="coerce").quantile(0.95))
    if not np.isfinite(p95) or p95 <= 0:
        raise SeverityInputError(
            f"Impact severity is unavailable: the 95th percentile for "
            f"{component} is not positive, so it cannot be normalized without "
            "inventing a denominator."
        )
    return p95


def compute_index(
    frame: pd.DataFrame,
    thresholds: Sequence[float] | None = None,
    manifest: Mapping[str, object] | None = None,
    weights: Mapping[str, float] | None = None,
    infra_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Add normalized components, IEI and exposure tier to a prepared frame.

    When the fuel component (``fuel_area_km2``) is entirely absent across all
    cells (or all values are NaN), IEI is computed from the remaining population
    and infrastructure components with their weights renormalized (0.35/0.65 and
    0.30/0.65). If some cells have fuel and others do not, an error is raised
    because methods cannot be mixed and missing fuel is never treated as zero.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SeverityInputError(
            "Impact severity is unavailable: no exposure rows were supplied."
        )

    infra_weights = dict(infra_weights or INFRA_SUBWEIGHTS)
    _validate_weights(infra_weights, ("road", "building", "critical"))
    resolved = resolve_thresholds(thresholds, manifest)

    # Check fuel availability
    fuel_in_cols = "fuel_area_km2" in frame.columns
    fuel_series = (
        pd.to_numeric(frame["fuel_area_km2"], errors="coerce")
        if fuel_in_cols
        else None
    )

    if not fuel_in_cols or fuel_series is None or fuel_series.isna().all():
        fuel_missing = True
    elif fuel_series.isna().any():
        offending = [
            str(frame.loc[idx, config.CELL_ID_COLUMN])
            if config.CELL_ID_COLUMN in frame.columns
            else str(idx)
            for idx in fuel_series.index[fuel_series.isna()]
        ]
        raise SeverityInputError(
            "Impact severity is unavailable: fuel_area_km2 is missing in some cells "
            f"({offending[:5]}) but present in others. Missing data is never treated "
            "as zero and scoring methods cannot be mixed across cells."
        )
    else:
        fuel_missing = False

    if fuel_missing:
        if weights is not None:
            used_weights = dict(weights)
            _validate_weights(used_weights, ("population", "infrastructure"))
        else:
            used_weights = dict(IEI_WEIGHTS_NO_FUEL)

        numeric = _numeric_required(frame, required=IEI_NON_FUEL_COLUMNS)
        pop_p95 = _p95(numeric["pop_cell"], "population")
        road_p95 = _p95(numeric["road_km_total"], "road length")
        building_p95 = _p95(numeric["building_footprint_km2"], "building footprint")
        critical_p95 = _p95(numeric["critical_count_total"], "critical facilities")

        out = frame.copy()
        for column in IEI_NON_FUEL_COLUMNS:
            out[column] = numeric[column]

        out["fuel_norm"] = np.nan
        out["pop_norm"] = (
            np.log1p(numeric["pop_cell"]) / np.log1p(pop_p95)
        ).clip(0.0, 1.0)
        out["road_norm"] = (numeric["road_km_total"] / road_p95).clip(0.0, 1.0)
        out["building_norm"] = (
            numeric["building_footprint_km2"] / building_p95
        ).clip(0.0, 1.0)
        out["critical_norm"] = (
            numeric["critical_count_total"] / critical_p95
        ).clip(0.0, 1.0)

        out["infra_norm"] = (
            infra_weights["road"] * out["road_norm"]
            + infra_weights["building"] * out["building_norm"]
            + infra_weights["critical"] * out["critical_norm"]
        )
        out["iei"] = (
            used_weights["population"] * out["pop_norm"]
            + used_weights["infrastructure"] * out["infra_norm"]
        ).clip(0.0, 1.0)
        out["iei_components_used"] = "pop+infra"
        out["iei_fuel_missing"] = True
        if "fuel_area_km2" not in out.columns:
            out["fuel_area_km2"] = np.nan
    else:
        used_weights = dict(weights or IEI_WEIGHTS)
        _validate_weights(used_weights, ("fuel", "population", "infrastructure"))

        numeric = _numeric_required(frame, required=IEI_REQUIRED_COLUMNS)
        area = numeric[config.CELL_AREA_COLUMN]

        if (numeric["fuel_area_km2"] > area * 1.05).any():
            raise SeverityInputError(
                "Impact severity is unavailable: flammable area exceeds cell area "
                "by more than 5 %."
            )

        pop_p95 = _p95(numeric["pop_cell"], "population")
        road_p95 = _p95(numeric["road_km_total"], "road length")
        building_p95 = _p95(numeric["building_footprint_km2"], "building footprint")
        critical_p95 = _p95(numeric["critical_count_total"], "critical facilities")

        out = frame.copy()
        for column in IEI_REQUIRED_COLUMNS:
            out[column] = numeric[column]

        out["fuel_norm"] = (numeric["fuel_area_km2"] / area).clip(0.0, 1.0)
        out["pop_norm"] = (
            np.log1p(numeric["pop_cell"]) / np.log1p(pop_p95)
        ).clip(0.0, 1.0)
        out["road_norm"] = (numeric["road_km_total"] / road_p95).clip(0.0, 1.0)
        out["building_norm"] = (
            numeric["building_footprint_km2"] / building_p95
        ).clip(0.0, 1.0)
        out["critical_norm"] = (
            numeric["critical_count_total"] / critical_p95
        ).clip(0.0, 1.0)

        out["infra_norm"] = (
            infra_weights["road"] * out["road_norm"]
            + infra_weights["building"] * out["building_norm"]
            + infra_weights["critical"] * out["critical_norm"]
        )
        out["iei"] = (
            used_weights["fuel"] * out["fuel_norm"]
            + used_weights["population"] * out["pop_norm"]
            + used_weights["infrastructure"] * out["infra_norm"]
        ).clip(0.0, 1.0)
        out["iei_components_used"] = "fuel+pop+infra"
        out["iei_fuel_missing"] = False

    out["exposure_tier"] = classify_iei(out["iei"], resolved)
    return out


# ---------------------------------------------------------------------------
# Priority matrix
# ---------------------------------------------------------------------------
def normalize_risk_band(value: object) -> str | None:
    """Normalize a model risk label to one of :data:`RISK_BANDS`."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip().upper().replace("_", " ")
    if text in PRIORITY_MATRIX:
        return text
    return None


def priority_for(risk_band: str | None, exposure_tier: str | None) -> int | None:
    """Return the 1–5 priority for one risk/tier pair, or ``None``."""
    band = normalize_risk_band(risk_band)
    if band is None or exposure_tier not in TIER_LABELS:
        return None
    return PRIORITY_MATRIX[band][str(exposure_tier)]


def assign_priority(
    frame: pd.DataFrame,
    risk_column: str = "risk",
) -> pd.DataFrame:
    """Add ``impact_priority`` (1–5) and ``impact_priority_label``.

    Rows outside the model area, or with an unknown risk band, receive a null
    priority and ``"Not applicable"`` — never a fabricated value.
    """
    if "exposure_tier" not in frame.columns:
        raise SeverityInputError(
            "Impact severity is unavailable: exposure_tier is missing; call "
            "compute_index first."
        )
    if risk_column not in frame.columns:
        raise SeverityInputError(
            f"Impact severity is unavailable: risk column {risk_column!r} is "
            "missing."
        )

    out = frame.copy()
    bands = out[risk_column].map(normalize_risk_band)
    tiers = out["exposure_tier"].astype("string")
    priorities = [
        priority_for(band, tier)
        for band, tier in zip(bands.tolist(), tiers.tolist())
    ]
    out["impact_priority"] = pd.array(priorities, dtype="Int64")
    out["impact_priority_label"] = [
        PRIORITY_LABELS.get(priority, NOT_APPLICABLE)
        for priority in priorities
    ]
    return out


def assess_impact(
    frame: pd.DataFrame,
    risk_column: str = "risk",
    thresholds: Sequence[float] | None = None,
    manifest: Mapping[str, object] | None = None,
) -> SeverityResult:
    """Compute IEI and priority, returning an explicit availability result."""
    try:
        resolved = resolve_thresholds(thresholds, manifest)
        enriched = compute_index(
            frame, thresholds=resolved, manifest=manifest
        )
        if risk_column in enriched.columns:
            enriched = assign_priority(enriched, risk_column=risk_column)
        else:
            enriched["impact_priority"] = pd.array(
                [None] * len(enriched), dtype="Int64"
            )
            enriched["impact_priority_label"] = NOT_APPLICABLE
    except SeverityInputError as exc:
        fuel_missing_in_frame = (
            frame is None
            or "fuel_area_km2" not in frame.columns
            or pd.to_numeric(frame.get("fuel_area_km2"), errors="coerce").isna().all()
        )
        required_cols = (
            IEI_NON_FUEL_COLUMNS if fuel_missing_in_frame else IEI_REQUIRED_COLUMNS
        )
        return SeverityResult(
            status=config.STATUS_UNAVAILABLE,
            frame=None,
            message=str(exc),
            thresholds=tuple(thresholds or DEFAULT_TIER_THRESHOLDS),
            missing_components=tuple(
                column
                for column in required_cols
                if frame is None or column not in frame.columns
            ),
        )

    fuel_missing = bool(
        enriched["iei_fuel_missing"].iloc[0]
        if "iei_fuel_missing" in enriched.columns and not enriched.empty
        else False
    )
    message = (
        "Impact Exposure Index and priority computed without fuel component "
        "(pop+infra only; weights renormalized: pop 0.35/0.65, infra 0.30/0.65). "
        "Priority is a heuristic, not a damage probability."
        if fuel_missing
        else (
            "Impact Exposure Index and priority computed. Priority is a "
            "heuristic, not a damage probability."
        )
    )

    return SeverityResult(
        status=config.STATUS_AVAILABLE,
        frame=enriched,
        message=message,
        thresholds=tuple(resolved),
    )


# ---------------------------------------------------------------------------
# Documentation helpers
# ---------------------------------------------------------------------------
def formula_text(fuel_missing: bool = False) -> str:
    """Human-readable IEI formula for reports and UI."""
    if fuel_missing:
        return (
            "IEI = (0.35/0.65) × pop_norm + (0.30/0.65) × infra_norm; "
            "infra_norm = 0.45 × road_norm + 0.35 × building_norm + "
            "0.20 × critical_norm. Computed without fuel component (WorldCover "
            "not ingested; weights renormalized to population and infrastructure). "
            "pop_norm and infrastructure sub-components are divided by their "
            "study-area 95th percentile."
        )
    return (
        "IEI = 0.35 × fuel_norm + 0.35 × pop_norm + 0.30 × infra_norm; "
        "infra_norm = 0.45 × road_norm + 0.35 × building_norm + "
        "0.20 × critical_norm. fuel_norm = flammable area / cell area; "
        "pop_norm and infrastructure sub-components are divided by their "
        "study-area 95th percentile."
    )


def priority_matrix_rows() -> list[dict[str, object]]:
    """Flatten the priority matrix into report-friendly rows."""
    rows: list[dict[str, object]] = []
    for band in RISK_BANDS:
        for tier in TIER_LABELS:
            priority = PRIORITY_MATRIX[band][tier]
            rows.append(
                {
                    "risk_band": band,
                    "exposure_tier": tier,
                    "priority": priority,
                    "priority_label": PRIORITY_LABELS[priority],
                }
            )
    return rows


__all__ = [
    "IEI_WEIGHTS",
    "INFRA_SUBWEIGHTS",
    "TIER_LABELS",
    "DEFAULT_TIER_THRESHOLDS",
    "RISK_BANDS",
    "PRIORITY_MATRIX",
    "PRIORITY_LABELS",
    "IEI_REQUIRED_COLUMNS",
    "IEI_NON_FUEL_COLUMNS",
    "IEI_WEIGHTS_NO_FUEL",
    "IEI_COMPONENT_COLUMNS",
    "NOT_APPLICABLE",
    "TIER_UNAVAILABLE",
    "SeverityInputError",
    "SeverityResult",
    "resolve_thresholds",
    "classify_iei",
    "tier_for_iei",
    "compute_index",
    "normalize_risk_band",
    "priority_for",
    "assign_priority",
    "assess_impact",
    "formula_text",
    "priority_matrix_rows",
]
