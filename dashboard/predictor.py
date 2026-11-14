"""
FIREWATCH AI — Prediction-first UI helpers.

Pure, Streamlit-free functions powering the landing-page predictor and the
risk cockpit:

* build the 20-feature model row from a small set of human inputs,
* score a set of conditions with the calibrated model,
* explain the score with an honest one-feature-at-a-time sensitivity analysis,
* turn a risk band into concrete, non-alarmist guidance.

Keeping this module free of Streamlit makes it directly unit-testable.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import numpy as np
import pandas as pd

import app_core
from dashboard.config import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MODEL_FEATURES,
)

# ---------------------------------------------------------------------------
# Scenario presets shared by the landing page and the cockpit
# ---------------------------------------------------------------------------

PRESET_ORDER = ["Summer heatwave", "Dry spring", "Mild autumn", "Winter wet"]

PRESETS: dict[str, dict[str, float | str]] = {
    "Summer heatwave": {
        "label": "Summer heatwave",
        "description": "43 °C peaks, 15 % humidity, 42 dry days — the classic August fire setup.",
        "T2M_MAX": 43.5,
        "RH2M": 15.0,
        "WS2M": 8.5,
        "PRECTOTCORR": 0.0,
        "GWETROOT": 0.12,
        "dry_days": 42.0,
    },
    "Dry spring": {
        "label": "Dry spring",
        "description": "26 °C, 46 % humidity, a little rain — intermittent spring risk.",
        "T2M_MAX": 26.8,
        "RH2M": 46.0,
        "WS2M": 4.2,
        "PRECTOTCORR": 0.6,
        "GWETROOT": 0.44,
        "dry_days": 4.0,
    },
    "Mild autumn": {
        "label": "Mild autumn",
        "description": "22 °C, 58 % humidity, moderate soil moisture — low-to-moderate conditions.",
        "T2M_MAX": 22.0,
        "RH2M": 58.0,
        "WS2M": 3.4,
        "PRECTOTCORR": 1.2,
        "GWETROOT": 0.55,
        "dry_days": 3.0,
    },
    "Winter wet": {
        "label": "Winter wet",
        "description": "14 °C, 84 % humidity, 16 mm of rain — saturated fuels, minimal ignition.",
        "T2M_MAX": 14.0,
        "RH2M": 84.0,
        "WS2M": 5.0,
        "PRECTOTCORR": 16.5,
        "GWETROOT": 0.86,
        "dry_days": 0.0,
    },
}

# Features whose value is lowered/raised to a fire-suppressing reference in
# the sensitivity analysis.  Units are the model's native units.
SUPPRESSING_REFERENCE: dict[str, float] = {
    "T2M": 12.0,
    "T2M_MAX": 18.0,
    "RH2M": 85.0,
    "WS2M": 0.5,
    "PRECTOTCORR": 15.0,
    "GWETROOT": 0.85,
    "dry_days": 0.0,
    "tmax_7d": 18.0,
    "tmax_14d": 18.0,
    "tmax_30d": 18.0,
    "rhmin_7d": 75.0,
    "rhmin_14d": 75.0,
    "rhmin_30d": 75.0,
}

# Rain is "more is safer", so it needs a raised reference rather than a fixed
# one: add a realistic wet spell on top of whatever the user entered.
WET_REFERENCE_ADD = {
    "rain_7d": 40.0,
    "rain_14d": 70.0,
    "rain_30d": 120.0,
}

# The shared design system uses four title-case bands.  app_core.risk_display
# deliberately renders VERY HIGH as "Extreme" for compact badges, so the
# landing page keeps its own explicit mapping instead.
DISPLAY_LABELS: dict[str, str] = {
    "LOW": "Low",
    "MODERATE": "Moderate",
    "HIGH": "High",
    "VERY HIGH": "Very High",
}

FEATURE_LABELS: dict[str, str] = {
    "T2M_MAX": "Daily max temperature",
    "T2M": "Mean temperature",
    "RH2M": "Relative humidity",
    "WS2M": "Wind speed",
    "PRECTOTCORR": "Rain today",
    "GWETROOT": "Root-zone soil wetness",
    "dry_days": "Consecutive dry days",
    "rain_7d": "Rain over 7 days",
    "rain_14d": "Rain over 14 days",
    "rain_30d": "Rain over 30 days",
    "tmax_7d": "7-day mean max temperature",
    "tmax_14d": "14-day mean max temperature",
    "tmax_30d": "30-day mean max temperature",
    "rhmin_7d": "7-day minimum humidity",
    "rhmin_14d": "14-day minimum humidity",
    "rhmin_30d": "30-day minimum humidity",
}


@dataclass(frozen=True)
class Prediction:
    """A single scored set of conditions."""

    probability: float
    risk_level: str          # Title-case label used by the design system
    risk_label: str          # Raw model-card label (LOW, MODERATE, ...)
    risk_index: float        # 0–100 presentational index
    source: str              # REAL or DEMO
    raw_prediction: int

    @property
    def percent(self) -> float:
        return self.probability * 100.0


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _feature_names(card: Mapping[str, Any] | None) -> list[str]:
    return list(card["feature_names"]) if card else list(MODEL_FEATURES)


def build_feature_row(
    lat: float,
    lon: float,
    when: Any,
    *,
    t2m_max: float,
    rh2m: float,
    ws2m: float,
    precip: float,
    gwetroot: float,
    dry_days: float,
    card: Mapping[str, Any] | None = None,
    antecedent: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build a complete model row from a compact set of user inputs.

    ``app_core.build_manual_features`` supplies a transparent persistence
    assumption for the 7/14/30-day windows (today's weather repeated).
    ``antecedent`` may override any of those engineered features explicitly.
    """
    mean_temperature = float(t2m_max) - 5.0
    frame = app_core.build_manual_features(
        lat,
        lon,
        when,
        t2m=mean_temperature,
        t2m_max=float(t2m_max),
        rh2m=float(rh2m),
        ws2m=float(ws2m),
        precip=float(precip),
        gwetroot=float(gwetroot),
        dry_days=float(dry_days),
        card=card,
    )
    values = {name: float(value) for name, value in frame.iloc[0].items()}

    for name, value in (antecedent or {}).items():
        if name in values and value is not None and math.isfinite(float(value)):
            values[name] = float(value)

    names = _feature_names(card)
    missing = [name for name in names if name not in values]
    if missing:  # pragma: no cover - build_manual_features already enforces this
        raise app_core.FeatureContractError(
            f"Feature builder missed model feature(s): {missing}."
        )
    return {name: values[name] for name in names}


def condition_summary(features: Mapping[str, float]) -> dict[str, Any]:
    """Human-readable headline of the driving conditions."""
    return {
        "max_temp": float(features.get("T2M_MAX", float("nan"))),
        "humidity": float(features.get("RH2M", float("nan"))),
        "wind": float(features.get("WS2M", float("nan"))),
        "rain_7d": float(features.get("rain_7d", float("nan"))),
        "dry_days": float(features.get("dry_days", float("nan"))),
        "soil_wetness": float(features.get("GWETROOT", float("nan"))),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_conditions(
    bundle: Any,
    features: Mapping[str, float],
) -> Prediction:
    """Score one feature mapping with the real model (DEMO heuristic fallback)."""
    frame = pd.DataFrame([{name: float(features[name]) for name in _feature_names(bundle.card)}])
    probability, source = app_core.score_features(bundle, frame)
    label = app_core.risk_level(probability, bundle.card)
    display = DISPLAY_LABELS.get(str(label).upper(), str(label).title())

    return Prediction(
        probability=float(probability),
        risk_level=display,
        risk_label=str(label),
        risk_index=app_core.risk_index(probability, bundle.card),
        source=source,
        raw_prediction=int(probability >= 0.5),
    )


def feature_sensitivity(
    bundle: Any,
    features: Mapping[str, float],
    *,
    top_n: int = 7,
) -> pd.DataFrame:
    """One-feature-at-a-time sensitivity of the calibrated probability.

    Each row replaces a single predictor with a fire-suppressing reference and
    measures how much the probability falls (positive ``delta_pp``) or rises
    (negative ``delta_pp``).  This is a local, model-agnostic explanation, not
    a causal attribution.
    """
    names = _feature_names(bundle.card)
    present = [name for name in names if name in features]
    base_frame = pd.DataFrame(
        [{name: float(features[name]) for name in names}], columns=names
    )
    base_probability = float(app_core.predict_batch(bundle, base_frame)[0])

    rows: list[dict[str, Any]] = []
    for name in present:
        if name in {"lat_cell", "lon_cell", "doy_sin", "doy_cos"}:
            continue

        original = float(features[name])
        if name in SUPPRESSING_REFERENCE:
            reference = SUPPRESSING_REFERENCE[name]
        elif name in WET_REFERENCE_ADD:
            reference = original + WET_REFERENCE_ADD[name]
        else:
            continue

        if math.isclose(reference, original, rel_tol=0.0, abs_tol=1e-9):
            continue

        probe = dict(features)
        probe[name] = reference
        probe_frame = pd.DataFrame(
            [{key: float(probe[key]) for key in names}], columns=names
        )
        probe_probability = float(app_core.predict_batch(bundle, probe_frame)[0])

        delta_pp = (base_probability - probe_probability) * 100.0
        if abs(delta_pp) < 1e-4:
            continue

        rows.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": original,
                "reference": reference,
                "delta_pp": delta_pp,
                "direction": "increases" if delta_pp > 0 else "reduces",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["feature", "label", "value", "reference", "delta_pp", "direction"]
        )

    table = pd.DataFrame(rows)
    table["abs_delta"] = table["delta_pp"].abs()
    return (
        table.sort_values("abs_delta", ascending=False)
        .head(int(top_n))
        .drop(columns=["abs_delta"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------

_GUIDANCE: dict[str, list[str]] = {
    "LOW": [
        "Routine monitoring only — conditions do not favour ignition.",
        "Keep seasonal fuel-break maintenance on schedule.",
    ],
    "MODERATE": [
        "Avoid open flames and machinery that throws sparks near dry grass.",
        "Check local wind forecasts before any outdoor burning.",
        "Keep a water source and a phone within reach for field work.",
    ],
    "HIGH": [
        "Postpone burning, welding and brush clearing to a cooler, calmer day.",
        "Pre-position water tankers near the most exposed settlements.",
        "Brief field crews on escape routes and assembly points.",
        "Report the first smell of smoke immediately — early response is decisive.",
    ],
    "VERY HIGH": [
        "Treat the area as an active fire-weather warning zone.",
        "Suspend all outdoor burning and spark-generating work.",
        "Alert civil-protection units and pre-stage suppression resources.",
        "Prepare evacuation routes for settlements downwind of dry fuels.",
        "Re-check the forecast every few hours; conditions can escalate quickly.",
    ],
}


def guidance_for(risk_label: str) -> list[str]:
    """Return practical guidance for a model-card risk label."""
    key = str(risk_label).upper()
    return list(_GUIDANCE.get(key, _GUIDANCE["MODERATE"]))


def risk_narrative(risk_label: str) -> str:
    """One-sentence plain-language meaning of the risk band."""
    key = str(risk_label).upper()
    return {
        "LOW": "Fire ignition is unlikely under these conditions.",
        "MODERATE": "Fire is possible; dry fuels and breeze can carry a small ignition.",
        "HIGH": "Fire is plausible; heat, dry air and cured fuels line up.",
        "VERY HIGH": "Fire weather is critical; any ignition could spread rapidly.",
    }.get(key, "Conditions are being evaluated.")


def baseline_comparison(probability: float, base_rate: float) -> dict[str, Any]:
    """Compare a prediction against the historical fire base rate."""
    p = float(probability)
    base = float(base_rate)
    if base <= 0:
        return {"multiple": float("nan"), "delta_pp": float("nan")}
    return {
        "multiple": p / base,
        "delta_pp": (p - base) * 100.0,
    }


def format_pp(value: float, digits: int = 2) -> str:
    """Format a percentage-point value with an explicit sign."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.{digits}f} pp"


def rank_risk(probabilities: Sequence[float]) -> np.ndarray:
    """Percentile rank of probabilities within a batch (0–100)."""
    values = np.asarray(list(probabilities), dtype=float)
    if values.size == 0:
        return values
    order = values.argsort().argsort()
    return order / max(values.size - 1, 1) * 100.0


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------

def compare_presets(
    bundle: Any,
    lat: float,
    lon: float,
    when: Any,
    *,
    card: Mapping[str, Any] | None = None,
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score every scenario preset at the same location and date.

    Returns one row per scenario with the probability, the risk band and how
    that probability compares with the historical base rate.  This turns the
    presets into a small sensitivity study of the model itself.
    """
    card = card if card is not None else getattr(bundle, "card", None)
    base_rate = 0.0237
    if card:
        base_rate = float(card.get("metrics", {}).get("base_rate", base_rate) or base_rate)

    selected = list(names) if names is not None else list(PRESET_ORDER)
    rows: list[dict[str, Any]] = []
    for name in selected:
        preset = PRESETS[name]
        features = build_feature_row(
            lat,
            lon,
            when,
            t2m_max=float(preset["T2M_MAX"]),
            rh2m=float(preset["RH2M"]),
            ws2m=float(preset["WS2M"]),
            precip=float(preset["PRECTOTCORR"]),
            gwetroot=float(preset["GWETROOT"]),
            dry_days=float(preset["dry_days"]),
            card=card,
        )
        prediction = score_conditions(bundle, features)
        rows.append(
            {
                "scenario": name,
                "description": preset.get("description", ""),
                "probability": prediction.probability,
                "percent": prediction.percent,
                "risk_level": prediction.risk_level,
                "multiple": (
                    prediction.probability / base_rate if base_rate > 0 else float("nan")
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "description",
            "probability",
            "percent",
            "risk_level",
            "multiple",
        ],
    )


# ---------------------------------------------------------------------------
# Shareable links
# ---------------------------------------------------------------------------

def share_query(
    *,
    lat: float,
    lon: float,
    when: Any,
    scenario: str | None = None,
    run: bool = True,
) -> dict[str, str]:
    """Build the query parameters that reproduce one prediction."""
    date = when.isoformat() if hasattr(when, "isoformat") else str(when)
    params = {
        "lat": f"{float(lat):g}",
        "lon": f"{float(lon):g}",
        "date": date[:10],
    }
    if scenario:
        params["scenario"] = str(scenario)
    params["run"] = "1" if run else "0"
    return params


def share_url(base_url: str, params: Mapping[str, str]) -> str:
    """Append an encoded query string to a base URL."""
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{urlencode(dict(params))}"


def _to_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, low), high)


def read_share_params(
    params: Mapping[str, Any],
    *,
    default_date: dt.date | None = None,
) -> dict[str, Any]:
    """Normalise untrusted query parameters into usable widget defaults.

    Values outside the study area are clamped instead of rejected so a shared
    link always renders; an unknown scenario simply leaves the form as-is.
    """
    default_date = default_date or dt.date.today()
    when = default_date
    raw_date = params.get("date")
    if raw_date:
        try:
            when = dt.date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            when = default_date

    scenario = params.get("scenario")
    if scenario is not None and scenario not in PRESETS:
        scenario = None

    has_location = any(key in params for key in ("lat", "lon", "date", "scenario"))
    run_value = str(params.get("run", "")).strip().lower()
    return {
        "lat": _to_float(params.get("lat"), 36.5, LAT_MIN, LAT_MAX),
        "lon": _to_float(params.get("lon"), 3.125, LON_MIN, LON_MAX),
        "date": when,
        "scenario": scenario,
        "run": run_value in {"1", "true", "yes"},
        "from_link": has_location,
    }


# ---------------------------------------------------------------------------
# Staleness and export
# ---------------------------------------------------------------------------

def changed_features(
    current: Mapping[str, float],
    snapshot: Mapping[str, float],
    *,
    tolerance: float = 1e-6,
) -> list[str]:
    """Feature names whose value differs between the inputs and a snapshot."""
    changed: list[str] = []
    for name, previous in snapshot.items():
        if name not in current:
            continue
        try:
            same = math.isclose(
                float(current[name]), float(previous), rel_tol=0.0, abs_tol=tolerance
            )
        except (TypeError, ValueError):
            same = current[name] == previous
        if not same:
            changed.append(name)
    return changed


def feature_label(name: str) -> str:
    """Readable label for a model feature name."""
    return FEATURE_LABELS.get(name, name)


def prediction_payload(
    *,
    prediction: Any,
    features: Mapping[str, float],
    lat: float,
    lon: float,
    when: Any,
    scenario: str | None,
    model_version: str,
    base_rate: float,
    using_live: bool = False,
    sensitivity: pd.DataFrame | None = None,
    generated_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """JSON-serialisable record of a single prediction."""
    generated = generated_at or dt.datetime.now(dt.timezone.utc)
    comparison = baseline_comparison(prediction.probability, base_rate)
    drivers: list[dict[str, Any]] = []
    if sensitivity is not None and not sensitivity.empty:
        drivers = [
            {
                "feature": str(row.feature),
                "label": str(row.label),
                "delta_pp": round(float(row.delta_pp), 4),
                "direction": str(row.direction),
            }
            for row in sensitivity.itertuples(index=False)
        ]

    return {
        "project": "FIREWATCH AI",
        "generated_at": generated.isoformat(timespec="seconds"),
        "model": {
            "version": model_version,
            "source": prediction.source,
            "horizon_days": 3,
        },
        "location": {
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "date": str(when)[:10],
            "grid_cell": {
                "lat_cell": round(float(features.get("lat_cell", float("nan"))), 4),
                "lon_cell": round(float(features.get("lon_cell", float("nan"))), 4),
            },
        },
        "scenario": scenario,
        "inputs": {name: round(float(value), 4) for name, value in features.items()},
        "result": {
            "probability": round(float(prediction.probability), 6),
            "percent": round(float(prediction.percent), 4),
            "risk_level": prediction.risk_level,
            "risk_label": prediction.risk_label,
            "risk_index": round(float(prediction.risk_index), 2),
            "multiple_of_base_rate": (
                round(float(comparison["multiple"]), 3)
                if math.isfinite(float(comparison["multiple"]))
                else None
            ),
            "base_rate": round(float(base_rate), 6),
            "weather_source": "nasa_power_live" if using_live else "manual_inputs",
        },
        "drivers": drivers,
        "disclaimer": (
            "Calibrated probability of an observed fire in the next 3 days for a "
            "coarse grid cell. Decision-support signal, not an official emergency "
            "forecast."
        ),
    }

# ---------------------------------------------------------------------------
# Metric provenance
# ---------------------------------------------------------------------------

DRIFT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("pr_auc", "Precision-Recall AUC", "{:.4f}"),
    ("base_rate", "Base fire rate", "{:.2%}"),
    ("brier", "Brier score", "{:.5f}"),
    ("top_decile_fire_share", "Fires in the top decile", "{:.1%}"),
)


def read_evaluation_metrics(path: str | Path) -> dict[str, float]:
    """Read the test-period metrics from ``report/evaluation_summary.json``.

    Returns an empty mapping when the file is missing or malformed so the
    landing page can fall back to the model card without failing.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        return {}
    clean: dict[str, float] = {}
    for key in ("pr_auc", "base_rate", "brier", "top_decile_fire_share"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            clean[key] = float(value)
    return clean


def metric_drift(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Compare the model-card snapshot with the reproducible evaluation.

    Returns one row per shared metric. A row is marked ``changed`` when the two
    values differ beyond ``tolerance``; the raw delta is kept for plotting or
    sorting. Metrics missing from either side are skipped.
    """
    rows: list[dict[str, Any]] = []
    for key, label, template in DRIFT_METRICS:
        if key not in recorded or key not in current:
            continue
        try:
            old = float(recorded[key])
            new = float(current[key])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(old) and math.isfinite(new)):
            continue
        rows.append(
            {
                "metric": label,
                "model_card": template.format(old),
                "reproducible": template.format(new),
                "delta": new - old,
                "changed": not math.isclose(old, new, rel_tol=tolerance, abs_tol=tolerance),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["metric", "model_card", "reproducible", "delta", "changed"],
    )
