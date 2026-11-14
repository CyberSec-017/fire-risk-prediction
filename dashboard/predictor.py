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

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import app_core
from dashboard.config import MODEL_FEATURES

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
