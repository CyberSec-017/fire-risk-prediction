"""Core logic for the wildfire-risk Streamlit dashboard.

This module deliberately has **no Streamlit imports** so the prediction,
feature-building and analytics logic can be unit-tested directly.  The UI in
``app.py`` is a thin presentation layer over these functions.

The authoritative model contract is ``models/model_card.json``:
feature names, their order, the feature hash and the risk bands all come from
that file.  The app never hard-codes a different feature order.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import requests

# The pipeline modules use ``from config import ...`` and are normally run as
# ``python src/<script>.py``.  Make src/ importable before importing them.
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from build_panel import add_features  # noqa: E402

# Model artifacts ship in <project>/models/.  These are the only two path
# constants for the trained model and its contract.
MODEL_PATH = ROOT / "models" / "fire_model.joblib"
MODEL_CARD_PATH = ROOT / "models" / "model_card.json"
FIRE_PANEL_PATH = config.FIRE_PANEL

# The model's own training facts.  These deliberately do NOT come from
# src/config.py, whose START_YEAR/END_YEAR describe the data-collection window
# (2024-2026), not the years used to fit/calibrate/test this artifact.
MODEL_TRAIN_YEARS = (2018, 2020)
MODEL_CALIBRATION_YEAR = 2021
MODEL_TEST_YEARS = (2022, 2024)
MODEL_LAT_MIN, MODEL_LAT_MAX = 32.0, 37.5
MODEL_LON_MIN, MODEL_LON_MAX = -2.0, 9.0
MODEL_LAT_STEP = 0.5
MODEL_LON_STEP = 0.625

# NASA POWER's regional endpoint is used by the pipeline; a single point uses
# the sibling /point endpoint.
POWER_POINT_URL = config.POWER_URL.replace("/regional", "/point")
HISTORY_DAYS = 120
REQUEST_TIMEOUT = 90

# Fallbacks are used only when the model card is missing *and* the model is
# also missing (a pure DEMO run).  They mirror the trained contract so the
# dashboard remains usable, but they are never used to validate a real model.
FALLBACK_FEATURE_NAMES = [
    "lat_cell", "lon_cell",
    "T2M", "T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT",
    "rain_7d", "tmax_7d", "rhmin_7d",
    "rain_14d", "tmax_14d", "rhmin_14d",
    "rain_30d", "tmax_30d", "rhmin_30d",
    "dry_days", "doy_sin", "doy_cos",
]
FALLBACK_RISK_BINS = [0.0, 0.01, 0.05, 0.15, 1.0]
FALLBACK_RISK_LABELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]

RISK_COLORS = {
    "LOW": "#2ecc71",
    "MODERATE": "#f1c40f",
    "HIGH": "#ff8c42",
    "VERY HIGH": "#ff3b30",
}
RISK_DISPLAY = {
    "LOW": "Low",
    "MODERATE": "Moderate",
    "HIGH": "High",
    "VERY HIGH": "Extreme",
}

FEATURE_DESCRIPTIONS = {
    "lat_cell": "Latitude of the snapped NASA POWER grid cell (degrees).",
    "lon_cell": "Longitude of the snapped NASA POWER grid cell (degrees).",
    "T2M": "Daily mean air temperature at 2 m (°C).",
    "T2M_MAX": "Daily maximum air temperature at 2 m (°C).",
    "RH2M": "Daily relative humidity at 2 m (%).",
    "WS2M": "Daily wind speed at 2 m (m/s).",
    "PRECTOTCORR": "Daily corrected precipitation (mm/day).",
    "GWETROOT": "Root-zone soil wetness (0–1).",
    "rain_7d": "Cumulative precipitation over the previous 7 days (mm).",
    "tmax_7d": "Mean daily maximum temperature over the previous 7 days (°C).",
    "rhmin_7d": "Minimum relative humidity over the previous 7 days (%).",
    "rain_14d": "Cumulative precipitation over the previous 14 days (mm).",
    "tmax_14d": "Mean daily maximum temperature over the previous 14 days (°C).",
    "rhmin_14d": "Minimum relative humidity over the previous 14 days (%).",
    "rain_30d": "Cumulative precipitation over the previous 30 days (mm).",
    "tmax_30d": "Mean daily maximum temperature over the previous 30 days (°C).",
    "rhmin_30d": "Minimum relative humidity over the previous 30 days (%).",
    "dry_days": "Consecutive days since precipitation exceeded 1 mm.",
    "doy_sin": "Sine of day-of-year; captures seasonality smoothly.",
    "doy_cos": "Cosine of day-of-year; captures seasonality smoothly.",
}

SENSITIVITY_SPECS = {
    "T2M": {"label": "Mean temperature (°C)", "values": [float(v) for v in range(0, 51, 5)]},
    "T2M_MAX": {"label": "Daily max temperature (°C)", "values": [float(v) for v in range(5, 56, 5)]},
    "RH2M": {"label": "Relative humidity (%)", "values": [float(v) for v in range(5, 101, 5)]},
    "WS2M": {"label": "Wind speed (m/s)", "values": [float(v) for v in range(0, 21, 2)]},
    "PRECTOTCORR": {"label": "Daily precipitation (mm)", "values": [float(v) for v in range(0, 21, 2)]},
    "GWETROOT": {"label": "Root-zone soil wetness (0–1)", "values": [round(float(v), 2) for v in np.linspace(0, 1, 11)]},
    "dry_days": {"label": "Days since rain > 1 mm", "values": [float(v) for v in range(0, 61, 5)]},
}


class FeatureContractError(ValueError):
    """The model card and the feature vector disagree."""


class FeatureBuildError(ValueError):
    """A usable feature row could not be built from the supplied inputs."""


class PowerFetchError(RuntimeError):
    """NASA POWER could not supply usable weather history."""


class ModelNotReadyError(RuntimeError):
    """The app tried to score with a model that is not available."""


@dataclass(frozen=True)
class ModelBundle:
    """Result of loading and validating the model + model card."""

    status: str                 # "REAL", "DEMO" or "ERROR"
    model: Any | None
    card: dict[str, Any] | None
    error: str | None
    model_path: Path
    card_path: Path

    @property
    def is_real(self) -> bool:
        return self.status == "REAL"

    @property
    def feature_names(self) -> list[str]:
        return list(self.card["feature_names"]) if self.card else list(FALLBACK_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------

def feature_hash(features: Sequence[str]) -> str:
    """Stable fingerprint of feature order, matching train_baseline.py."""
    payload = json.dumps(list(features), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_card(card: Mapping[str, Any]) -> None:
    """Raise FeatureContractError if a model card is incomplete or inconsistent."""
    if not isinstance(card, Mapping):
        raise FeatureContractError("Model card is not a JSON object.")

    required = ("feature_names", "n_features", "feature_hash", "risk_bins", "risk_labels")
    missing = [key for key in required if key not in card]
    if missing:
        raise FeatureContractError(f"Model card is missing required key(s): {missing}.")

    names = card["feature_names"]
    if not isinstance(names, list) or not names:
        raise FeatureContractError("Model card 'feature_names' must be a non-empty list.")
    if any(not isinstance(name, str) or not name for name in names):
        raise FeatureContractError("Every model feature name must be a non-empty string.")
    if len(set(names)) != len(names):
        raise FeatureContractError("Model card feature names contain duplicates.")
    if int(card["n_features"]) != len(names):
        raise FeatureContractError(
            f"Model card n_features={card['n_features']} but lists {len(names)} features."
        )
    if card["feature_hash"] != feature_hash(names):
        raise FeatureContractError(
            "Model card feature_hash does not match its feature_names/order."
        )

    bins = card["risk_bins"]
    labels = card["risk_labels"]
    if not isinstance(bins, list) or len(bins) < 2:
        raise FeatureContractError("Model card 'risk_bins' must contain at least two edges.")
    if not isinstance(labels, list) or len(labels) != len(bins) - 1:
        raise FeatureContractError("Model card risk labels must match the number of risk bins.")
    try:
        numeric_bins = [float(edge) for edge in bins]
    except (TypeError, ValueError) as exc:
        raise FeatureContractError("Model card risk bin edges must be numeric.") from exc
    if any(b <= a for a, b in zip(numeric_bins, numeric_bins[1:])):
        raise FeatureContractError("Model card risk bin edges must be strictly increasing.")
    if numeric_bins[0] > 0 or numeric_bins[-1] < 1:
        raise FeatureContractError("Risk bins must start at 0 and end at 1 or above.")


def unwrap_model_artifact(payload: Any, card: Mapping[str, Any]) -> Any:
    """Return the estimator from a bare model or a packaged artifact.

    The leader's artifact is a dict::

        {"model": <estimator>, "features": [...], "horizon_days": 3,
         "risk_bins": [...], "risk_labels": [...]}

    Older artifacts are a bare estimator.  When a package is supplied, its
    metadata is cross-checked against the card so a mismatched pair fails loudly
    instead of silently scoring with the wrong contract.
    """
    if not isinstance(payload, dict):
        return payload

    if "model" not in payload:
        raise FeatureContractError(
            "Model artifact is a dict without a 'model' entry."
        )
    estimator = payload["model"]

    if "features" in payload:
        features = [str(name) for name in payload["features"]]
        if features != list(card["feature_names"]):
            raise FeatureContractError(
                "Artifact feature names/order do not match model_card.json."
            )

    if "risk_bins" in payload:
        bins = [float(edge) for edge in payload["risk_bins"]]
        card_bins = [float(edge) for edge in card["risk_bins"]]
        if bins != card_bins:
            raise FeatureContractError(
                "Artifact risk bins do not match model_card.json."
            )

    if "risk_labels" in payload:
        labels = [str(label) for label in payload["risk_labels"]]
        card_labels = [str(label) for label in card["risk_labels"]]
        if labels != card_labels:
            raise FeatureContractError(
                "Artifact risk labels do not match model_card.json."
            )

    if "horizon_days" in payload and "horizon_days" in card:
        if int(payload["horizon_days"]) != int(card["horizon_days"]):
            raise FeatureContractError(
                "Artifact horizon_days does not match model_card.json."
            )

    return estimator


def validate_model(model: Any, card: Mapping[str, Any]) -> None:
    """Raise FeatureContractError if a loaded model is incompatible with the card."""
    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        raise FeatureContractError("Loaded model has no callable predict_proba().")

    n_features = getattr(model, "n_features_in_", None)
    if n_features is None:
        raise FeatureContractError("Loaded model does not expose n_features_in_.")
    if int(n_features) != int(card["n_features"]):
        raise FeatureContractError(
            f"Model expects {n_features} features but the card declares "
            f"{card['n_features']}."
        )

    names_in = getattr(model, "feature_names_in_", None)
    if names_in is not None:
        names_in = [str(name) for name in names_in]
        expected = list(card["feature_names"])
        if names_in != expected:
            raise FeatureContractError(
                "Model feature names/order do not match model_card.json. "
                "Retrain or restore the matching artifact pair."
            )

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise FeatureContractError("Loaded model does not expose classes_.")
    if 1 not in list(classes):
        raise FeatureContractError("Loaded model has no positive (fire) class.")


def load_model_bundle(
    model_path: str | Path = MODEL_PATH,
    card_path: str | Path = MODEL_CARD_PATH,
) -> ModelBundle:
    """Load and validate the model/card pair.

    States:
        REAL  - both artifacts exist, are readable and agree with each other.
        DEMO  - the model artifact is missing; the UI uses a transparent
                heuristic scorer.  A missing card is tolerated here.
        ERROR - the model exists but the card is missing/invalid, the model
                cannot be unpickled, or the contract does not match.  There is
                deliberately no silent fallback to DEMO in this state.
    """
    model_path = Path(model_path)
    card_path = Path(card_path)

    card: dict[str, Any] | None = None
    card_error: str | None = None
    if card_path.exists():
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            validate_card(card)
        except Exception as exc:  # noqa: BLE001 - report any card problem verbatim
            card_error = f"Model card is invalid: {exc}"
            card = None
    else:
        card_error = f"Model card not found at {card_path}."

    if not model_path.exists():
        return ModelBundle(
            status="DEMO",
            model=None,
            card=card,
            error=card_error,
            model_path=model_path,
            card_path=card_path,
        )

    if card is None:
        return ModelBundle(
            status="ERROR",
            model=None,
            card=None,
            error=card_error or "Model card is required to validate the model.",
            model_path=model_path,
            card_path=card_path,
        )

    try:
        payload = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - corrupt/unreadable artifact
        return ModelBundle(
            status="ERROR",
            model=None,
            card=card,
            error=f"Could not load model artifact: {exc}",
            model_path=model_path,
            card_path=card_path,
        )

    try:
        model = unwrap_model_artifact(payload, card)
    except Exception as exc:  # noqa: BLE001 - packaged metadata disagrees with card
        return ModelBundle(
            status="ERROR",
            model=None,
            card=card,
            error=f"Model artifact does not match the model card: {exc}",
            model_path=model_path,
            card_path=card_path,
        )

    try:
        validate_model(model, card)
    except Exception as exc:  # noqa: BLE001 - incompatible artifact pair
        return ModelBundle(
            status="ERROR",
            model=None,
            card=card,
            error=str(exc),
            model_path=model_path,
            card_path=card_path,
        )

    return ModelBundle(
        status="REAL",
        model=model,
        card=card,
        error=None,
        model_path=model_path,
        card_path=card_path,
    )


# ---------------------------------------------------------------------------
# Grid / date helpers
# ---------------------------------------------------------------------------

def snap_to_grid(lat: float, lon: float) -> tuple[float, float]:
    """Snap a coordinate to the NASA POWER grid used by the pipeline."""
    lat_cell = round(float(lat) / MODEL_LAT_STEP) * MODEL_LAT_STEP
    lon_cell = round(float(lon) / MODEL_LON_STEP) * MODEL_LON_STEP
    return round(float(lat_cell), 4), round(float(lon_cell), 4)


def doy_features(when: str | date | datetime | pd.Timestamp) -> tuple[float, float]:
    """Return the same sin/cos seasonality features used by build_panel.py."""
    doy = pd.Timestamp(when).day_of_year
    return (
        float(np.sin(2 * np.pi * doy / 365.25)),
        float(np.cos(2 * np.pi * doy / 365.25)),
    )


def training_bounds() -> dict[str, Any]:
    """The geographic and temporal envelope used to train the model."""
    return {
        "lat_min": MODEL_LAT_MIN,
        "lat_max": MODEL_LAT_MAX,
        "lon_min": MODEL_LON_MIN,
        "lon_max": MODEL_LON_MAX,
        "start_year": MODEL_TRAIN_YEARS[0],
        "end_year": MODEL_TEST_YEARS[1],
        "lat_step": MODEL_LAT_STEP,
        "lon_step": MODEL_LON_STEP,
        "train_years": MODEL_TRAIN_YEARS,
        "calibration_year": MODEL_CALIBRATION_YEAR,
        "test_years": MODEL_TEST_YEARS,
    }


def data_period_note(date: str | date | datetime | pd.Timestamp) -> str:
    """Describe where a date sits in the model's own train/calibrate/test timeline."""
    year = int(pd.Timestamp(date).year)
    if MODEL_TRAIN_YEARS[0] <= year <= MODEL_TRAIN_YEARS[1]:
        return "training period"
    if year == MODEL_CALIBRATION_YEAR:
        return "calibration year"
    if MODEL_TEST_YEARS[0] <= year <= MODEL_TEST_YEARS[1]:
        return "original test period"
    return "new data, never used for this model"


def check_training_area(
    lat: float,
    lon: float,
    when: str | date | datetime | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Check a snapped point (and optionally date) against the model envelope."""
    lat_cell, lon_cell = snap_to_grid(lat, lon)
    in_box = (
        MODEL_LAT_MIN <= lat_cell <= MODEL_LAT_MAX
        and MODEL_LON_MIN <= lon_cell <= MODEL_LON_MAX
    )
    period = data_period_note(when) if when is not None else None
    in_years = (
        period != "new data, never used for this model"
        if period is not None
        else True
    )

    warnings: list[str] = []
    if not in_box:
        warnings.append(
            "This point is outside the model's study area "
            f"({MODEL_LAT_MIN}–{MODEL_LAT_MAX}°N, "
            f"{MODEL_LON_MIN}–{MODEL_LON_MAX}°E). "
            "The model is extrapolating; treat the result as indicative only."
        )
    if not in_years:
        warnings.append(
            "This date is outside the period used for this model "
            f"(training {MODEL_TRAIN_YEARS[0]}–{MODEL_TRAIN_YEARS[1]}, "
            f"calibration {MODEL_CALIBRATION_YEAR}, "
            f"test {MODEL_TEST_YEARS[0]}–{MODEL_TEST_YEARS[1]}). "
            "The result is an extrapolation, not an in-sample estimate."
        )

    return {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "in_box": bool(in_box),
        "in_years": bool(in_years),
        "period": period,
        "warnings": warnings,
    }


def point_in_training_box(
    lat: float,
    lon: float,
    when: str | date | datetime | pd.Timestamp | None = None,
) -> bool:
    """Convenience boolean wrapper around check_training_area()."""
    info = check_training_area(lat, lon, when)
    return bool(info["in_box"] and info["in_years"])


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _feature_order(card: Mapping[str, Any] | None) -> list[str]:
    names = list(card["feature_names"]) if card else list(FALLBACK_FEATURE_NAMES)
    if len(set(names)) != len(names):
        raise FeatureContractError("Feature contract contains duplicate names.")
    return names


def _finite_row(values: Mapping[str, float], names: Sequence[str]) -> pd.DataFrame:
    missing = [name for name in names if name not in values]
    if missing:
        raise FeatureContractError(
            "The feature builder cannot produce the model contract; "
            f"missing: {missing}."
        )
    row: dict[str, float] = {}
    for name in names:
        try:
            value = float(values[name])
        except (TypeError, ValueError) as exc:
            raise FeatureBuildError(f"Feature {name!r} is not numeric.") from exc
        if not math.isfinite(value):
            raise FeatureBuildError(f"Feature {name!r} is not finite.")
        row[name] = value
    return pd.DataFrame([row], columns=list(names))


def build_manual_features(
    lat: float,
    lon: float,
    when: str | date | datetime | pd.Timestamp,
    *,
    t2m: float,
    t2m_max: float,
    rh2m: float,
    ws2m: float,
    precip: float,
    gwetroot: float,
    dry_days: float | None = None,
    card: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the model feature row from a single day of manual weather.

    Manual mode cannot know the previous 30 days.  It therefore uses a
    transparent persistence assumption: the entered daily weather is treated
    as representative of the antecedent window.  The UI labels this clearly.

    ``dry_days`` is exposed as an advanced input because it is the one
    antecedent feature that cannot be inferred from a single day at all.
    """
    lat_cell, lon_cell = snap_to_grid(lat, lon)
    doy_sin, doy_cos = doy_features(when)

    precip = float(precip)
    if dry_days is None:
        dry_days = 0.0 if precip > 1.0 else 10.0

    values: dict[str, float] = {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "T2M": float(t2m),
        "T2M_MAX": float(t2m_max),
        "RH2M": float(rh2m),
        "WS2M": float(ws2m),
        "PRECTOTCORR": precip,
        "GWETROOT": float(gwetroot),
        "rain_7d": precip * 7.0,
        "tmax_7d": float(t2m_max),
        "rhmin_7d": float(rh2m),
        "rain_14d": precip * 14.0,
        "tmax_14d": float(t2m_max),
        "rhmin_14d": float(rh2m),
        "rain_30d": precip * 30.0,
        "tmax_30d": float(t2m_max),
        "rhmin_30d": float(rh2m),
        "dry_days": max(0.0, float(dry_days)),
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }
    return _finite_row(values, _feature_order(card))


def parse_power_point_json(
    payload: Mapping[str, Any] | str,
    parameters: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Parse a NASA POWER daily point response into a tidy DataFrame."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise PowerFetchError("NASA POWER returned malformed JSON.") from exc

    try:
        parameter_data = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        raise PowerFetchError(
            "NASA POWER response is missing properties.parameter."
        ) from exc
    if not isinstance(parameter_data, Mapping):
        raise PowerFetchError("NASA POWER properties.parameter is not an object.")

    requested = list(parameters) if parameters is not None else list(parameter_data.keys())
    if not requested:
        raise PowerFetchError("No NASA POWER parameters were requested.")

    dates: set[str] = set()
    for series in parameter_data.values():
        if isinstance(series, Mapping):
            dates.update(str(key) for key in series.keys())
    if not dates:
        raise PowerFetchError("NASA POWER returned no daily values.")

    records: list[dict[str, Any]] = []
    for key in sorted(dates):
        try:
            stamp = datetime.strptime(key, "%Y%m%d")
        except ValueError:
            continue
        row: dict[str, Any] = {"date": pd.Timestamp(stamp)}
        for name in requested:
            series = parameter_data.get(name)
            if not isinstance(series, Mapping):
                raise PowerFetchError(
                    f"NASA POWER response is missing requested parameter {name}."
                )
            raw = series.get(key, np.nan)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = np.nan
            # POWER uses large negative sentinels for missing values.
            if math.isfinite(value) and value <= -900:
                value = np.nan
            row[name] = value
        records.append(row)

    if not records:
        raise PowerFetchError("NASA POWER returned no parseable dates.")
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def fetch_power_history(
    lat: float,
    lon: float,
    end_date: str | date | datetime | pd.Timestamp,
    *,
    days: int = HISTORY_DAYS,
    timeout: int = REQUEST_TIMEOUT,
) -> pd.DataFrame:
    """Fetch daily NASA POWER history for one point.

    The window is deliberately longer than the 30-day rolling features so
    ``dry_days`` has enough antecedent history.  Data is returned in a tidy
    frame with the snapped grid-cell coordinates attached.
    """
    end = pd.Timestamp(end_date).normalize()
    start = end - pd.Timedelta(days=max(1, int(days)) - 1)

    params = {
        "parameters": ",".join(config.PARAMETERS),
        "community": "AG",
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }

    try:
        response = requests.get(POWER_POINT_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise PowerFetchError(f"Could not reach NASA POWER: {exc}") from exc

    if response.status_code != 200:
        raise PowerFetchError(
            f"NASA POWER returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise PowerFetchError("NASA POWER returned a non-JSON response.") from exc

    frame = parse_power_point_json(payload, config.PARAMETERS)
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if frame.empty:
        raise PowerFetchError(
            "NASA POWER returned no data for the requested window. "
            "Choose an earlier date or check the coordinates."
        )

    if frame["date"].max() < end:
        raise PowerFetchError(
            f"NASA POWER has data only through {frame['date'].max():%Y-%m-%d}; "
            f"choose a date on or before that."
        )

    missing_columns = [name for name in config.PARAMETERS if name not in frame.columns]
    if missing_columns:
        raise PowerFetchError(
            f"NASA POWER response is missing column(s): {missing_columns}."
        )

    # Rolling features are only trustworthy if the recent weather is complete.
    if frame["PRECTOTCORR"].isna().any():
        raise PowerFetchError(
            "NASA POWER returned gaps in PRECTOTCORR over the requested window; "
            "the dry-spell features would be unreliable. Try a different date."
        )
    recent = frame.tail(31)
    for name in ("T2M_MAX", "RH2M"):
        if recent[name].isna().any():
            raise PowerFetchError(
                f"NASA POWER returned gaps in {name} within the last 31 days; "
                "the rolling features would be unreliable. Try a different date."
            )

    last = frame.iloc[-1]
    missing_last = [name for name in config.PARAMETERS if pd.isna(last[name])]
    if missing_last:
        raise PowerFetchError(
            f"NASA POWER has no usable {missing_last} value on the selected date."
        )

    lat_cell, lon_cell = snap_to_grid(lat, lon)
    frame["lat_cell"] = lat_cell
    frame["lon_cell"] = lon_cell
    frame[list(config.PARAMETERS)] = (
        frame[list(config.PARAMETERS)]
        .replace([np.inf, -np.inf], np.nan)
    )
    return frame.sort_values("date").reset_index(drop=True)


def build_live_features(
    history: pd.DataFrame,
    lat: float,
    lon: float,
    when: str | date | datetime | pd.Timestamp,
    *,
    card: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the model feature row from fetched NASA POWER history.

    Uses the pipeline's own ``build_panel.add_features`` so the rolling
    windows are computed exactly as they were during training.
    """
    required = {
        "date", "lat_cell", "lon_cell",
        "T2M", "T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise FeatureBuildError(f"History is missing column(s): {missing}.")
    if history.empty:
        raise FeatureBuildError("History is empty.")

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = (
        frame.drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    lat_cell, lon_cell = snap_to_grid(lat, lon)
    frame["lat_cell"] = lat_cell
    frame["lon_cell"] = lon_cell

    featured = add_features(frame)
    target = pd.Timestamp(when).normalize()
    matches = featured[featured["date"] == target]
    if matches.empty:
        earlier = featured[featured["date"] < target]
        if earlier.empty:
            raise FeatureBuildError(
                f"No weather history on or before {target:%Y-%m-%d}."
            )
        row = earlier.iloc[-1]
    else:
        row = matches.iloc[-1]

    names = _feature_order(card)
    values: dict[str, float] = {}
    for name in names:
        if name not in row.index:
            raise FeatureContractError(
                f"The live feature builder cannot produce model feature {name!r}."
            )
        values[name] = row[name]
    return _finite_row(values, names)


def _ensure_feature_frame(
    features: pd.DataFrame | Mapping[str, float],
    card: Mapping[str, Any] | None,
) -> pd.DataFrame:
    names = _feature_order(card)
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise FeatureContractError("Expected exactly one feature row.")
        if list(features.columns) != names:
            if set(features.columns) == set(names):
                features = features[names]
            else:
                raise FeatureContractError(
                    "Feature columns do not match model_card.json."
                )
        return features
    if isinstance(features, Mapping):
        return _finite_row(features, names)
    raise TypeError("Features must be a one-row DataFrame or a mapping.")


# ---------------------------------------------------------------------------
# Scoring and risk presentation
# ---------------------------------------------------------------------------

def predict_probability(
    bundle: ModelBundle,
    features: pd.DataFrame | Mapping[str, float],
) -> float:
    """Return the calibrated probability for the card's target horizon.

    The shipped model predicts at least one fire in the cell within
    ``horizon_days`` (3) days, not merely on the supplied date.
    """
    if bundle.status != "REAL" or bundle.model is None:
        raise ModelNotReadyError(
            bundle.error or "The trained model is not available."
        )

    frame = _ensure_feature_frame(features, bundle.card)
    try:
        proba = bundle.model.predict_proba(frame)
    except Exception as exc:  # noqa: BLE001 - surface the model's own message
        raise ModelNotReadyError(f"Model prediction failed: {exc}") from exc

    classes = list(getattr(bundle.model, "classes_", [0, 1]))
    try:
        positive_index = classes.index(1)
    except ValueError as exc:
        raise ModelNotReadyError("Model has no positive class.") from exc

    probability = float(proba[0, positive_index])
    if not math.isfinite(probability):
        raise ModelNotReadyError("Model returned a non-finite probability.")
    return min(max(probability, 0.0), 1.0)


def predict_batch(
    bundle: ModelBundle,
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """Score many rows at once, preserving missing values for the model.

    Columns are selected in the model-card order (extra columns are ignored).
    NaN values are deliberately left untouched: the trained artifact handles
    missing predictors such as GWETROOT natively.
    """
    if bundle.status != "REAL" or bundle.model is None:
        raise ModelNotReadyError(
            bundle.error or "The trained model is not available."
        )
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    names = _feature_order(bundle.card)
    missing = [name for name in names if name not in dataframe.columns]
    if missing:
        raise FeatureContractError(
            "Feature columns do not match model_card.json; "
            f"missing: {missing}."
        )

    frame = dataframe.loc[:, names].copy()
    try:
        probabilities = bundle.model.predict_proba(frame)
    except Exception as exc:  # noqa: BLE001 - surface the model's own message
        raise ModelNotReadyError(f"Model prediction failed: {exc}") from exc

    classes = list(getattr(bundle.model, "classes_", [0, 1]))
    try:
        positive_index = classes.index(1)
    except ValueError as exc:
        raise ModelNotReadyError("Model has no positive class.") from exc

    scored = np.asarray(probabilities[:, positive_index], dtype=float)
    return np.clip(scored, 0.0, 1.0)


def demo_probability(
    features: pd.DataFrame | Mapping[str, float],
    card: Mapping[str, Any] | None = None,
) -> float:
    """Deterministic, clearly-labelled heuristic used only in DEMO mode.

    This is not a trained model and must never be described as one.  It is a
    monotone, bounded score over the same weather inputs so the dashboard can
    still be demonstrated when the artifact is absent.
    """
    frame = _ensure_feature_frame(features, card)
    row = frame.iloc[0]

    def get(name: str, default: float) -> float:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    t2m = get("T2M", 20.0)
    tmax = get("T2M_MAX", t2m)
    rh = get("RH2M", 50.0)
    wind = get("WS2M", 3.0)
    precip = get("PRECTOTCORR", 0.0)
    soil = get("GWETROOT", 0.3)
    dry = get("dry_days", 0.0)
    doy_cos = get("doy_cos", 0.0)

    heat = min(max((tmax - 20.0) / 20.0, 0.0), 1.0)
    dry_air = min(max(1.0 - rh / 100.0, 0.0), 1.0)
    wind_norm = min(max(wind / 15.0, 0.0), 1.0)
    rain = min(max(precip / 10.0, 0.0), 1.0)
    soil_dry = min(max(1.0 - soil, 0.0), 1.0)
    dry_norm = min(max(dry / 30.0, 0.0), 1.0)
    season = min(max(-doy_cos, 0.0), 1.0)

    logit = (
        -4.6
        + 1.35 * heat
        + 1.15 * dry_air
        + 0.55 * wind_norm
        - 0.75 * rain
        + 0.85 * soil_dry
        + 0.90 * dry_norm
        + 1.10 * season
    )
    probability = 1.0 / (1.0 + math.exp(-logit)) * 0.22
    return float(min(max(probability, 0.0002), 0.5))


def score_features(
    bundle: ModelBundle,
    features: pd.DataFrame | Mapping[str, float],
) -> tuple[float, str]:
    """Score with the real model, or with the DEMO heuristic when appropriate."""
    if bundle.status == "REAL":
        return predict_probability(bundle, features), "REAL"
    if bundle.status == "DEMO":
        return demo_probability(features, bundle.card), "DEMO"
    raise ModelNotReadyError(bundle.error or "Model is not available.")


def _risk_spec(
    card: Mapping[str, Any] | None,
) -> tuple[list[float], list[str]]:
    if card and "risk_bins" in card and "risk_labels" in card:
        bins = [float(edge) for edge in card["risk_bins"]]
        labels = [str(label) for label in card["risk_labels"]]
    else:
        bins = list(FALLBACK_RISK_BINS)
        labels = list(FALLBACK_RISK_LABELS)
    return bins, labels


def classify_probabilities(
    probabilities: Sequence[float] | np.ndarray,
    card: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Vectorised risk-band assignment using the card's bins."""
    bins, labels = _risk_spec(card)
    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    indices = np.searchsorted(np.asarray(bins, dtype=float), probs, side="left") - 1
    indices = np.clip(indices, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[indices]


def risk_level(
    probability: float,
    card: Mapping[str, Any] | None = None,
) -> str:
    """Return the card's risk label for a calibrated probability."""
    return str(classify_probabilities([float(probability)], card)[0])


def risk_index(
    probability: float,
    card: Mapping[str, Any] | None = None,
) -> float:
    """Map a probability onto a presentational 0–100 index.

    Each risk band occupies an equal quarter of the index, with linear
    interpolation inside the band.  This is a display aid, not a model output.
    """
    bins, labels = _risk_spec(card)
    p = min(max(float(probability), 0.0), 1.0)
    band = int(np.searchsorted(np.asarray(bins, dtype=float), p, side="right") - 1)
    band = max(0, min(band, len(labels) - 1))
    low, high = bins[band], bins[band + 1]
    fraction = 0.0 if high <= low else min(max((p - low) / (high - low), 0.0), 1.0)
    return float((band + fraction) * (100.0 / len(labels)))


def risk_display(label: str | None) -> str:
    if label is None:
        return "Unknown"
    return RISK_DISPLAY.get(str(label).upper(), str(label).title())


def risk_color(label: str | None) -> str:
    if label is None:
        return "#ff6b35"
    return RISK_COLORS.get(str(label).upper(), "#ff6b35")


def format_probability(probability: float | None) -> str:
    if probability is None or not math.isfinite(float(probability)):
        return "—"
    p = float(probability)
    if p <= 0:
        return "0.00%"
    if p < 0.0001:
        return "<0.01%"
    if p < 0.01:
        return f"{p * 100:.2f}%"
    return f"{p:.1%}"


def format_percent(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def format_number(value: float | int | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    try:
        if not math.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return f"{float(value):,.{digits}f}"


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

def perturb_feature_row(
    base_row: pd.DataFrame | Mapping[str, float],
    variable: str,
    value: float,
) -> pd.DataFrame:
    """Return a copy of a feature row with one weather driver varied.

    Derived rolling features are updated with the same persistence assumption
    used by manual mode, so the sensitivity chart stays internally consistent.
    """
    if isinstance(base_row, pd.DataFrame):
        if len(base_row) != 1:
            raise FeatureContractError("Sensitivity needs exactly one base row.")
        columns = list(base_row.columns)
        values = {name: base_row.iloc[0][name] for name in columns}
    elif isinstance(base_row, Mapping):
        values = dict(base_row)
        columns = list(values.keys())
    else:
        raise TypeError("base_row must be a one-row DataFrame or a mapping.")

    variable = str(variable)
    value = float(value)
    if variable == "T2M":
        values["T2M"] = value
    elif variable == "T2M_MAX":
        values["T2M_MAX"] = value
        for window in (7, 14, 30):
            values[f"tmax_{window}d"] = value
    elif variable == "RH2M":
        values["RH2M"] = value
        for window in (7, 14, 30):
            values[f"rhmin_{window}d"] = value
    elif variable == "WS2M":
        values["WS2M"] = value
    elif variable == "PRECTOTCORR":
        values["PRECTOTCORR"] = value
        for window in (7, 14, 30):
            values[f"rain_{window}d"] = value * window
    elif variable == "GWETROOT":
        values["GWETROOT"] = value
    elif variable == "dry_days":
        values["dry_days"] = value
    else:
        raise ValueError(f"Unsupported sensitivity variable: {variable!r}")

    return pd.DataFrame(
        [{column: values[column] for column in columns}],
        columns=columns,
    )


def sensitivity_table(
    bundle: ModelBundle,
    base_row: pd.DataFrame | Mapping[str, float],
    variable: str,
    values: Sequence[float],
) -> pd.DataFrame:
    """Score a one-variable sweep for the native Streamlit charts."""
    rows: list[dict[str, Any]] = []
    for value in values:
        row = perturb_feature_row(base_row, variable, value)
        probability, _source = score_features(bundle, row)
        label = risk_level(probability, bundle.card)
        rows.append({
            "value": float(value),
            "probability": probability,
            "risk": label,
            "risk_display": risk_display(label),
        })
    return pd.DataFrame(rows, columns=["value", "probability", "risk", "risk_display"])


# ---------------------------------------------------------------------------
# Panel analytics
# ---------------------------------------------------------------------------

def panel_summary(panel: pd.DataFrame) -> dict[str, Any]:
    """Headline facts about the processed cell-day panel."""
    if panel is None or panel.empty:
        return {
            "rows": 0, "cells": 0, "start": None, "end": None,
            "years": 0, "fire_days": 0, "fire_rate": float("nan"),
        }
    dates = pd.to_datetime(panel["date"])
    fire_days = int(panel["burned"].sum()) if "burned" in panel else 0
    fire_rate = float(panel["burned"].mean()) if "burned" in panel else float("nan")
    return {
        "rows": int(len(panel)),
        "cells": int(panel[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]),
        "start": dates.min(),
        "end": dates.max(),
        "years": int(dates.dt.year.nunique()),
        "fire_days": fire_days,
        "fire_rate": fire_rate,
    }


def cell_fire_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-cell historical fire-day rate, used for the map overlay."""
    required = {"lat_cell", "lon_cell", "burned"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise FeatureBuildError(f"Panel is missing column(s): {missing}.")
    stats = (
        panel.groupby(["lat_cell", "lon_cell"], as_index=False)
        .agg(
            fire_days=("burned", "sum"),
            observed_days=("burned", "size"),
            fire_rate=("burned", "mean"),
        )
    )
    stats["fire_days"] = stats["fire_days"].astype(int)
    stats["observed_days"] = stats["observed_days"].astype(int)
    return stats.sort_values("fire_rate", ascending=False).reset_index(drop=True)


def monthly_fire_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly fire-day rate across the whole panel."""
    required = {"date", "burned"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise FeatureBuildError(f"Panel is missing column(s): {missing}.")
    frame = panel.assign(month=pd.to_datetime(panel["date"]).dt.month)
    stats = (
        frame.groupby("month", as_index=False)
        .agg(fire_days=("burned", "sum"), fire_rate=("burned", "mean"))
    )
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    stats["month_name"] = stats["month"].map(month_names)
    return stats


def yearly_fire_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Yearly fire-day counts and rates across the whole panel."""
    required = {"date", "burned"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise FeatureBuildError(f"Panel is missing column(s): {missing}.")
    frame = panel.assign(year=pd.to_datetime(panel["date"]).dt.year)
    return (
        frame.groupby("year", as_index=False)
        .agg(fire_days=("burned", "sum"), fire_rate=("burned", "mean"))
    )


def median_feature_row(
    panel: pd.DataFrame,
    card: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Median feature vector, used for global sensitivity charts."""
    names = _feature_order(card)
    missing = [name for name in names if name not in panel.columns]
    if missing:
        raise FeatureBuildError(
            f"Panel is missing model feature(s): {missing}."
        )
    medians = panel[names].median(numeric_only=True)
    if medians.isna().any():
        bad = [name for name in names if pd.isna(medians[name])]
        raise FeatureBuildError(f"Could not compute medians for: {bad}.")
    return pd.DataFrame([{name: float(medians[name]) for name in names}], columns=names)


def risk_calibration(
    panel: pd.DataFrame,
    bundle: ModelBundle,
    *,
    sample_n: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare predicted and observed fire rates by risk band.

    Rows are sampled from the test years (years after the card's ``val_year``)
    to avoid presenting training-year fit as out-of-sample evidence.  The
    result is a sampled diagnostic, not a replacement for the full test
    metrics in the model card.

    The shipped model predicts fire within ``horizon_days`` days, while the
    panel's ``burned`` column is a same-day label.  When a horizon greater than
    one is declared, the observed rate is therefore built from a forward-looking
    window of that length (right-censored at the end of each cell's series).
    """
    if bundle.status != "REAL" or bundle.card is None:
        return pd.DataFrame(columns=["risk", "n", "predicted", "observed"])

    names = _feature_order(bundle.card)
    missing = [name for name in names if name not in panel.columns]
    if missing:
        raise FeatureBuildError(f"Panel is missing model feature(s): {missing}.")

    frame = panel
    if "date" in frame.columns and "val_year" in bundle.card:
        years = pd.to_datetime(frame["date"]).dt.year
        test_frame = frame[years > int(bundle.card["val_year"])]
        if not test_frame.empty:
            frame = test_frame

    frame = frame.copy()
    horizon = max(int(bundle.card.get("horizon_days", 1)), 1)
    if horizon > 1 and {"lat_cell", "lon_cell", "date"}.issubset(frame.columns):
        frame = frame.sort_values(["lat_cell", "lon_cell", "date"])
        frame["observed_target"] = (
            frame.groupby(["lat_cell", "lon_cell"], sort=False)["burned"]
            .transform(
                lambda values: values[::-1]
                .rolling(horizon, min_periods=1)
                .max()[::-1]
            )
        )
    else:
        frame["observed_target"] = frame["burned"]

    if len(frame) > sample_n:
        frame = frame.sample(n=sample_n, random_state=random_state)

    features = frame[names].astype(float)
    probabilities = bundle.model.predict_proba(features)
    classes = list(getattr(bundle.model, "classes_", [0, 1]))
    positive_index = classes.index(1)
    predicted = probabilities[:, positive_index]
    labels = classify_probabilities(predicted, bundle.card)

    scored = pd.DataFrame({
        "burned": frame["observed_target"].to_numpy(),
        "probability": predicted,
        "risk": labels,
    })

    rows: list[dict[str, Any]] = []
    for label in bundle.card["risk_labels"]:
        subset = scored[scored["risk"] == label]
        rows.append({
            "risk": label,
            "n": int(len(subset)),
            "predicted": float(subset["probability"].mean()) if len(subset) else float("nan"),
            "observed": float(subset["burned"].mean()) if len(subset) else float("nan"),
        })
    return pd.DataFrame(rows, columns=["risk", "n", "predicted", "observed"])


# ---------------------------------------------------------------------------
# Presentation helpers shared by the UI
# ---------------------------------------------------------------------------

def feature_group(name: str) -> str:
    if name in ("lat_cell", "lon_cell"):
        return "Location"
    if name in ("T2M", "T2M_MAX", "RH2M", "WS2M", "PRECTOTCORR", "GWETROOT"):
        return "Daily weather"
    if name.startswith("rain_"):
        return "Antecedent rain"
    if name.startswith("tmax_"):
        return "Antecedent heat"
    if name.startswith("rhmin_"):
        return "Antecedent humidity"
    if name == "dry_days":
        return "Dry spell"
    if name.startswith("doy_"):
        return "Seasonality"
    return "Other"


def feature_contract_table(card: Mapping[str, Any] | None = None) -> pd.DataFrame:
    names = _feature_order(card)
    return pd.DataFrame({
        "feature": names,
        "group": [feature_group(name) for name in names],
        "description": [FEATURE_DESCRIPTIONS.get(name, "Model feature.") for name in names],
    })
