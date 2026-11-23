"""Pure batch-inference logic for FireWatch AI.

The module validates an uploaded CSV against the authoritative model card,
scores only complete and valid rows with the deployed calibrated model, and
builds a transparent export.  It deliberately performs **no imputation**:
missing or invalid values are reported per row instead of being replaced.

Keeping this logic free of Streamlit makes the validation, scoring and export
contracts directly unit-testable.  The Streamlit renderer lives in
``dashboard/batch_view.py``.
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_core  # noqa: E402

HORIZON_DAYS = 3

DERIVED_COLUMNS = [
    "row_number",
    "validation_status",
    "predicted_probability",
    "risk_category",
    "horizon_days",
    "rejection_reasons",
]

STATUS_ACCEPTED = "ACCEPTED"
STATUS_REJECTED = "REJECTED"

# Generous physical bounds: they reject impossible / mistyped values without
# rejecting observations present in the historical panel.  Coordinates are
# checked separately against the model's grid and geographic area.
FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "lat_cell": {"label": "Latitude (centre de cellule)", "unit": "°N", "min": 32.0, "max": 37.5},
    "lon_cell": {"label": "Longitude (centre de cellule)", "unit": "°E", "min": -2.0, "max": 9.0},
    "T2M": {"label": "Température moyenne à 2 m", "unit": "°C", "min": -20.0, "max": 55.0},
    "T2M_MAX": {"label": "Température maximale à 2 m", "unit": "°C", "min": -20.0, "max": 60.0},
    "RH2M": {"label": "Humidité relative à 2 m", "unit": "%", "min": 0.0, "max": 100.0},
    "WS2M": {"label": "Vitesse du vent à 2 m", "unit": "m/s", "min": 0.0, "max": 40.0},
    "PRECTOTCORR": {"label": "Précipitation quotidienne corrigée", "unit": "mm", "min": 0.0, "max": 1000.0},
    "GWETROOT": {"label": "Humidité du sol racinaire", "unit": "fraction", "min": 0.0, "max": 1.0},
    "rain_7d": {"label": "Cumul de pluie sur 7 jours", "unit": "mm", "min": 0.0, "max": 2000.0},
    "tmax_7d": {"label": "Température max moyenne sur 7 jours", "unit": "°C", "min": -20.0, "max": 60.0},
    "rhmin_7d": {"label": "Humidité min moyenne sur 7 jours", "unit": "%", "min": 0.0, "max": 100.0},
    "rain_14d": {"label": "Cumul de pluie sur 14 jours", "unit": "mm", "min": 0.0, "max": 2500.0},
    "tmax_14d": {"label": "Température max moyenne sur 14 jours", "unit": "°C", "min": -20.0, "max": 60.0},
    "rhmin_14d": {"label": "Humidité min moyenne sur 14 jours", "unit": "%", "min": 0.0, "max": 100.0},
    "rain_30d": {"label": "Cumul de pluie sur 30 jours", "unit": "mm", "min": 0.0, "max": 3500.0},
    "tmax_30d": {"label": "Température max moyenne sur 30 jours", "unit": "°C", "min": -20.0, "max": 60.0},
    "rhmin_30d": {"label": "Humidité min moyenne sur 30 jours", "unit": "%", "min": 0.0, "max": 100.0},
    "dry_days": {"label": "Nombre de jours secs", "unit": "jours", "min": 0.0, "max": 365.0},
    "doy_sin": {"label": "Saisonnalité (sin)", "unit": "[-1; 1]", "min": -1.0, "max": 1.0},
    "doy_cos": {"label": "Saisonnalité (cos)", "unit": "[-1; 1]", "min": -1.0, "max": 1.0},
}


@dataclass(frozen=True)
class BatchValidation:
    """Row-level validation result, aligned with the uploaded DataFrame."""

    numeric: pd.DataFrame
    valid_mask: np.ndarray
    reasons: list[list[str]]


@dataclass(frozen=True)
class BatchInferenceResult:
    """Everything the UI and the CSV export need after one batch run."""

    table: pd.DataFrame
    accepted_row_numbers: list[int]
    accepted_features: pd.DataFrame
    accepted_probabilities: np.ndarray
    rejected_count: int

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_row_numbers)


def read_csv_bytes(payload: bytes | BinaryIO) -> pd.DataFrame:
    """Read an uploaded CSV without turning blanks or text into fake numbers."""
    source: BinaryIO = io.BytesIO(payload) if isinstance(payload, bytes) else payload
    return pd.read_csv(source, dtype=str, keep_default_na=False)


def template_csv_bytes(card: Mapping[str, Any]) -> bytes:
    """Return a header-only CSV containing exactly the card feature order."""
    features = [str(name) for name in card["feature_names"]]
    return pd.DataFrame(columns=features).to_csv(index=False).encode("utf-8-sig")


def model_features(card: Mapping[str, Any]) -> list[str]:
    """Return the 20 feature names in the deployed model-card order."""
    features = [str(name) for name in card["feature_names"]]
    if len(features) != int(card.get("n_features", len(features))):
        raise app_core.FeatureContractError(
            "The model card feature count does not match its feature_names list."
        )
    return features


def _is_blank(value: Any) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return True
    return isinstance(value, str) and not value.strip()


def _grid_reason(name: str, value: float) -> str | None:
    if name == "lat_cell":
        step = app_core.MODEL_LAT_STEP
    elif name == "lon_cell":
        step = app_core.MODEL_LON_STEP
    else:
        return None
    snapped = round(value / step) * step
    if not math.isclose(value, snapped, rel_tol=0.0, abs_tol=1e-6):
        return f"{name} n'est pas aligné sur la grille du modèle (pas de {step}°)"
    return None


def validate_batch(raw: pd.DataFrame, card: Mapping[str, Any]) -> BatchValidation:
    """Validate every row against the 20-feature model contract.

    A row is accepted only when every required column exists, every value is
    numeric and finite, every value is within a physically plausible range,
    and the coordinates lie on the model grid.  No value is imputed.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw must be a pandas DataFrame.")
    features = model_features(card)
    reasons_by_row: list[list[str]] = [[] for _ in range(len(raw))]

    missing_columns = [name for name in features if name not in raw.columns]
    numeric = pd.DataFrame(index=raw.index, columns=features, dtype=float)
    if missing_columns:
        joined = ", ".join(missing_columns)
        for reasons in reasons_by_row:
            reasons.append(f"Colonne requise absente: {joined}")
        return BatchValidation(
            numeric=numeric,
            valid_mask=np.zeros(len(raw), dtype=bool),
            reasons=reasons_by_row,
        )

    converted: dict[str, pd.Series] = {}
    for name in features:
        converted[name] = pd.to_numeric(raw[name], errors="coerce")
        numeric[name] = converted[name].to_numpy(dtype=float, na_value=np.nan)

    for row_position in range(len(raw)):
        row_reasons = reasons_by_row[row_position]
        for name in features:
            original = raw[name].iloc[row_position]
            value = converted[name].iloc[row_position]
            if _is_blank(original):
                row_reasons.append(f"Valeur manquante pour {name}")
                continue
            if pd.isna(value):
                shown = str(original).strip()[:40]
                row_reasons.append(f"Valeur non numérique pour {name}: « {shown} »")
                continue
            number = float(value)
            if not math.isfinite(number):
                row_reasons.append(f"Valeur non finie pour {name}")
                continue

            spec = FEATURE_SPECS[name]
            low, high = float(spec["min"]), float(spec["max"])
            if number < low - 1e-9 or number > high + 1e-9:
                row_reasons.append(
                    f"{name} hors plage [{low:g}; {high:g}] {spec['unit']}"
                )
                continue

            grid_reason = _grid_reason(name, number)
            if grid_reason:
                row_reasons.append(grid_reason)

    valid_mask = np.array([not reasons for reasons in reasons_by_row], dtype=bool)
    return BatchValidation(
        numeric=numeric.loc[:, features],
        valid_mask=valid_mask,
        reasons=reasons_by_row,
    )


def _output_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy the original inputs, avoiding collisions with derived columns."""
    renamed = raw.copy()
    rename: dict[Any, str] = {}
    used = set(str(column) for column in renamed.columns)
    for column in renamed.columns:
        name = str(column)
        if name not in DERIVED_COLUMNS:
            continue
        candidate = f"input_{name}"
        suffix = 2
        while candidate in used:
            candidate = f"input_{name}_{suffix}"
            suffix += 1
        rename[column] = candidate
        used.add(candidate)
    if rename:
        renamed = renamed.rename(columns=rename)
    return renamed.reset_index(drop=True)


def build_result_table(
    raw: pd.DataFrame,
    validation: BatchValidation,
    probabilities: np.ndarray,
    risk_labels: np.ndarray,
) -> pd.DataFrame:
    """Combine original inputs, validation status and model outputs."""
    original = _output_raw(raw)
    probabilities = np.asarray(probabilities, dtype=float)
    risk_labels = np.asarray(risk_labels, dtype=object)
    accepted_positions = np.flatnonzero(validation.valid_mask)

    derived = pd.DataFrame(
        {
            "row_number": np.arange(1, len(raw) + 1, dtype=int),
            "validation_status": np.where(
                validation.valid_mask, STATUS_ACCEPTED, STATUS_REJECTED
            ),
            "predicted_probability": np.full(len(raw), np.nan, dtype=float),
            "risk_category": np.full(len(raw), "", dtype=object),
            "horizon_days": np.full(len(raw), HORIZON_DAYS, dtype=int),
            "rejection_reasons": [
                "; ".join(reasons) for reasons in validation.reasons
            ],
        }
    )
    if len(accepted_positions):
        derived.loc[accepted_positions, "predicted_probability"] = probabilities
        derived.loc[accepted_positions, "risk_category"] = risk_labels

    return pd.concat([derived, original], axis=1).reset_index(drop=True)


def run_batch_inference(
    raw: pd.DataFrame,
    bundle: app_core.ModelBundle,
) -> BatchInferenceResult:
    """Validate, score with the deployed model, and build the export table."""
    if bundle.status != "REAL" or bundle.model is None:
        raise app_core.ModelNotReadyError(
            bundle.error or "The trained model is not available."
        )
    card = bundle.card or {}
    validation = validate_batch(raw, card)
    accepted = validation.numeric.loc[validation.valid_mask].reset_index(drop=True)

    if len(accepted):
        probabilities = app_core.predict_batch(bundle, accepted)
        risk_labels = app_core.classify_probabilities(probabilities, card)
    else:
        probabilities = np.array([], dtype=float)
        risk_labels = np.array([], dtype=object)

    table = build_result_table(raw, validation, probabilities, risk_labels)
    return BatchInferenceResult(
        table=table,
        accepted_row_numbers=(np.flatnonzero(validation.valid_mask) + 1).tolist(),
        accepted_features=accepted,
        accepted_probabilities=probabilities,
        rejected_count=int((~validation.valid_mask).sum()),
    )


def to_csv_bytes(table: pd.DataFrame) -> bytes:
    """Serialise a result table to a UTF-8 CSV with a BOM for Excel users."""
    return table.to_csv(index=False).encode("utf-8-sig")


def feature_schema_frame(card: Mapping[str, Any]) -> pd.DataFrame:
    """Return the documented 20-feature CSV schema for the UI."""
    features = model_features(card)
    rows = []
    for index, name in enumerate(features, start=1):
        spec = FEATURE_SPECS.get(
            name, {"label": name, "unit": "", "min": np.nan, "max": np.nan}
        )
        interval = f"{spec['min']:g} … {spec['max']:g}" if spec["unit"] else "—"
        rows.append(
            {
                "ordre": index,
                "colonne": name,
                "description": spec["label"],
                "unité": spec["unit"],
                "plage acceptée": interval,
            }
        )
    return pd.DataFrame(rows)
