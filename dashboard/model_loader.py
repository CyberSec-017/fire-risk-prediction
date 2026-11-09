"""
FIREWATCH AI — Model loader.

Thin Streamlit-facing wrapper around ``app_core``.  The authoritative model
contract (feature names/order, feature hash and risk bands) lives in
``models/model_card.json``; this module does not duplicate it.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import (
    MODEL_CARD_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    PROJECT_ROOT,
)

# Import the shared core from the project root (src/config.py is imported by
# app_core itself for the data pipeline, not for the model contract).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_core  # noqa: E402

# Card labels are uppercase; the dashboard design system uses title case.
RISK_LEVEL_MAP = {
    "LOW": "Low",
    "MODERATE": "Moderate",
    "HIGH": "High",
    "VERY HIGH": "Very High",
}

OUTSIDE_RISK = "OUTSIDE"


def model_area_mask(frame: pd.DataFrame) -> pd.Series:
    """True where a cell is on the model's training grid.

    A valid cell is both inside the model's geographic bounds and aligned to
    the model's latitude/longitude grid steps.  ``fire_panel.parquet`` contains
    extra off-grid longitudes that the model never saw; those cells must not be
    scored.
    """
    lat = pd.to_numeric(frame["lat_cell"], errors="coerce")
    lon = pd.to_numeric(frame["lon_cell"], errors="coerce")
    lat_grid = np.round(lat / app_core.MODEL_LAT_STEP) * app_core.MODEL_LAT_STEP
    lon_grid = np.round(lon / app_core.MODEL_LON_STEP) * app_core.MODEL_LON_STEP
    on_grid = np.isclose(lat, lat_grid, atol=1e-6) & np.isclose(
        lon, lon_grid, atol=1e-6
    )
    inside = lat.between(
        app_core.MODEL_LAT_MIN, app_core.MODEL_LAT_MAX
    ) & lon.between(app_core.MODEL_LON_MIN, app_core.MODEL_LON_MAX)
    return (on_grid & inside).fillna(False)


@st.cache_resource(show_spinner="Loading ML model …")
def load_model() -> app_core.ModelBundle:
    """Load and validate the model + card pair.

    Returns the bundle with its state intact (``REAL``, ``DEMO`` or ``ERROR``).
    A broken model/card pair is never silently downgraded to DEMO.
    """
    return app_core.load_model_bundle(MODEL_PATH, MODEL_CARD_PATH)


def get_model_features() -> list[str]:
    """Return the 20 features in model-card order."""
    return list(MODEL_FEATURES)


def score_panel_date(
    bundle: app_core.ModelBundle,
    panel: pd.DataFrame,
    date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Score every panel cell for one date with the model-card feature order.

    Returns one row per cell with ``lat_cell``, ``lon_cell``, ``probability``,
    ``risk``, ``burned`` and ``in_model_area``.  Cells off the model's training
    grid are left unscored (``probability`` is NaN, ``risk`` is ``OUTSIDE``).
    Missing feature columns raise rather than being defaulted; NaN feature
    values for valid cells are passed through to the model.
    """
    columns = [
        "lat_cell",
        "lon_cell",
        "probability",
        "risk",
        "burned",
        "in_model_area",
    ]
    empty = pd.DataFrame(columns=columns)
    if bundle.status != "REAL" or bundle.model is None:
        raise app_core.ModelNotReadyError(
            bundle.error or "The trained model is not available."
        )
    if panel is None or panel.empty:
        return empty

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    target = pd.Timestamp(date).normalize()
    day = frame[frame["date"].dt.normalize() == target].copy()
    if day.empty:
        return empty

    in_area = model_area_mask(day).to_numpy()
    probabilities = np.full(len(day), np.nan, dtype=float)
    risks = np.full(len(day), OUTSIDE_RISK, dtype=object)

    if in_area.any():
        inside_probabilities = app_core.predict_batch(bundle, day.loc[in_area])
        probabilities[in_area] = inside_probabilities
        risks[in_area] = app_core.classify_probabilities(
            inside_probabilities, bundle.card
        )

    if "burned" in day.columns:
        burned = (
            pd.to_numeric(day["burned"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
    else:
        burned = np.zeros(len(day), dtype=int)

    return pd.DataFrame(
        {
            "lat_cell": day["lat_cell"].to_numpy(),
            "lon_cell": day["lon_cell"].to_numpy(),
            "probability": probabilities,
            "risk": risks,
            "burned": burned,
            "in_model_area": in_area,
        }
    ).reset_index(drop=True)


def predict_fire_risk(
    model: app_core.ModelBundle,
    input_data: dict[str, float],
) -> dict[str, Any]:
    """Run a single prediction with the bundle returned by :func:`load_model`.

    Parameters
    ----------
    model : app_core.ModelBundle
        Loaded and validated model bundle.
    input_data : dict mapping feature name → value
        Must contain every feature in ``MODEL_FEATURES``.  Missing features are
        an error; they are never defaulted to 0.0.

    Returns
    -------
    result : dict with keys:
        - 'risk_level' : str ('Low', 'Moderate', 'High', 'Very High')
        - 'probability' : float (0–1)
        - 'raw_prediction' : int (0 or 1)
    """
    if not isinstance(model, app_core.ModelBundle):
        raise TypeError("model must be the ModelBundle returned by load_model().")

    missing = [name for name in MODEL_FEATURES if name not in input_data]
    if missing:
        raise ValueError(
            "Missing model feature(s): "
            f"{missing}. The model needs all 20 card features."
        )

    frame = pd.DataFrame(
        [{name: input_data[name] for name in MODEL_FEATURES}],
        columns=MODEL_FEATURES,
    )

    if model.status == "REAL":
        probability = float(app_core.predict_batch(model, frame)[0])
    else:
        probability, _source = app_core.score_features(model, frame)

    card_label = app_core.risk_level(probability, model.card)
    risk_level = RISK_LEVEL_MAP.get(card_label.upper(), card_label.title())

    raw_prediction = int(probability >= 0.5)
    if model.status == "REAL" and model.model is not None:
        try:
            raw_prediction = int(model.model.predict(frame)[0])
        except Exception:  # noqa: BLE001 - keep probability as the primary output
            raw_prediction = int(probability >= 0.5)

    return {
        "risk_level": risk_level,
        "probability": float(probability),
        "raw_prediction": raw_prediction,
    }
