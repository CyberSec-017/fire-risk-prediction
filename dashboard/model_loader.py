"""
FIREWATCH AI — Model loader.

Provides a clean integration point for Member 4's ML model.
When a trained model file is available, it will be loaded here.
Until then, all functions return None / sentinel values so that
the prediction page can display an informative placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.config import (
    PROJECT_ROOT,
    EXCLUDE_FROM_PREDICTION,
    WEATHER_VARIABLES,
    ENGINEERED_FEATURES,
)

# ---------------------------------------------------------------------------
# Expected model path — Member 4 should save the trained model here
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fire_risk_model.pkl"

# Feature list the model expects (weather + engineered, no coords/target)
MODEL_FEATURES: list[str] = (
    list(WEATHER_VARIABLES.keys()) + ENGINEERED_FEATURES
)


@st.cache_resource(show_spinner="Loading ML model …")
def load_model() -> Any | None:
    """Load the trained fire risk model.

    Returns
    -------
    model : sklearn estimator or None
        The fitted model if the file exists, otherwise None.
    """
    if not MODEL_PATH.exists():
        return None

    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        return model
    except Exception:
        try:
            import pickle
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


def get_model_features() -> list[str]:
    """Return the list of features the model expects."""
    return MODEL_FEATURES


def predict_fire_risk(
    model: Any,
    input_data: dict[str, float],
) -> dict[str, Any]:
    """Run a prediction using the loaded model.

    Parameters
    ----------
    model : fitted sklearn estimator
    input_data : dict mapping feature name → value

    Returns
    -------
    result : dict with keys:
        - 'risk_level' : str ('Low', 'Moderate', 'High', 'Very High')
        - 'probability' : float (0–1) or None
        - 'raw_prediction' : int (0 or 1)
    """
    import numpy as np

    features = get_model_features()
    X = pd.DataFrame([{f: input_data.get(f, 0.0) for f in features}])

    raw_pred = int(model.predict(X)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # Map probability to risk levels
    if probability is not None:
        if probability < 0.15:
            risk_level = "Low"
        elif probability < 0.40:
            risk_level = "Moderate"
        elif probability < 0.70:
            risk_level = "High"
        else:
            risk_level = "Very High"
    else:
        risk_level = "High" if raw_pred == 1 else "Low"

    return {
        "risk_level": risk_level,
        "probability": probability,
        "raw_prediction": raw_pred,
    }

