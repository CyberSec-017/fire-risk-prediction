"""Tests for the opt-in experimental ColumnTransformer pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experimental_pipeline import build_experimental_pipeline  # noqa: E402


FEATURES = ["lat_cell", "T2M", "RH2M"]


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "lat_cell": rng.uniform(32.0, 37.5, 60),
            "T2M": rng.uniform(10.0, 40.0, 60),
            "RH2M": rng.uniform(10.0, 90.0, 60),
            "ignored_column": rng.normal(size=60),
        }
    )


def _labels() -> np.ndarray:
    return np.arange(60) % 2


def test_pipeline_selects_numeric_features_and_drops_the_rest():
    pipeline = build_experimental_pipeline(
        HistGradientBoostingClassifier(max_iter=5, random_state=42),
        features=FEATURES,
    )

    assert isinstance(pipeline, Pipeline)
    preprocessor = pipeline.named_steps["preprocess"]
    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.remainder == "drop"
    assert preprocessor.transformers[0][0] == "numeric"
    assert preprocessor.transformers[0][1] == "passthrough"
    assert preprocessor.transformers[0][2] == FEATURES


def test_pipeline_rejects_invalid_feature_specs():
    with pytest.raises(ValueError, match="at least one"):
        build_experimental_pipeline(features=[])
    with pytest.raises(ValueError, match="duplicates"):
        build_experimental_pipeline(features=["a", "a"])
    with pytest.raises(ValueError, match="requires impute"):
        build_experimental_pipeline(features=["a"], add_missing_indicator=True)


def test_passthrough_pipeline_matches_raw_histogram_gradient_boosting():
    frame = _frame()
    y = _labels()
    base = HistGradientBoostingClassifier(max_iter=10, random_state=42)

    raw = clone(base).fit(frame[FEATURES], y)
    piped = build_experimental_pipeline(clone(base), features=FEATURES).fit(frame, y)

    expected = raw.predict_proba(frame[FEATURES])[:, 1]
    actual = piped.predict_proba(frame)[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_optional_imputation_and_scaling_are_available_for_other_estimators():
    frame = _frame()
    frame.loc[0, "RH2M"] = np.nan
    y = _labels()

    pipeline = build_experimental_pipeline(
        LogisticRegression(max_iter=500, random_state=42),
        features=FEATURES,
        impute=True,
        scale=True,
        add_missing_indicator=True,
    )
    pipeline.fit(frame, y)
    probabilities = pipeline.predict_proba(frame)

    steps = pipeline.named_steps["preprocess"].transformers[0][1]
    assert isinstance(steps, Pipeline)
    assert [name for name, _ in steps.steps] == ["imputer", "scaler"]
    assert steps.named_steps["imputer"].add_indicator is True
    assert np.isfinite(probabilities).all()
