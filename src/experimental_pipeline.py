"""Opt-in scikit-learn Pipeline/ColumnTransformer for training experiments only.

The deployed model is a calibrated HistGradientBoostingClassifier trained on the
20 contract features in ``app_core.py``. It already handles NaN values natively,
so imputation or scaling must not be forced into production inference.

This module exists for offline feature-engineering experiments. It makes the
numeric feature schema explicit with ``ColumnTransformer`` and can optionally
add imputation/scaling for estimators that require it (for example logistic
regression or KNN). By default the transformer is a passthrough, so results stay
comparable with the existing raw-DataFrame experiment runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_experimental_pipeline(
    estimator: Any | None = None,
    *,
    features: Sequence[str],
    impute: bool = False,
    scale: bool = False,
    add_missing_indicator: bool = False,
) -> Pipeline:
    """Wrap an estimator in a numeric-only ColumnTransformer pipeline.

    Parameters
    ----------
    estimator:
        Base classifier. Defaults to a fresh HistGradientBoostingClassifier.
        The caller is responsible for passing tuned experimental parameters.
    features:
        Exact ordered feature names to select from the input DataFrame. This is
        the schema guard provided by ColumnTransformer.
    impute, scale:
        Optional steps for estimators that cannot handle NaN/unscaled inputs.
        Leave both ``False`` for HistGradientBoosting experiments so its native
        missing-value handling and split behaviour are preserved.
    add_missing_indicator:
        Passed to ``SimpleImputer`` when imputation is enabled.
    """
    feature_names = list(features)
    if not feature_names:
        raise ValueError("features must contain at least one column name")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("features must not contain duplicates")
    if add_missing_indicator and not impute:
        raise ValueError("add_missing_indicator requires impute=True")

    steps: list[tuple[str, Any]] = []
    if impute:
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=add_missing_indicator,
                ),
            )
        )
    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_transformer: Any = Pipeline(steps) if steps else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = estimator if estimator is not None else HistGradientBoostingClassifier(
        random_state=42
    )
    return Pipeline([("preprocess", preprocessor), ("clf", classifier)])
