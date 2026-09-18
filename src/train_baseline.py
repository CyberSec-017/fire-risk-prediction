"""
STEP 5 -- Modeling and Evaluation

Trains a gradient-boosted classifier on the fire panel and evaluates it
with metrics appropriate for a rare-event problem (PR-AUC, Brier score)
rather than plain accuracy, which is meaningless when ~98% of rows are
the negative class.

The train/test split is done BY TIME (train on early years, test on the
most recent year), not at random -- adjacent days are highly correlated,
so a random split would leak information from the test set into training.

Coordinates (lat_cell, lon_cell) are deliberately excluded from the
feature set: including them lets the model just memorize which squares
are fire-prone, instead of learning from weather, which defeats the
project's stated goal of predicting risk from user-supplied weather
inputs.

Input:  data/processed/fire_panel.parquet
Usage:
    python src/train_baseline.py
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from config import FIRE_PANEL, END_YEAR

EXCLUDE_COLS = {
    "lat_cell", "lon_cell", "date", "burned", "n_detections", "total_frp",
}


def load_data():
    df = pd.read_parquet(FIRE_PANEL).dropna(subset=["T2M"])
    features = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, features


def time_split(df: pd.DataFrame):
    cutoff = pd.Timestamp(f"{END_YEAR - 1}-01-01")
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    print(f"  train: {len(train):,} rows (before {cutoff.date()})")
    print(f"  test : {len(test):,} rows (from {cutoff.date()} onward)")
    return train, test


def main():
    print("Loading panel ...")
    df, features = load_data()
    print(f"  {len(features)} features: {features}")

    train, test = time_split(df)

    print("Training HistGradientBoostingClassifier ...")
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced"
    )
    model.fit(train[features], train["burned"])

    proba = model.predict_proba(test[features])[:, 1]

    pr_auc = average_precision_score(test["burned"], proba)
    base_rate = test["burned"].mean()
    brier = brier_score_loss(test["burned"], proba)

    print("\nResults")
    print(f"  PR-AUC     : {pr_auc:.4f}")
    print(f"  base rate  : {base_rate:.4f}   (PR-AUC of a random guess)")
    print(f"  Brier score: {brier:.4f}   (lower is better, calibration)")

    # Feature importance is useful for the report's "which factors matter
    # most" discussion.
    importances = pd.Series(
        model.feature_importances_ if hasattr(model, "feature_importances_")
        else [None] * len(features),
        index=features,
    )
    if importances.notna().any():
        print("\nTop features:")
        print(importances.sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
