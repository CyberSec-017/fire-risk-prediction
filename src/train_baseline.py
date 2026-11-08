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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from config import FIRE_PANEL, END_YEAR

USE_COORDS = True    # run once with True, once with False, and report both

EXCLUDE_COLS = {"date", "burned", "n_detections", "total_frp"}
if not USE_COORDS:
    EXCLUDE_COLS |= {"lat_cell", "lon_cell"}


def load_data():
    df = pd.read_parquet(FIRE_PANEL).dropna(subset=["T2M"])
    FEATURE_SET = "full"    # run with "geo_season" once, then back to "full"
    if FEATURE_SET == "geo_season":
        features = ["lat_cell", "lon_cell", "doy_sin", "doy_cos"]
    else:
        features = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, features

VAL_YEAR = 2021   # change after Step 1 if that year looks unusual

def time_split(df: pd.DataFrame):
    val_start = pd.Timestamp(f"{VAL_YEAR}-01-01")
    test_start = pd.Timestamp(f"{VAL_YEAR + 1}-01-01")
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    print(f"  train {len(train):,} | val {len(val):,} | test {len(test):,}")
    return train, val, test


def main():
    print("Loading panel ...")
    df, features = load_data()
    print(f"  {len(features)} features: {features}")

    train, val, test = time_split(df)
    tr = train.assign(m=train.date.dt.month)
    clim = tr.groupby(["lat_cell", "lon_cell", "m"])["burned"].mean().rename("clim")
    te = test.assign(m=test.date.dt.month).join(clim, on=["lat_cell", "lon_cell", "m"])
    print("climatology PR-AUC:", average_precision_score(te["burned"], te["clim"].fillna(0)))

    print("Training HistGradientBoostingClassifier ...")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=5,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False
    )
    model.fit(train[features], train["burned"])
    pv = model.predict_proba(val[features])[:, 1]
    print("val PR-AUC:", average_precision_score(val["burned"], pv))
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    cal.fit(val[features], val["burned"])

    proba = cal.predict_proba(test[features])[:, 1]
    pv_cal = cal.predict_proba(val[features])[:, 1]
    bins = [0, 0.005, 0.02, 0.06, 1.0]
    labels = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
    for name, data, p in [("val", val, pv_cal), ("test", test, proba)]:
        lv = pd.cut(p, bins=bins, labels=labels, include_lowest=True)
        print(name)
        print(data.assign(level=lv).groupby("level", observed=True)["burned"]
                  .agg(n="size", fire_rate="mean"))
    chk = test.assign(pred=proba)
    print(chk.groupby(chk.date.dt.year).agg(observed=("burned", "mean"), predicted=("pred", "mean")))

    deciles = pd.qcut(proba, 10, duplicates="drop")
    print(chk.groupby(deciles, observed=True).agg(
        n=("burned", "size"), predicted=("pred", "mean"), observed=("burned", "mean")))

    top = proba >= pd.Series(proba).quantile(0.9)
    print("share of fires in top 10% risk:", test.loc[top, "burned"].sum() / test["burned"].sum())

    pr_auc = average_precision_score(test["burned"], proba)
    base_rate = test["burned"].mean()
    brier = brier_score_loss(test["burned"], proba)
    
    bss = 1 - brier / (base_rate * (1 - base_rate))
    print(f"  Brier skill: {bss:.4f}")

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
