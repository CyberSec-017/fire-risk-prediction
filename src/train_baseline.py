"""
STEP 5 -- Modeling and Evaluation

Trains a gradient-boosted classifier on the fire panel, calibrates its
probabilities, and evaluates it with metrics suited to a rare-event
problem (PR-AUC, Brier skill, risk-level tables) rather than accuracy,
which is meaningless when ~98% of rows are the negative class.

Splits are done BY TIME, never at random, because adjacent days and
cells are strongly correlated and a random split would leak information:
    train = years before VAL_YEAR
    val   = VAL_YEAR (used only to fit the probability calibrator)
    test  = every year after VAL_YEAR

Baseline for comparison: per-cell, per-month fire climatology (no weather).

Experiment switches (see constants below):
    USE_COORDS   include lat/lon as features or not
    FEATURE_SET  "full" or "geo_season" (location + season only)

Input:  data/processed/fire_panel.parquet
Usage:
    python src/train_baseline.py
"""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

from config import FIRE_PANEL

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
USE_COORDS = True
FEATURE_SET = "full"     # "full" or "geo_season"
VAL_YEAR = 2021          # calibration year; test = all years after it

# Risk levels on the calibrated probability (tune on the validation table).
RISK_BINS = [0, 0.005, 0.02, 0.06, 1.0]
RISK_LABELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]

EXCLUDE_COLS = {"date", "burned", "n_detections", "total_frp"}
if not USE_COORDS:
    EXCLUDE_COLS |= {"lat_cell", "lon_cell"}


def load_data():
    df = pd.read_parquet(FIRE_PANEL).dropna(subset=["T2M"])
    if FEATURE_SET == "geo_season":
        features = ["lat_cell", "lon_cell", "doy_sin", "doy_cos"]
    else:
        features = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, features


def time_split(df: pd.DataFrame):
    val_start = pd.Timestamp(f"{VAL_YEAR}-01-01")
    test_start = pd.Timestamp(f"{VAL_YEAR + 1}-01-01")
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    print(f"  train {len(train):,} | val {len(val):,} | test {len(test):,}")
    return train, val, test


def climatology_pr_auc(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """PR-AUC of predicting each cell's usual fire rate for the month."""
    clim = (train.assign(m=train.date.dt.month)
                 .groupby(["lat_cell", "lon_cell", "m"])["burned"].mean()
                 .rename("clim"))
    te = test.assign(m=test.date.dt.month).join(clim, on=["lat_cell", "lon_cell", "m"])
    return average_precision_score(te["burned"], te["clim"].fillna(0))


def risk_level_table(data: pd.DataFrame, p) -> pd.DataFrame:
    levels = pd.cut(p, bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True)
    return (data.assign(level=levels)
                .groupby("level", observed=True)["burned"]
                .agg(n="size", fire_rate="mean"))


def main():
    print("Loading panel ...")
    df, features = load_data()
    print(f"  {len(features)} features")

    train, val, test = time_split(df)
    print(f"  climatology PR-AUC: {climatology_pr_auc(train, test):.4f}")

    print("Training HistGradientBoostingClassifier ...")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=5,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False,   # its default validation split is random -> leaks
    )
    model.fit(train[features], train["burned"])

    cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    cal.fit(val[features], val["burned"])

    p_val = cal.predict_proba(val[features])[:, 1]
    proba = cal.predict_proba(test[features])[:, 1]
    print(f"  val PR-AUC: {average_precision_score(val['burned'], p_val):.4f}")

    # --- risk levels -------------------------------------------------------
    print("\nRisk levels (validation)")
    print(risk_level_table(val, p_val))
    print("\nRisk levels (test)")
    print(risk_level_table(test, proba))

    # --- calibration views -------------------------------------------------
    chk = test.assign(pred=proba)
    print("\nObserved vs predicted, per test year")
    print(chk.groupby(chk.date.dt.year)
             .agg(observed=("burned", "mean"), predicted=("pred", "mean")))

    deciles = pd.qcut(proba, 10, duplicates="drop")
    print("\nObserved vs predicted, per risk decile")
    print(chk.groupby(deciles, observed=True)
             .agg(n=("burned", "size"), predicted=("pred", "mean"),
                  observed=("burned", "mean")))

    top = proba >= pd.Series(proba).quantile(0.9)
    top_share = test.loc[top, "burned"].sum() / test["burned"].sum()

    # --- headline metrics --------------------------------------------------
    pr_auc = average_precision_score(test["burned"], proba)
    base_rate = test["burned"].mean()
    brier = brier_score_loss(test["burned"], proba)
    brier_skill = 1 - brier / (base_rate * (1 - base_rate))

    print("\nResults (test)")
    print(f"  PR-AUC            : {pr_auc:.4f}   (random guess = {base_rate:.4f})")
    print(f"  Brier score       : {brier:.4f}")
    print(f"  Brier skill       : {brier_skill:.4f}   (0 = no better than base rate)")
    print(f"  Fires in top 10%  : {top_share:.1%}")


if __name__ == "__main__":
    main()
