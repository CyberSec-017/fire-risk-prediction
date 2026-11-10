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
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


from config import FIRE_PANEL,H,ROOT

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
USE_COORDS = True
FEATURE_SET = "full"     # "full" or "geo_season"
VAL_YEAR = 2022         # calibration year; test = all years after it
LABEL = "target"
# Risk levels on the calibrated probability (tune on the validation table).
RISK_BINS = [0, 0.01, 0.05, 0.15, 1.0]
RISK_LABELS = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]

EXCLUDE_COLS = {"date", "burned", "n_detections", "total_frp", "target"}
if not USE_COORDS:
    EXCLUDE_COLS |= {"lat_cell", "lon_cell"}


def load_data():
    df = pd.read_parquet(FIRE_PANEL).dropna(subset=["T2M"])
    df = df.drop(columns=["burned", "n_detections", "total_frp"])
    if FEATURE_SET == "geo_season":
        features = ["lat_cell", "lon_cell", "doy_sin", "doy_cos"]
    else:
        features = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, features


def time_split(df: pd.DataFrame):
    val_start = pd.Timestamp(f"{VAL_YEAR}-01-01")
    test_start = pd.Timestamp(f"{VAL_YEAR + 1}-01-01")
    train = df[df["date"] < val_start - pd.Timedelta(days=H)]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start - pd.Timedelta(days=H))]
    test = df[df["date"] >= test_start]
    print(f"  train {len(train):,} | val {len(val):,} | test {len(test):,}")
    return train, val, test


def climatology_proba(train, test):
    clim = (train.assign(m=train.date.dt.month)
                 .groupby(["lat_cell", "lon_cell", "m"])[LABEL].mean()
                 .rename("clim"))
    te = test.assign(m=test.date.dt.month).join(clim, on=["lat_cell", "lon_cell", "m"])
    return te["clim"].fillna(0).to_numpy()


def risk_level_table(data: pd.DataFrame, p) -> pd.DataFrame:
    levels = pd.cut(p, bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True)
    return (data.assign(level=levels)
                .groupby("level", observed=True)[LABEL]
                .agg(n="size", fire_rate="mean"))

def calib_slope_intercept(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf).fit(z, y)
    return lr.coef_[0][0], lr.intercept_[0]

def ece(y, p, bins=10):
    g = (pd.DataFrame({"y": y, "p": p})
           .groupby(pd.qcut(p, bins, duplicates="drop"), observed=True)
           .agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean")))
    return (g.n * (g.y - g.p).abs()).sum() / g.n.sum()    


def main():
    print("Loading panel ...")
    df, features = load_data()
    print(f"  {len(features)} features")

    train, val, test = time_split(df)
    

    print("Training HistGradientBoostingClassifier ...")
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=5,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False,   # its default validation split is random -> leaks
        random_state=0
    )
    model.fit(train[features], train[LABEL])

    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    cal.fit(val[features], val[LABEL])

    p_val = cal.predict_proba(val[features])[:, 1]
    proba = cal.predict_proba(test[features])[:, 1]
    raw = model.predict_proba(test[features])[:, 1]
    print(f"  val PR-AUC: {average_precision_score(val[LABEL], p_val):.4f}")

    # --- risk levels -------------------------------------------------------
    print("\nRisk levels (validation)")
    print(risk_level_table(val, p_val))
    print("\nRisk levels (test)")
    print(risk_level_table(test, proba))

    # --- calibration views -------------------------------------------------
    chk = test.assign(pred=proba)
    print("\nObserved vs predicted, per test year")
    print(chk.groupby(chk.date.dt.year)
             .agg(observed=(LABEL, "mean"), predicted=("pred", "mean")))
    
    for y_, g in chk.groupby(chk.date.dt.year):
        s, i = calib_slope_intercept(g[LABEL].to_numpy(), g["pred"].to_numpy())
        print(y_, f"slope {s:.2f}  intercept {i:+.2f}  ECE {ece(g[LABEL].to_numpy(), g['pred'].to_numpy()):.4f}")

    deciles = pd.qcut(proba, 10, duplicates="drop")
    print("\nObserved vs predicted, per risk decile")
    print(chk.groupby(deciles, observed=True)
             .agg(n=(LABEL, "size"), predicted=("pred", "mean"),
                  observed=(LABEL, "mean")))

    top = raw >= np.quantile(raw, 0.9)
    top_share = test.loc[top, LABEL].sum() / test[LABEL].sum()

    # --- headline metrics --------------------------------------------------
    pr_auc = average_precision_score(test[LABEL], raw)
    base_rate = test[LABEL].mean()
    brier = brier_score_loss(test[LABEL], proba)
    brier_skill = 1 - brier / (base_rate * (1 - base_rate))

        
    p_clim = climatology_proba(train, test)
    clim_pr = average_precision_score(test[LABEL], p_clim)
    clim_brier = brier_score_loss(test[LABEL], p_clim)
    clim_skill = 1 - clim_brier / (base_rate * (1 - base_rate))
    cmp = test.assign(model=raw, clim=p_clim)
    top_c = p_clim >= np.quantile(p_clim, 0.9)
    print(f"  Fires in top 10% (climatology): {test.loc[top_c, LABEL].sum() / test[LABEL].sum():.1%}")

    for yr_, g in cmp.groupby(cmp.date.dt.year):
        print(yr_, "model", round(average_precision_score(g[LABEL], g["model"]), 4),
                 "| climatology", round(average_precision_score(g[LABEL], g["clim"]), 4))
        
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(climatology_proba(train, val), val[LABEL])
    p_clim_cal = iso.predict(p_clim)
    clim_cal_brier = brier_score_loss(test[LABEL], p_clim_cal)
    print(f"  Brier vs calibrated climatology: {1 - brier / clim_cal_brier:+.2%}")
    
    

     # ---- 2) bootstrap block (new) ----
    rng = np.random.default_rng(0)
    y = test[LABEL].to_numpy()
    wk = test["date"].dt.to_period("W").astype(str).to_numpy()
    weeks = np.unique(wk)
    where = {w: np.where(wk == w)[0] for w in weeks}
    d_ap, d_br = [], []
    for _ in range(300):
        ii = np.concatenate([where[w] for w in rng.choice(weeks, len(weeks))])
        d_ap.append(average_precision_score(y[ii], raw[ii]) - average_precision_score(y[ii], p_clim[ii]))
        d_br.append(brier_score_loss(y[ii], p_clim_cal[ii]) - brier_score_loss(y[ii], proba[ii]))
    print("PR-AUC gain, 95% CI:", np.percentile(d_ap, [2.5, 97.5]).round(4))
    print("Brier gain, 95% CI:", np.percentile(d_br, [2.5, 97.5]).round(5))
    
    print(f"  Climatology PR-AUC: {clim_pr:.4f}")
    print(f"  Climatology Brier skill: {clim_skill:.4f}")
    print("\nResults (test)")
    print(f"  PR-AUC            : {pr_auc:.4f}   (random guess = {base_rate:.4f})")
    print(f"  Brier score       : {brier:.4f}")
    print(f"  Brier skill       : {brier_skill:.4f}   (0 = no better than base rate)")
    print(f"  Fires in top 10%  : {top_share:.1%}")
    
    model_path = ROOT / "models" / "fire_model.joblib"
    model_path.parent.mkdir(exist_ok=True)
    if FEATURE_SET == "full" and USE_COORDS:
        model_path = ROOT / "models" / "fire_model.joblib"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump({"model": cal, "features": features, "horizon_days": H,
                     "risk_bins": RISK_BINS, "risk_labels": RISK_LABELS}, model_path)


if __name__ == "__main__":
    main()
