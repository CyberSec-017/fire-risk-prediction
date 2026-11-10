"""
Rolling-origin (expanding-window) evaluation.

For each test year Y the model is trained on the years before Y-1,
calibrated on year Y-1, and tested on year Y. Every fold trains on more
data than it tests on and never sees the future. Both calibration methods
are fitted in every fold, so they can be compared on average calibration
quality without tuning against one fixed test period.

Needs train_baseline.py to have USE_COORDS = True and FEATURE_SET = "full".

Usage (from the src folder):
    python evaluate_rolling.py
"""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from config import H
from train_baseline import (EXCLUDE_COLS, LABEL, load_data, climatology_proba,
                            calib_slope_intercept, ece)

TEST_YEARS = [2021, 2022, 2023, 2024]
GAP = pd.Timedelta(days=H)   # purge: each target looks H days ahead


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=False, random_state=0)


df, _ = load_data()
features = [c for c in df.columns if c not in EXCLUDE_COLS]

rank_rows, prob_rows = [], []
for Y in TEST_YEARS:
    cal_start = pd.Timestamp(f"{Y - 1}-01-01")
    test_start = pd.Timestamp(f"{Y}-01-01")
    test_end = pd.Timestamp(f"{Y + 1}-01-01")

    train = df[df["date"] < cal_start - GAP]
    cal = df[(df["date"] >= cal_start) & (df["date"] < test_start - GAP)]
    test = df[(df["date"] >= test_start) & (df["date"] < test_end)]
    print(f"fold {Y}: train {len(train):,} | calibrate {len(cal):,} | test {len(test):,}")

    model = make_model().fit(train[features], train[LABEL])
    raw = model.predict_proba(test[features])[:, 1]
    y = test[LABEL].to_numpy()
    base = y.mean()
    ref = base * (1 - base)

    # baseline: per-cell monthly climatology, calibrated on the same year
    p_clim = climatology_proba(train, test)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(climatology_proba(train, cal), cal[LABEL])
    brier_clim = brier_score_loss(y, iso.predict(p_clim))

    ap_m = average_precision_score(y, raw)
    ap_c = average_precision_score(y, p_clim)
    rank_rows.append(dict(test_year=Y, n_train=len(train), n_test=len(test),
                          base=base, ap_model=ap_m, ap_clim=ap_c, gain=ap_m - ap_c))

    for method in ("sigmoid", "isotonic"):
        cm = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        cm.fit(cal[features], cal[LABEL])
        p = cm.predict_proba(test[features])[:, 1]
        b = brier_score_loss(y, p)
        slope, icpt = calib_slope_intercept(y, p)
        prob_rows.append(dict(test_year=Y, method=method, mean_pred=p.mean(),
                              base=base, bss=1 - b / ref,
                              vs_clim_cal=1 - b / brier_clim,
                              slope=slope, intercept=icpt, ece=ece(y, p)))

print("\nRanking: PR-AUC of raw model scores vs per-cell monthly climatology")
print(pd.DataFrame(rank_rows).round(4).to_string(index=False))

pt = pd.DataFrame(prob_rows)
print("\nProbabilities, per fold")
print(pt.round(4).to_string(index=False))
print("\nProbabilities, mean over folds")
print(pt.groupby("method")[["bss", "vs_clim_cal", "slope", "intercept", "ece"]]
        .mean().round(4))