import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss
from train_baseline import load_data, time_split, LABEL, EXCLUDE_COLS


df, _ = load_data()
features = [c for c in df.columns if c not in EXCLUDE_COLS | {"lat_cell", "lon_cell"}]
df["block"] = ((df.lat_cell >= 34.75).astype(int) * 3
               + pd.cut(df.lon_cell, [-3, 1.6667, 5.3333, 10], labels=False))
train, val, test = time_split(df)

rows = []
for b in sorted(df["block"].unique()):
    tr, va, te = train[train.block != b], val[val.block != b], test[test.block == b]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(tr[features], tr[LABEL])
    cal = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic").fit(va[features], va[LABEL])
    trva = pd.concat([tr, va])
    seas = trva.groupby(trva.date.dt.month)[LABEL].mean() 
    p_seas = te.date.dt.month.map(seas).to_numpy()
    p = cal.predict_proba(te[features])[:, 1]
    base = te[LABEL].mean()
    rows.append(dict(block=b, cells=te[["lat_cell", "lon_cell"]].drop_duplicates().shape[0],
        base=base,
        ap_model=average_precision_score(te[LABEL], m.predict_proba(te[features])[:, 1]),
        ap_season=average_precision_score(te[LABEL], p_seas),
        bss_model=1 - brier_score_loss(te[LABEL], p) / (base * (1 - base)),
        bss_season=1 - brier_score_loss(te[LABEL], p_seas) / (base * (1 - base))))
print(pd.DataFrame(rows).round(4))