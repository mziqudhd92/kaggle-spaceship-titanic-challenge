# v6a — random-search HPO for lgbm / xgb / histgb (catboost already tuned).
# Quick 3-fold eval per config; best params saved to v5_ckpt/.

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

DATA = Path(__import__("os").environ.get("SST_DATA_DIR", "."))
CKPT = DATA / "v5_ckpt"
SEED = 42
np.random.seed(SEED)

base = open(DATA / "v5_stack.py").read().split("# ---------------- OOF loop")[0]
exec(base)

train = pd.read_csv(DATA / "train.csv")
y = train["Transported"].astype(int).values
n = len(train)

skf = StratifiedKFold(3, shuffle=True, random_state=7)
fold_idx = list(skf.split(np.zeros(n), y))
acc = lambda p, t=0.5: accuracy_score(y, p > t)

def score_lgbm(params):
    import lightgbm as lgb
    a = []
    for tr_i, va_i in fold_idx:
        m = lgb.LGBMClassifier(**params, random_state=SEED, verbose=-1)
        m.fit(M_tr[tr_i], y[tr_i], categorical_feature=cat_idx,
              eval_set=[(M_tr[va_i], y[va_i])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        a.append(accuracy_score(y[va_i], m.predict_proba(M_tr[va_i])[:, 1] > 0.5))
    return np.mean(a)

def score_xgb(params):
    import xgboost as xgb
    a = []
    for tr_i, va_i in fold_idx:
        dtr = xgb.DMatrix(X_tr[tr_i], label=y[tr_i])
        dva = xgb.DMatrix(X_tr[va_i], label=y[va_i])
        m = xgb.train(params, dtr, num_boost_round=3000, early_stopping_rounds=80,
                      evals=[(dva, "val")], verbose_eval=False)
        a.append(accuracy_score(y[va_i], m.predict(dva, iteration_range=(0, m.best_iteration + 1)) > 0.5))
    return np.mean(a)

def score_histgb(params):
    XA = fe_tr[num_cols + cat_cols].copy()
    for c in cat_cols:
        XA[c] = XA[c].fillna("Missing").astype("category")
    a = []
    for tr_i, va_i in fold_idx:
        m = HistGradientBoostingClassifier(**params, random_state=SEED)
        m.fit(XA.iloc[tr_i], y[tr_i])
        a.append(accuracy_score(y[va_i], m.predict_proba(XA.iloc[va_i])[:, 1] > 0.5))
    return np.mean(a)

# ---- lightgbm search ----
lgb_best = (0, None)
for i in range(20):
    p = dict(objective="binary",
             learning_rate=float(10 ** np.random.uniform(np.log10(0.02), np.log10(0.1))),
             num_leaves=int(np.random.choice([15, 31, 63, 127])),
             min_child_samples=int(np.random.choice([5, 10, 20, 40])),
             subsample=float(np.random.uniform(0.6, 1.0)),
             subsample_freq=1,
             colsample_bytree=float(np.random.uniform(0.5, 1.0)),
             reg_lambda=float(10 ** np.random.uniform(-1, 1)),
             n_estimators=3000)
    s = score_lgbm(p)
    if s > lgb_best[0]:
        lgb_best = (s, p)
    print(f"lgbm {i + 1}/20: {s:.4f} (best {lgb_best[0]:.4f})", flush=True)
np.save(CKPT / "v6_lgbm_best.npy", lgb_best[1], allow_pickle=True)
print("LGBM BEST:", lgb_best[0], lgb_best[1], flush=True)

# ---- xgboost search ----
xgb_best = (0, None)
for i in range(20):
    p = dict(objective="binary:logistic", nthread=0, eval_metric="logloss",
             eta=float(10 ** np.random.uniform(np.log10(0.02), np.log10(0.1))),
             max_depth=int(np.random.choice([3, 4, 5, 6, 7, 8])),
             min_child_weight=int(np.random.choice([1, 2, 5, 10])),
             subsample=float(np.random.uniform(0.6, 1.0)),
             colsample_bytree=float(np.random.uniform(0.5, 1.0)),
             reg_lambda=float(10 ** np.random.uniform(-1, 1)))
    s = score_xgb(p)
    if s > xgb_best[0]:
        xgb_best = (s, p)
    print(f"xgb {i + 1}/20: {s:.4f} (best {xgb_best[0]:.4f})", flush=True)
np.save(CKPT / "v6_xgb_best.npy", xgb_best[1], allow_pickle=True)
print("XGB BEST:", xgb_best[0], xgb_best[1], flush=True)

# ---- histgb search ----
hb_best = (0, None)
for i in range(12):
    p = dict(max_iter=800,
             learning_rate=float(10 ** np.random.uniform(np.log10(0.02), np.log10(0.1))),
             max_leaf_nodes=int(np.random.choice([15, 31, 63, 127])),
             min_samples_leaf=int(np.random.choice([5, 10, 20, 40])),
             l2_regularization=float(10 ** np.random.uniform(-2, 1)),
             categorical_features="from_dtype")
    s = score_histgb(p)
    if s > hb_best[0]:
        hb_best = (s, p)
    print(f"histgb {i + 1}/12: {s:.4f} (best {hb_best[0]:.4f})", flush=True)
np.save(CKPT / "v6_histgb_best.npy", hb_best[1], allow_pickle=True)
print("HISTGB BEST:", hb_best[0], hb_best[1], flush=True)
print("ALL SEARCHES DONE", flush=True)
