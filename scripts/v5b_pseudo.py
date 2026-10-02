# v5b — stacking + pseudo-labeling round on top of v5 base-model checkpoints
#
# Stage 1: logistic stack on base OOF probs -> test probs.
# Stage 2: take confident test predictions (p > HI or p < LO) as pseudo-labels,
#          refit every base model on train + pseudo rows, recompute OOF, restack.
#          Val folds never contain pseudo rows, so the OOF estimate stays honest
#          (mild optimism only via the full-data model that generated the labels).
# Writes submission_v5_pseudo.csv and reports both OOF accuracies.

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

DATA = Path(__import__("os").environ.get("SST_DATA_DIR", "."))
CKPT = DATA / "v5_ckpt"
SEED = 42
N_FOLDS, REPEATS = 5, 2
HI, LO = 0.92, 0.08
np.random.seed(SEED)

# rebuild the feature tables (same code path as v5_stack.py)
base = open(DATA / "v5_stack.py").read().split("# ---------------- OOF loop")[0]
exec(base)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int).values
n = len(train)

splits = []
for s in range(REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 + s)
    splits += list(skf.split(np.zeros(n), y))

model_names = ["catboost", "lgbm", "xgb", "histgb"]
oof = {k: np.load(CKPT / f"oof_{k}.npy") for k in model_names}
test_p = {k: np.load(CKPT / f"test_{k}.npy") for k in model_names}
acc = lambda p, t=0.5: accuracy_score(y, p > t)

oofM = np.column_stack([oof[k] for k in model_names])
stack1 = LogisticRegression(C=1.0, max_iter=1000).fit(oofM, y)
oof_stack1 = stack1.predict_proba(oofM)[:, 1]
testM = np.column_stack([test_p[k] for k in model_names])
test_stack1 = stack1.predict_proba(testM)[:, 1]
print(f"stage1 OOF stack: {acc(oof_stack1):.4f}", flush=True)

# ---------------- pseudo-labels ---------------------------------------------------
pseudo_idx = np.where((test_stack1 > HI) | (test_stack1 < LO))[0]
pseudo_y = (test_stack1[pseudo_idx] > 0.5).astype(int)
print(f"pseudo-labeling {len(pseudo_idx)}/{len(test_stack1)} test rows "
      f"({(pseudo_y == 1).mean():.2%} positive)", flush=True)

cb_tr, cb_te = cb_frame(fe_tr), cb_frame(fe_te)
cb_all = pd.concat([cb_tr, cb_te], ignore_index=True)
hist_cols = num_cols + cat_cols

def fit_predict(name, fit_rows, fit_y, predict_rows):
    """fit on fit_rows (train-row indices + n+pseudo), predict rows (train or test)."""
    if name == "catboost":
        from catboost import CatBoostClassifier, Pool
        XA = cb_all.iloc[fit_rows]
        cat_pos = [XA.columns.get_loc(c) for c in cat_cols]
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                               loss_function="Logloss", od_type="Iter", od_wait=100,
                               random_seed=SEED, verbose=0, allow_writing_files=False)
        m.fit(Pool(XA, fit_y, cat_features=cat_pos))
        Xp = cb_all.iloc[predict_rows]
        return m.predict_proba(Pool(Xp, cat_features=cat_pos))[:, 1]
    if name == "lgbm":
        import lightgbm as lgb
        M_all = np.vstack([M_tr, M_te])
        m = lgb.LGBMClassifier(objective="binary", learning_rate=0.046, num_leaves=63,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, n_estimators=1200, verbose=-1, random_state=SEED)
        m.fit(M_all[fit_rows], fit_y, categorical_feature=cat_idx)
        return m.predict_proba(M_all[predict_rows])[:, 1]
    if name == "xgb":
        import xgboost as xgb
        X_all = np.vstack([X_tr, X_te])
        params = dict(objective="binary:logistic", eta=0.047, max_depth=6, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, nthread=0,
                      eval_metric="logloss")
        m = xgb.train(params, xgb.DMatrix(X_all[fit_rows], label=fit_y), num_boost_round=1200)
        return m.predict(xgb.DMatrix(X_all[predict_rows]))
    if name == "histgb":
        H_all = pd.concat([fe_tr[hist_cols], fe_te[hist_cols]], ignore_index=True)
        for c in cat_cols:
            H_all[c] = H_all[c].fillna("Missing").astype("category")
        m = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05,
                                           categorical_features="from_dtype", random_state=SEED)
        m.fit(H_all.iloc[fit_rows], fit_y)
        return m.predict_proba(H_all.iloc[predict_rows])[:, 1]

# ---------------- stage 2: refit with pseudo rows, recompute OOF -------------------
pseudo_rows = n + pseudo_idx
oof2 = {}
for name in model_names:
    o = np.zeros(n)
    for i, (tr_i, va_i) in enumerate(splits):
        fit_rows = np.concatenate([tr_i, pseudo_rows])
        fit_y = np.concatenate([y[tr_i], pseudo_y])
        o[va_i] = fit_predict(name, fit_rows, fit_y, va_i)
        if (i + 1) % 5 == 0:
            print(f"{name} pseudo fold {i + 1}/10: {acc(o, 0.5):.4f}", flush=True)
    oof2[name] = o
    print(f"== OOF+pseudo {name}: {acc(o):.4f}", flush=True)

oofM2 = np.column_stack([oof2[k] for k in model_names])
stack2 = LogisticRegression(C=1.0, max_iter=1000).fit(oofM2, y)
oof_stack2 = stack2.predict_proba(oofM2)[:, 1]
print(f"stage2 OOF stack (with pseudo): {acc(oof_stack2):.4f}", flush=True)

# ---------------- final test predictions with pseudo-trained models ----------------
test2 = {k: np.zeros(len(test)) for k in model_names}
for s in range(REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 + s)
    for tr_i, _ in skf.split(np.zeros(n), y):
        fit_rows = np.concatenate([tr_i, pseudo_rows])
        fit_y = np.concatenate([y[tr_i], pseudo_y])
        for name in model_names:
            test2[name] += fit_predict(name, fit_rows, fit_y, n + np.arange(len(test))) / (N_FOLDS * REPEATS)
    print(f"seed {s} test preds done", flush=True)

testM2 = np.column_stack([test2[k] for k in model_names])
test_stack2 = stack2.predict_proba(testM2)[:, 1]

# choose the better-validated stack
if acc(oof_stack2) >= acc(oof_stack1):
    final_test, chosen = test_stack2, "stack+pseudo"
else:
    final_test, chosen = test_stack1, "stack only"
print(f"chosen: {chosen} (OOF {max(acc(oof_stack1), acc(oof_stack2)):.4f})", flush=True)

pd.DataFrame({"PassengerId": test["PassengerId"],
              "Transported": np.where(final_test > 0.5, "True", "False")}
             ).to_csv(DATA / "submission_v5_pseudo.csv", index=False)
np.save(CKPT / "oof_stack2.npy", oof_stack2)
np.save(CKPT / "test_stack2.npy", test_stack2)
print("Wrote submission_v5_pseudo.csv; DONE", flush=True)
