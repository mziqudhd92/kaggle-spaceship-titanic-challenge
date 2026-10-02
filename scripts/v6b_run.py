# v6b — final pipeline: tuned boosters -> OOF -> iterative pseudo-labeling
# (threshold 0.90/0.10, labels frozen when a row joins) -> best-round submission.
# Stops when the stack OOF fails to improve for 2 consecutive rounds.

import os
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

DATA = Path(__import__("os").environ.get("SST_DATA_DIR", "."))
CKPT = DATA / "v5_ckpt"
SEED = 42
np.random.seed(SEED)

base = open(DATA / "v5_stack.py").read().split("# ---------------- OOF loop")[0]
exec(base)
hist_cols = num_cols + cat_cols

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int).values
n = len(train)
acc = lambda p, t=0.5: accuracy_score(y, p > t)

splits = []
for s in range(2):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + s)
    splits += list(skf.split(np.zeros(n), y))

cb_params = np.load(CKPT / "cb_best_params.npy", allow_pickle=True).item()
lgb_params = np.load(CKPT / "v6_lgbm_best.npy", allow_pickle=True).item()
xgb_params = np.load(CKPT / "v6_xgb_best.npy", allow_pickle=True).item()
hist_params = (np.load(CKPT / "v6_histgb_best.npy", allow_pickle=True).item()
               if (CKPT / "v6_histgb_best.npy").exists()
               else dict(max_iter=800, learning_rate=0.05, categorical_features="from_dtype"))
model_names = ["catboost", "lgbm", "xgb", "histgb"]

cb_all = pd.concat([cb_frame(fe_tr), cb_frame(fe_te)], ignore_index=True)
cb_cat_pos = [cb_all.columns.get_loc(c) for c in cat_cols]
H_all = pd.concat([fe_tr[hist_cols], fe_te[hist_cols]], ignore_index=True)
for c in cat_cols:
    H_all[c] = H_all[c].fillna("Missing").astype("category")
M_all = np.vstack([M_tr, M_te])
X_all = np.vstack([X_tr, X_te])

def fit_predict(name, fit_rows, fit_y, pred_rows):
    if name == "catboost":
        from catboost import CatBoostClassifier, Pool
        m = CatBoostClassifier(**cb_params, loss_function="Logloss", od_type="Iter",
                               od_wait=100, random_seed=SEED, verbose=0,
                               allow_writing_files=False)
        m.fit(Pool(cb_all.iloc[fit_rows], fit_y, cat_features=cb_cat_pos))
        return m.predict_proba(Pool(cb_all.iloc[pred_rows], cat_features=cb_cat_pos))[:, 1]
    if name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**lgb_params, random_state=SEED, verbose=-1)
        m.fit(M_all[fit_rows], fit_y, categorical_feature=cat_idx)
        return m.predict_proba(M_all[pred_rows])[:, 1]
    if name == "xgb":
        import xgboost as xgb
        m = xgb.train(xgb_params, xgb.DMatrix(X_all[fit_rows], label=fit_y),
                      num_boost_round=3000)
        return m.predict(xgb.DMatrix(X_all[pred_rows]))
    if name == "histgb":
        m = HistGradientBoostingClassifier(**hist_params, random_state=SEED)
        m.fit(H_all.iloc[fit_rows], fit_y)
        return m.predict_proba(H_all.iloc[pred_rows])[:, 1]

def stack_oof(pseudo_idx, pseudo_y):
    pseudo_rows = n + np.array(pseudo_idx, int) if len(pseudo_idx) else np.array([], int)
    p_y = np.array(pseudo_y, int)
    oof = {}
    for name in model_names:
        o = np.zeros(n)
        for i, (tr_i, va_i) in enumerate(splits):
            fit_rows = np.concatenate([tr_i, pseudo_rows]) if len(pseudo_idx) else tr_i
            fit_y = np.concatenate([y[tr_i], p_y]) if len(pseudo_idx) else y[tr_i]
            o[va_i] = fit_predict(name, fit_rows, fit_y, va_i)
        oof[name] = o
        print(f"  {name}: {acc(o):.4f}", flush=True)
    nn_oof = np.load(CKPT / "oof_nn.npy")
    cols = model_names + ["nn"]
    mats = [oof[k] for k in model_names] + [nn_oof]
    oofM = np.column_stack(mats)
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(oofM, y)
    oof_stack = stack.predict_proba(oofM)[:, 1]
    gid = fe_tr["group_id"].values
    gmean = pd.Series(oof_stack).groupby(gid).transform("mean").values
    best = (acc(oof_stack), 0.0, oof_stack)
    for alpha in [0.15, 0.2, 0.3]:
        s_ = alpha * gmean + (1 - alpha) * oof_stack
        a = acc(s_)
        if a > best[0]:
            best = (a, alpha, s_)
    return best[0], best[1], stack, oof_stack

def test_preds(stack, pseudo_idx, pseudo_y, n_seeds=3):
    pseudo_rows = n + np.array(pseudo_idx, int) if len(pseudo_idx) else np.array([], int)
    p_y = np.array(pseudo_y, int)
    tp = np.zeros(len(test))
    for s_ in range(n_seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=400 + s_)
        for tr_i, _ in skf.split(np.zeros(n), y):
            fit_rows = np.concatenate([tr_i, pseudo_rows]) if len(pseudo_idx) else tr_i
            fit_y = np.concatenate([y[tr_i], p_y]) if len(pseudo_idx) else y[tr_i]
            per = np.column_stack([
                fit_predict(name, fit_rows, fit_y, n + np.arange(len(test)))
                for name in model_names])
            per = np.column_stack([per, np.load(CKPT / "test_nn.npy")])
            tp += stack.predict_proba(per)[:, 1] / (5 * n_seeds)
    return tp

# ---------------- round 0: no pseudo ----------------
print("round 0 (no pseudo)", flush=True)
best_acc, best_alpha, stack0, _ = stack_oof([], [])
print(f"round 0 OOF: {best_acc:.4f} (alpha={best_alpha})", flush=True)

# seed the pseudo pool from the previous best test stack (v5e round-2 output)
prev_test = np.load(CKPT / "v5e_test_stack.npy")
pseudo_idx = list(np.where((prev_test > 0.90) | (prev_test < 0.10))[0])
pseudo_y = list((prev_test[pseudo_idx] > 0.5).astype(int))
print(f"initial pseudo pool: {len(pseudo_idx)} rows", flush=True)

best_test = None
no_improve = 0
for round_i in range(1, 6):
    t0 = time.time()
    a, alpha, stack, oof_stack = stack_oof(pseudo_idx, pseudo_y)
    print(f"round {round_i} OOF: {a:.4f} (alpha={alpha}, pool={len(pseudo_idx)}, "
          f"{time.time() - t0:.0f}s)", flush=True)
    if a > best_acc + 0.0002:
        best_acc, best_alpha = a, alpha
        no_improve = 0
        # refresh test predictions with this round's stack
        tp = test_preds(stack, pseudo_idx, pseudo_y)
        if alpha > 0:
            tg = fe_te["group_id"].values
            gmean_t = pd.Series(tp).groupby(tg).transform("mean").values
            tp = alpha * gmean_t + (1 - alpha) * tp
        best_test = tp
        np.save(CKPT / "v6_best_test.npy", tp)
        np.save(CKPT / "v6_best_oof_stack.npy", oof_stack)
        # grow pool from the refreshed test stack
        new = np.where((tp > 0.90) | (tp < 0.10))[0]
        known = set(pseudo_idx)
        added = 0
        for t in new:
            if int(t) not in known:
                pseudo_idx.append(int(t))
                pseudo_y.append(int(tp[t] > 0.5))
                added += 1
        print(f"  -> new best; added {added} pseudo rows (pool {len(pseudo_idx)})", flush=True)
    else:
        no_improve += 1
        if no_improve >= 2:
            print("converged — stopping", flush=True)
            break

print(f"BEST v6 OOF: {best_acc:.4f}", flush=True)
if best_test is not None and best_acc > 0.8124:  # beat v5e's OOF
    pd.DataFrame({"PassengerId": test["PassengerId"],
                  "Transported": np.where(best_test > 0.5, "True", "False")}
                 ).to_csv(DATA / "submission_v6.csv", index=False)
    print("Wrote submission_v6.csv", flush=True)
else:
    print("no improvement over v5e — nothing written", flush=True)
print("DONE", flush=True)
