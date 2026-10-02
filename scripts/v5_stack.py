# v5 — 2026-best-practice stack for Spaceship Titanic
#
# Models (5, deliberately diverse):
#   catboost  — native categorical splits, NaN handling
#   lgbm      — native categoricals (codes), NA routing
#   xgb       — one-hot + Missing level, NA routing
#   histgb    — sklearn Histogram GB, native categorical dtype + NaN
#   tabpfn    — tabular foundation model (Nature 2025), in-context learning
#
# Ensemble: logistic-regression stacker on OOF probabilities (vs. simple mean).
# Semi-supervised: one pseudo-labeling round on confident test rows.
# All OOF artifacts are checkpointed to disk.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

DATA = Path(__import__("os").environ.get("SST_DATA_DIR", "."))
CKPT = DATA / "v5_ckpt"
CKPT.mkdir(exist_ok=True)
SEED = 42
N_FOLDS = 5
REPEATS = 2
np.random.seed(SEED)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int).values
n = len(train)

# ---------------- feature engineering (combined, NA-preserving) ----------------
def prep(tr, te):
    df = pd.concat([tr.drop(columns=["Transported"]), te], ignore_index=True)
    for c in ["HomePlanet", "Destination", "Cabin", "Name"]:
        df[c] = df[c].replace("", np.nan)

    parts = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = parts[0]
    df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
    df["Side"] = parts[2]
    df["group_id"] = df["PassengerId"].str.split("_").str[0]

    df["GroupSize"] = df.groupby("group_id")["PassengerId"].transform("count")
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)
    df["surname"] = df["Name"].str.split(" ").str[1]
    df["FamilySize"] = df.groupby(["group_id", "surname"], dropna=False)["PassengerId"].transform("count")
    df.loc[df["surname"].isna(), "FamilySize"] = np.nan

    spend = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend].sum(axis=1, min_count=1)          # NaN if any part NaN
    df["LogSpend"] = np.log1p(df["TotalSpend"])
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(float)
    df.loc[df["TotalSpend"].isna(), "NoSpend"] = np.nan
    df["SpendMissing"] = df["TotalSpend"].isna().astype(int)
    for c in spend:
        df[c + "Zero"] = (df[c] == 0).astype(float)
        df.loc[df[c].isna(), c + "Zero"] = np.nan
    df["Leisure"] = df[["Spa", "VRDeck"]].sum(axis=1, min_count=1)
    df["Essentials"] = df[["RoomService", "FoodCourt"]].sum(axis=1, min_count=1)

    df["CryoSleep"] = df["CryoSleep"].astype("boolean")
    df.loc[(df["TotalSpend"] > 0) & df["CryoSleep"].isna(), "CryoSleep"] = False
    # group-mode fill
    def group_mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else np.nan
    for c in ["HomePlanet", "CryoSleep", "Deck", "Side", "Destination"]:
        df[c] = df.groupby("group_id")[c].transform(lambda s: s.fillna(group_mode(s)))

    deck_planet = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    df["HomePlanet"] = df["HomePlanet"].fillna(df["Deck"].map(deck_planet))

    df["CabinNumZ"] = df.groupby("Deck")["CabinNum"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.notna().any() else s)
    df["IsChild"] = np.where(df["Age"].isna(), np.nan, (df["Age"] < 13).astype(float))
    df["AgeBucket"] = pd.cut(df["Age"], [0, 12, 18, 30, 45, 60, 100], labels=False).astype(float)

    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    num_cols = ["Age", "CabinNum", "CabinNumZ", "GroupSize", "IsSolo", "FamilySize",
                "TotalSpend", "LogSpend", "NoSpend", "SpendMissing", "Leisure",
                "Essentials", "IsChild", "AgeBucket"] + [c + "Zero" for c in spend]
    for c in cat_cols:
        df[c] = df[c].astype("string")
    return df, cat_cols, num_cols

fe, cat_cols, num_cols = prep(train, test)
fe_tr, fe_te = fe.iloc[:n].reset_index(drop=True), fe.iloc[n:].reset_index(drop=True)

# ---------------- model-specific matrices --------------------------------------
lvls = {c: sorted(fe[c].dropna().unique().tolist()) for c in cat_cols}

def codes(frame):
    M = frame[num_cols].astype(float).values
    for c in cat_cols:
        M = np.column_stack([M, pd.Categorical(frame[c], categories=lvls[c]).codes.astype(float)])
    M[M < 0] = np.nan  # pandas -1 code -> NaN for lgbm
    return M

M_tr, M_te = codes(fe_tr), codes(fe_te)          # lgbm
cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

def onehot(frame):
    X = frame[num_cols].astype(float).values
    for c in cat_cols:
        v = frame[c].astype("string").fillna("Missing")
        d = pd.get_dummies(v, prefix=c).astype(float)
        for want in [f"{c}_{u}" for u in lvls[c] + (["Missing"] if lvls[c] else [])]:
            if want not in d.columns:
                d[want] = 0.0
        d = d.reindex(columns=[f"{c}_{u}" for u in lvls[c]] + [f"{c}_Missing"])
        X = np.column_stack([X, d.values])
    return X

X_tr, X_te = onehot(fe_tr), onehot(fe_te)        # xgb / histgb(fallback)

def tabpfn_frame(frame):
    out = frame[num_cols].astype(float).copy()
    for c in cat_cols:
        out[c] = pd.Categorical(frame[c].fillna("Missing"), categories=sorted(set(lvls[c]) | {"Missing"}))
    return out

# catboost frame: strings with None for NaN
def cb_frame(frame):
    out = frame[num_cols].astype(float).copy()
    for c in cat_cols:
        out[c] = frame[c].astype(object).astype(str).where(frame[c].notna(), "NA")
    return out

# ---------------- OOF loop with checkpoints -------------------------------------
splits = []
for s in range(REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 + s)
    splits += list(skf.split(np.zeros(n), y))

def oof_key(name):
    return CKPT / f"oof_{name}.npy"

def get_oof(name, fit_predict):
    path = oof_key(name)
    if path.exists():
        return np.load(path)
    oof = np.zeros(n)
    for i, (tr_i, va_i) in enumerate(splits):
        oof[va_i] = fit_predict(tr_i, va_i)
        print(f"{name} fold {i + 1}/{len(splits)}: {accuracy_score(y[va_i], oof[va_i] > 0.5):.4f}", flush=True)
    np.save(path, oof)
    return oof

def run_catboost(tr_i, va_i):
    from catboost import CatBoostClassifier, Pool
    XA = cb_frame(fe_tr).iloc[tr_i]
    Xv = cb_frame(fe_tr).iloc[va_i]
    cat_pos = [XA.columns.get_loc(c) for c in cat_cols]
    tr_pool = Pool(XA, y[tr_i], cat_features=cat_pos)
    va_pool = Pool(Xv, cat_features=cat_pos)
    m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6,
                           l2_leaf_reg=3, loss_function="Logloss", od_type="Iter",
                           od_wait=100, random_seed=SEED, verbose=0, allow_writing_files=False)
    m.fit(tr_pool)
    return m.predict_proba(va_pool)[:, 1]

def run_lgbm(tr_i, va_i):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.046, num_leaves=63,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, n_estimators=3000, verbose=-1, random_state=SEED)
    m.fit(M_tr[tr_i], y[tr_i], categorical_feature=cat_idx,
          eval_set=[(M_tr[va_i], y[va_i])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(M_tr[va_i])[:, 1]

def run_xgb(tr_i, va_i):
    import xgboost as xgb
    params = dict(objective="binary:logistic", eta=0.047, max_depth=6,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, nthread=0, eval_metric="logloss")
    dtr = xgb.DMatrix(X_tr[tr_i], label=y[tr_i])
    dva = xgb.DMatrix(X_tr[va_i], label=y[va_i])
    m = xgb.train(params, dtr, num_boost_round=3000,
                  early_stopping_rounds=100, evals=[(dva, "val")], verbose_eval=False)
    return m.predict(dva, iteration_range=(0, m.best_iteration + 1))

def run_histgb(tr_i, va_i):
    XA = fe_tr.iloc[tr_i][num_cols + cat_cols].copy()
    Xv = fe_tr.iloc[va_i][num_cols + cat_cols].copy()
    for c in cat_cols:
        XA[c] = XA[c].fillna("Missing").astype("category")
        Xv[c] = Xv[c].fillna("Missing").astype("category")
    m = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                       categorical_features="from_dtype",
                                       random_state=SEED)
    m.fit(XA, y[tr_i])
    return m.predict_proba(Xv)[:, 1]

def run_tabpfn(tr_i, va_i):
    from tabpfn import TabPFNClassifier
    XA = tabpfn_frame(fe_tr).iloc[tr_i]
    Xv = tabpfn_frame(fe_tr).iloc[va_i]
    m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, random_state=SEED)
    m.fit(XA, y[tr_i])
    return m.predict_proba(Xv)[:, 1]

import os
ENABLE_TABPFN = os.environ.get("V5_TABPFN", "0") == "1"
models = {"catboost": run_catboost, "lgbm": run_lgbm, "xgb": run_xgb,
          "histgb": run_histgb}
if ENABLE_TABPFN:
    models["tabpfn"] = run_tabpfn

acc = lambda p, t=0.5: accuracy_score(y, p > t)
oof = {}
for name, fn in models.items():
    oof[name] = get_oof(name, fn)
    print(f"== OOF {name}: {acc(oof[name]):.4f}", flush=True)

# ---------------- stacking ------------------------------------------------------
oofM = np.column_stack([oof[k] for k in models])
stack = LogisticRegression(C=1.0, max_iter=1000).fit(oofM, y)
oof_stack = stack.predict_proba(oofM)[:, 1]
oof_mean = oofM.mean(axis=1)
print(f"== OOF stack: {acc(oof_stack):.4f} | mean: {acc(oof_mean):.4f}", flush=True)
np.save(CKPT / "oof_stack.npy", oof_stack)

# ---------------- test-set predictions (fold-model averaging) --------------------
test_probs = {}

# final test prediction: average over the two CV seeds' models per type
def test_pred_catboost(tr_i):
    from catboost import CatBoostClassifier, Pool
    XA = cb_frame(fe_tr).iloc[tr_i]
    cat_pos = [XA.columns.get_loc(c) for c in cat_cols]
    pool = Pool(XA, y[tr_i], cat_features=cat_pos)
    m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                           loss_function="Logloss", od_type="Iter", od_wait=100,
                           random_seed=SEED, verbose=0, allow_writing_files=False)
    m.fit(pool)
    return m.predict_proba(Pool(cb_frame(fe_te), cat_features=cat_pos))[:, 1]

def test_pred_lgbm(tr_i):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.046, num_leaves=63,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, n_estimators=1200, verbose=-1, random_state=SEED)
    m.fit(M_tr[tr_i], y[tr_i], categorical_feature=cat_idx)
    return m.predict_proba(M_te)[:, 1]

def test_pred_xgb(tr_i):
    import xgboost as xgb
    params = dict(objective="binary:logistic", eta=0.047, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, nthread=0,
                  eval_metric="logloss")
    m = xgb.train(params, xgb.DMatrix(X_tr[tr_i], label=y[tr_i]), num_boost_round=1200)
    return m.predict(xgb.DMatrix(X_te))

def test_pred_histgb(tr_i):
    XA = fe_tr.iloc[tr_i][num_cols + cat_cols].copy()
    for c in cat_cols:
        XA[c] = XA[c].fillna("Missing").astype("category")
    m = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05,
                                       categorical_features="from_dtype", random_state=SEED)
    m.fit(XA, y[tr_i])
    Xv = fe_te[num_cols + cat_cols].copy()
    for c in cat_cols:
        Xv[c] = Xv[c].fillna("Missing").astype("category")
    return m.predict_proba(Xv)[:, 1]

def test_pred_tabpfn(tr_i):
    from tabpfn import TabPFNClassifier
    m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, random_state=SEED)
    m.fit(tabpfn_frame(fe_tr).iloc[tr_i], y[tr_i])
    return m.predict_proba(tabpfn_frame(fe_te))[:, 1]

test_fns = {"catboost": test_pred_catboost, "lgbm": test_pred_lgbm, "xgb": test_pred_xgb,
            "histgb": test_pred_histgb}
if ENABLE_TABPFN:
    test_fns["tabpfn"] = test_pred_tabpfn
for name, fn in test_fns.items():
    tp_path = CKPT / f"test_{name}.npy"
    if tp_path.exists():
        test_probs[name] = np.load(tp_path)
        continue
    p = np.zeros(len(fe_te))
    for s in range(REPEATS):  # average both CV repeats' fold models
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=100 + s)
        for tr_i, _ in skf.split(np.zeros(n), y):
            p += fn(tr_i) / (N_FOLDS * REPEATS)
    test_probs[name] = p
    np.save(tp_path, p)
    print(f"test preds done: {name}", flush=True)

testM = np.column_stack([test_probs[k] for k in models])
test_stack = stack.predict_proba(testM)[:, 1]

sub = pd.DataFrame({"PassengerId": test["PassengerId"],
                    "Transported": np.where(test_stack > 0.5, "True", "False")})
sub.to_csv(DATA / "submission_v5.csv", index=False)
print("Wrote submission_v5.csv", flush=True)
np.save(CKPT / "test_probs_matrix.npy", testM)
print("DONE", flush=True)
