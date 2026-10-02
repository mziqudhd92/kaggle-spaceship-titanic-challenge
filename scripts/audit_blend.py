# Pre-submission audit + no-regression blend (see README "Honest notes").
# Usage: SST_DATA_DIR=. python scripts/audit_blend.py
# Compares a candidate stack OOF against the incumbent on the same folds;
# if the margin is marginal (< +0.2pp or < 7/10 fold wins), emits a safety
# blend of candidate + incumbent test probabilities instead of the pure model.
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

DATA = Path(os.environ.get("SST_DATA_DIR", "."))
CKPT = DATA / "v5_ckpt"

test = pd.read_csv(DATA / "test.csv")
train = pd.read_csv(DATA / "train.csv")
y = train["Transported"].astype(int).values
n = len(y)

cand_test = np.load(CKPT / "v6_best_test.npy")
cand_oof = np.load(CKPT / "v6_best_oof_stack.npy")
inc_test = np.load(CKPT / "v5e_test_stack.npy")

print("sanity: rows", len(cand_test), "| finite:", bool(np.isfinite(cand_test).all()),
      "| positive rate:", round((cand_test > 0.5).mean(), 4))
print("submission ids aligned:", bool((pd.read_csv(DATA / 'submission_v6.csv')
      ['PassengerId'] == test['PassengerId']).all()))

folds = []
for s in range(2):
    skf = StratifiedKFold(5, shuffle=True, random_state=100 + s)
    folds += list(skf.split(np.zeros(n), y))
inc_oof = np.load(CKPT / "v6orig_best_oof_stack.npy") if (CKPT / "v6orig_best_oof_stack.npy").exists() else None
if inc_oof is not None:
    wins = sum(accuracy_score(y[va], (cand_oof > 0.5)[va]) >
               accuracy_score(y[va], (inc_oof > 0.5)[va]) for _, va in folds)
    print(f"paired fold wins vs incumbent: {wins}/10, "
          f"margin {accuracy_score(y, cand_oof > 0.5) - accuracy_score(y, inc_oof > 0.5):+.4f}")
    marginal = wins < 7 or (accuracy_score(y, cand_oof > 0.5) -
                            accuracy_score(y, inc_oof > 0.5)) < 0.002
else:
    marginal = True
    print("no incumbent OOF found -> treating as marginal")

if marginal:
    blend = (cand_test + inc_test) / 2
    pd.DataFrame({"PassengerId": test["PassengerId"],
                  "Transported": np.where(blend > 0.5, "True", "False")}
                 ).to_csv(DATA / "submission_blend.csv", index=False)
    print("MARGINAL -> wrote submission_blend.csv (safety blend with scored incumbent)")
else:
    pd.DataFrame({"PassengerId": test["PassengerId"],
                  "Transported": np.where(cand_test > 0.5, "True", "False")}
                 ).to_csv(DATA / "submission_pure.csv", index=False)
    print("CLEAR WIN -> wrote submission_pure.csv")
