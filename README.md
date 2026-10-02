# Spaceship Titanic — GBDT stack with native categoricals + pseudo-labeling

Solution for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition: predict which passengers were transported to an alternate dimension.

**Result: 0.80009 public leaderboard accuracy** (cross-validated OOF 0.8124), up from
0.7438 for a vanilla Random Forest baseline.

| Stage | OOF accuracy |
|---|---|
| Random Forest baseline (`scripts/spaceship_baseline.py`) | 0.7438 |
| XGBoost + engineered features | 0.7580 |
| Tuned LightGBM/XGBoost (NA-native, R) | 0.7606 |
| 4 GBDTs with native categorical splits, stacked | 0.8054 |
| + CatBoost tuning, RF/ExtraTrees, group smoothing | 0.8109 |
| + 2 rounds of pseudo-labeling (final) | **0.8124** |

## What moved the needle

1. **Native categorical handling** (+4 pts) — CatBoost/LightGBM split categorical
   columns directly instead of one-hot encoding; R's tree ecosystem lacks this,
   which is why the R baseline plateaued at 0.75.
2. **Missingness as signal** — `fread`/`read_csv` give empty strings, not `NaN`;
   normalizing `"" -> NA` before group-based imputation was worth several tenths.
3. **Feature engineering** — CryoSleep inference (cryo passengers never spend),
   group size from `PassengerId`, family size from surnames, log-spend,
   deck/side from `Cabin`, per-amenity zero flags.
4. **Pseudo-labeling** (+0.7 pts) — train on confidently-predicted test rows
   (p > 0.92 or p < 0.08), refit, re-stack, iterate to convergence.
5. **Stacking** — logistic regression on out-of-fold probabilities from four
   deliberately diverse models beats any single model and every weight blend.

## Repository layout

```
├── notebooks/
│   └── kaggle_best_model.ipynb   # best pipeline, Kaggle-ready (runs ~2 h on CPU)
├── scripts/
│   ├── spaceship_baseline.py     # Python RF baseline (0.744 CV)
│   ├── spaceship_baseline.R      # R ranger baseline (0.751 CV)
│   ├── v5_stack.py               # feature engineering + 4 GBDTs + OOF stack
│   ├── v5b_pseudo.py             # pseudo-labeling round (stage 2)
│   ├── v6a_tune.py               # random-search HPO for lgbm/xgb/histgb
│   ├── v6c_nn.py                 # PyTorch tabular MLP with entity embeddings
│   └── v6b_run.py                # final: tuned params + iterative pseudo-labeling
├── requirements.txt
└── LICENSE (MIT)
```

## Quickstart

```bash
# 1. environment (Python 3.10+)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. competition data (requires a Kaggle account; join the competition once)
kaggle competitions download -c spaceship-titanic && unzip spaceship-titanic.zip

# 3. baseline
.venv/bin/python scripts/spaceship_baseline.py        # -> submission.csv

# 4. full pipeline (set SST_DATA_DIR if the CSVs live elsewhere)
SST_DATA_DIR=. .venv/bin/python scripts/v5_stack.py   # OOF + test preds per model
.venv/bin/python scripts/v6a_tune.py                  # hyperparameter search (~1 h)
.venv/bin/python scripts/v6c_nn.py                    # tabular NN (~10 min CPU)
.venv/bin/python scripts/v5b_pseudo.py                # stack + pseudo round
.venv/bin/python scripts/v6b_run.py                   # final iterative pipeline
```

Each script checkpoints its out-of-fold/test probabilities into `v5_ckpt/`, so
stages can be re-run and inspected independently.

The R baseline needs R ≥ 4.2 with `data.table` and `ranger`.

## Kaggle notebooks

- [Spaceship Titanic Best Model](https://www.kaggle.com/code/moranzavdi/spaceship-titanic-best-model-0-800-lb)
  — the full pipeline as a runnable notebook (writes `submission.csv`).
- [Spaceship Titanic R Model Dev](https://www.kaggle.com/code/moranzavdi/spaceship-titanic-r-model-dev)
  — interactive R notebook for feature/model experimentation.

## Honest notes

- The public leaderboard has entries above 0.89 that are unreachable from the
  provided features; they match the documented test-label recovery/memorization
  pattern (see [Hollmann et al. 2025](https://arxiv.org/html/2404.06209v1) for the
  phenomenon on this dataset). This repo targets the legitimate ceiling (~0.81).
- TabPFN v2 ([Nature 2025](https://www.nature.com/articles/s41586-024-08328-6))
  was evaluated but segfaults on this machine's CPU; on GPU it is worth adding
  as a fifth stack member.

## License

[MIT](LICENSE) — use it, fork it, submit with it.

## Addendum — round 2 results (TabPFN + soft labels)

- TabPFN v2 on Kaggle GPU: 0.8059 OOF as a standalone member; adds +0.0004 to the
  stack at 15% blend weight (`notebooks/kaggle_tabpfn_gpu.ipynb`).
- Soft/confidence-weighted pseudo-labels (`scripts/v7_soft.py`): no gain (0.8126 vs
  0.8132) — hard labels converged.
- Error analysis found only 9.9% of multi-member groups have unanimous outcomes,
  invalidating family-consensus pseudo-labeling.
- Final submissions: safety blend 0.80219 (best), pure candidate 0.79939 — public-LB
  noise (~±0.3pp at 99% prediction agreement) exceeds all remaining OOF gains.
