# v6c — tabular neural network (PyTorch MLP with entity embeddings).
# Produces oof_nn.npy and test_nn.npy for the stack.
#
# Design (2026 tabular-NN best practice):
#   * categorical features -> learned entity embeddings (dim = min(8, ceil(card/2)))
#   * numeric features -> median-imputed + standardized, with explicit missing flags
#   * LayerNorm backbone, SiLU, dropout, AdamW, early stopping on the val fold

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

DATA = Path(__import__("os").environ.get("SST_DATA_DIR", "."))
CKPT = DATA / "v5_ckpt"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

base = open(DATA / "v5_stack.py").read().split("# ---------------- OOF loop")[0]
exec(base)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train["Transported"].astype(int).values
n = len(train)

# NN input frames: numeric median-imputed + missing flags; cats as integer codes
num_with_nan = [c for c in num_cols if fe[c].isna().any()]
def nn_frame(frame, medians=None):
    X = frame[num_cols].astype(float).copy()
    if medians is None:
        medians = X.median()
    X = X.fillna(medians)
    mu, sd = X.mean(), X.std().replace(0, 1)
    X = (X - mu) / sd
    miss = np.column_stack([frame[c].isna().astype(float).values for c in num_with_nan]) \
        if num_with_nan else np.zeros((len(frame), 0))
    cats = np.column_stack([
        pd.Categorical(frame[c], categories=lvls[c]).codes.astype(np.int64)
        for c in cat_cols])
    return X.values.astype(np.float32), cats, miss.astype(np.float32), medians

Xn_tr, C_tr, Mi_tr, med = nn_frame(fe_tr)
Xn_te, C_te, Mi_te, _ = nn_frame(fe_te, med)
# mark unseen codes as "missing" (last index)
card = [len(lvls[c]) for c in cat_cols]
for j in range(len(cat_cols)):
    C_te[C_te[:, j] < 0, j] = card[j]
card = [c + 1 for c in card]
C_tr[C_tr < 0] = 0  # train has no unseen codes; placeholder

class TabNN(nn.Module):
    def __init__(self, n_num, n_miss, cards, hidden=(256, 128), p=0.2):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, min(8, int(np.ceil(c / 2)))) for c in cards])
        emb_dim = sum(e.embedding_dim for e in self.emb)
        d_in = n_num + n_miss + emb_dim
        layers = [nn.LayerNorm(d_in), nn.Linear(d_in, hidden[0]), nn.SiLU(), nn.Dropout(p)]
        layers += [nn.Linear(hidden[0], hidden[1]), nn.SiLU(), nn.Dropout(p / 2),
                   nn.Linear(hidden[1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xnum, xmiss, xcat):
        e = [emb(xcat[:, j]) for j, emb in enumerate(self.emb)]
        return self.net(torch.cat([xnum, xmiss] + e, dim=1)).squeeze(-1)

def run_fold(tr_i, va_i):
    torch.manual_seed(SEED + len(tr_i))
    model = TabNN(Xn_tr.shape[1], Mi_tr.shape[1], card)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xt, Ct, Mt = torch.tensor(Xn_tr[tr_i]), torch.tensor(C_tr[tr_i]), torch.tensor(Mi_tr[tr_i])
    yt = torch.tensor(y[tr_i], dtype=torch.float32)
    Xv, Cv, Mv = torch.tensor(Xn_tr[va_i]), torch.tensor(C_tr[va_i]), torch.tensor(Mi_tr[va_i])
    yv = torch.tensor(y[va_i], dtype=torch.float32)
    best_acc, best_state, patience = 0.0, None, 0
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(200):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(Xt), 256):
            idx = perm[b:b + 256]
            opt.zero_grad()
            out = model(Xt[idx], Mt[idx], Ct[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = (torch.sigmoid(model(Xv, Mv, Cv)) > 0.5).float()
        a = (pv == yv).float().mean().item()
        if a > best_acc:
            best_acc, patience = a, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        return torch.sigmoid(model(Xv, Mv, Cv)).numpy(), model, best_acc

splits = []
for s in range(2):
    skf = StratifiedKFold(5, shuffle=True, random_state=100 + s)
    splits += list(skf.split(np.zeros(n), y))

oof = np.zeros(n)
test_p = np.zeros(len(test))
models = []
for i, (tr_i, va_i) in enumerate(splits):
    p, model, ba = run_fold(tr_i, va_i)
    oof[va_i] = p
    models.append(model)
    print(f"nn fold {i + 1}/10: {accuracy_score(y[va_i], p > 0.5):.4f} (best in-fold {ba:.4f})", flush=True)

print(f"== OOF nn: {accuracy_score(y, oof > 0.5):.4f}", flush=True)
np.save(CKPT / "oof_nn.npy", oof)

# test predictions: average the 10 fold models
Xt_te = torch.tensor(Xn_te); Ct_te = torch.tensor(C_te); Mt_te = torch.tensor(Mi_te)
with torch.no_grad():
    for m in models:
        m.eval()
        test_p += torch.sigmoid(m(Xt_te, Mt_te, Ct_te)).numpy() / len(models)
np.save(CKPT / "test_nn.npy", test_p)
print("saved oof_nn.npy / test_nn.npy — DONE", flush=True)
