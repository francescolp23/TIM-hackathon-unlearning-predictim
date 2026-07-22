"""
Hackathon TIM x Sapienza - Machine Unlearning
Smorzamento selettivo via Fisher + riparazione non pesata sul retain.
"""

import time, glob, os, copy, pickle, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils import functions as uf
from utils.model import DynamicMLP

SEED = 42
FOLDER = './data/'
ID_COL = 'user_id'
TARGET_PREFIX = 'target__'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FISHER_SUBSAMPLE = 16000
SSD_THR = 1.0
SSD_ALPHA = 5.0
REPAIR_EPOCHS = 30
REPAIR_LR = 2e-3
BATCH_SIZE = 4096

np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. dati e split
csv_files = glob.glob(os.path.join(FOLDER, '*c000.csv'))
df_all = pd.concat((pd.read_csv(f, sep=';') for f in csv_files), ignore_index=True)

# occhio: il forget usa la virgola
forget_df = pd.read_csv(os.path.join(FOLDER, 'forget_data.csv'), sep=',')
forget_ids = set(forget_df[ID_COL])
assert forget_df[ID_COL].isin(df_all[ID_COL]).all()

retain_all = df_all[~df_all[ID_COL].isin(forget_ids)].reset_index(drop=True)
retain_tmp, test_df = train_test_split(retain_all, test_size=0.15, random_state=SEED)
retain_df, val_df   = train_test_split(retain_tmp, test_size=0.15, random_state=SEED)
assert len(set(val_df[ID_COL]) & forget_ids) == 0
assert len(set(test_df[ID_COL]) & forget_ids) == 0
print(f"forget {len(forget_df)} | retain {len(retain_df)} | val {len(val_df)} | test {len(test_df)}")

# 2. feature e modello
X_retain, y_retain, feature_cols, target_cols = uf.prepare_data(
    retain_df, id_col=ID_COL, target_prefix=TARGET_PREFIX)
imputer = SimpleImputer(strategy='median')
X_retain = imputer.fit_transform(X_retain).astype(np.float32)

def transform(df):
    X, y, _, _ = uf.prepare_data(df, id_col=ID_COL, target_prefix=TARGET_PREFIX)
    return imputer.transform(X).astype(np.float32), y.astype(np.float32)

X_forget, y_forget = transform(forget_df)
X_val, y_val = transform(val_df)
X_test, y_test = transform(test_df)

Xr = torch.tensor(X_retain).to(DEVICE)
yr = torch.tensor(y_retain.astype(np.float32)).to(DEVICE)
Xf = torch.tensor(X_forget).to(DEVICE)
yf = torch.tensor(y_forget).to(DEVICE)

payload = uf.load_pickle(Path(FOLDER) / 'model_artifact')
arch = payload['architecture']
model = DynamicMLP(input_dim=arch['input_dim'],
                   hidden_layers=arch['hidden_layers'],
                   num_outputs=arch['num_outputs']).to(DEVICE)
model.load_state_dict(payload['state_dict'])
model.eval()
print("modello caricato:", arch)

# 3. unlearning
crit_plain = nn.BCEWithLogitsLoss()

def fisher_diagonal(m, X, y, criterion, batch=2048):
    m = copy.deepcopy(m).to(DEVICE); m.eval()
    fisher = [torch.zeros_like(p) for p in m.parameters()]
    n = X.shape[0]
    for i in range(0, n, batch):
        m.zero_grad()
        criterion(m(X[i:i+batch]), y[i:i+batch]).backward()
        for f, p in zip(fisher, m.parameters()):
            if p.grad is not None:
                f += p.grad.detach()**2 * len(X[i:i+batch])
    return [f/n for f in fisher]

t0 = time.time()

sub = torch.randperm(Xr.shape[0], device=DEVICE)[:FISHER_SUBSAMPLE]
fisher_f = fisher_diagonal(model, Xf, yf, crit_plain)
fisher_r = fisher_diagonal(model, Xr[sub], yr[sub], crit_plain)

unlearned = copy.deepcopy(model).to(DEVICE)
with torch.no_grad():
    n_sel = 0
    for p, ff, fr in zip(unlearned.parameters(), fisher_f, fisher_r):
        ratio = ff / (fr + 1e-8)
        sel = ratio > SSD_THR
        damp = torch.clamp(SSD_ALPHA * fr / (ff + 1e-8), max=1.0)
        p.mul_(torch.where(sel, damp, torch.ones_like(p)))
        n_sel += int(sel.sum())
print(f"pesi smorzati: {n_sel}")

opt = torch.optim.Adam(unlearned.parameters(), lr=REPAIR_LR)
n = Xr.shape[0]
unlearned.train()
for ep in range(REPAIR_EPOCHS):
    perm = torch.randperm(n, device=DEVICE)
    run = 0.0
    for i in range(0, n, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        opt.zero_grad()
        l = crit_plain(unlearned(Xr[idx]), yr[idx])
        l.backward(); opt.step()
        run += l.item()*len(idx)
    print(f"riparazione ep {ep+1}: loss {run/n:.4f}")

execution_time = time.time() - t0
unlearned.eval()
print(f"unlearning completato in {execution_time:.1f}s")

# 4. verifica
def precision_at_k(y_true, y_prob, k=10):
    idx = np.argsort(y_prob)[::-1][:k]
    return np.mean(y_true[idx])

def bce_per_sample(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return np.mean(-(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)), axis=1)

Xv = torch.tensor(X_val).to(DEVICE)
Xte = torch.tensor(X_test).to(DEVICE)

p_val = unlearned.predict_proba(Xv)
p10 = np.mean([precision_at_k(y_val[i], p_val[i]) for i in range(len(y_val))])

lf = bce_per_sample(y_forget, unlearned.predict_proba(Xf))
lt = bce_per_sample(y_test, unlearned.predict_proba(Xte))
labels = np.concatenate([np.ones_like(lf), np.zeros_like(lt)])
mia = roc_auc_score(labels, np.concatenate([-lf, -lt]))
print(f"\np@10 val {p10:.4f} | mia auc {mia:.4f}")

# 5. submission
out = dict(payload)
out['state_dict'] = {k: v.cpu() for k, v in unlearned.state_dict().items()}
with open('model_artifact', 'wb') as f:
    pickle.dump(out, f)
with open('execution_time.txt', 'w') as f:
    f.write(str(math.ceil(execution_time)))
val_df[[ID_COL]].to_csv('validation_ids.csv', index=False)
print("salvati: model_artifact, execution_time.txt, validation_ids.csv")
