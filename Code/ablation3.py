#!/usr/bin/env python3
"""
ablation3.py  --  Baseline model comparison on VNAT (VPN vs Non-VPN)

Trains four baseline classifiers on EXACTLY the same data as the PatchTST model:
    1. InceptionTime            (deep learning, time-series CNN)
    2. LSTM                     (deep learning, recurrent)
    3. GRU                      (deep learning, recurrent)
    4. Random Forest            (classical ML)

Data (shared with the PatchTST pipeline, nothing re-split):
  * train.csv / test.csv  -> 5-second windows, 1-second stride
  * the 3 newly added features (PTR, PSCR, Direction-Switch) are REMOVED for every baseline,
    so all baselines use the 36 original per-second features (180 columns)
  * StandardScaler fitted on TRAIN only, applied to test
  * deep models see a (5 seconds x 36 features) sequence; ML models see the 180 flat features

Training protocol for the three deep baselines (same for both):
  AdamW, lr 1e-3, wd 1e-4, batch 512, up to 50 epochs, cosine schedule, class-weighted
  cross-entropy, grad-clip 1.0, early stopping (patience 10), seed 42, best-macro-F1
  checkpoint on the eval split (same rule as the PatchTST pipeline).

PatchTST is NOT re-trained here.  To add your existing PatchTST result to the final
comparison table, fill in PATCHTST_RESULT below with your own measured numbers.

Outputs: baseline_comparison.csv (+ printed table) and one confusion-matrix PNG per baseline.

Usage:  python ablation3.py
"""

import gc
import os
import re
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────── CONFIG ────────────────────────
WINDOW     = 5
ALL_FEATURES = 39                      # per-second features in the CSVs
REMOVED_FEATURES = ("ptr", "pscr", "direction_switch")   # PTR, PSCR, Direction-Switch (the 3 new features)
N_FEATURES = ALL_FEATURES - len(REMOVED_FEATURES)         # 36 features used by every baseline

# Deep-baseline training config
BATCH_SIZE    = 512
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EARLY_STOP_PATIENCE = 5
SEED = 42

# InceptionTime (kernels adapted to the very short 5-step sequence)
INC_FILTERS     = 32
INC_BOTTLENECK  = 32
INC_KERNELS     = (1, 3, 5)
INC_DEPTH       = 6          # 6 inception modules, residual connection every 3

# LSTM / GRU
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.1

# ML baselines
RF_TREES = 100

# Optional: put YOUR measured PatchTST results here (use the PatchTST run that ALSO uses the
# 36 baseline features, i.e. the 'Baseline' row of ablation_study.py, for a fair comparison)
# Original note: put YOUR measured PatchTST test results here to include it in the table.
# Leave values as None to omit the row (nothing is invented by this script).
PATCHTST_RESULT = {"Accuracy": 0.9952, "Macro-F1": 0.9953, "ROC-AUC": 0.9998}

TRAIN_PATH = r"D:\sample_pcap_tor\CODE\take2\take2\train.csv"
TEST_PATH  = r"D:\sample_pcap_tor\CODE\take2\take2\test.csv"
OUT_DIR    = r"D:\sample_pcap_tor\CODE\take2\take2\ablation3"

EXCLUDE = {"label", "time_idx", "group_id", "file_id"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fail(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def _resolve(path):
    if os.path.exists(path):
        return path
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        alt = os.path.join(base, os.path.basename(path))
        if os.path.exists(alt):
            return alt
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────── MODELS ────────────────────────
class InceptionModule(nn.Module):
    def __init__(self, in_ch, nf, bottleneck, kernels):
        super().__init__()
        self.use_bottleneck = in_ch > 1
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if self.use_bottleneck else None
        b_ch = bottleneck if self.use_bottleneck else in_ch
        self.convs = nn.ModuleList(
            [nn.Conv1d(b_ch, nf, k, padding=k // 2, bias=False) for k in kernels])
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(kernels) + 1))
        self.out_ch = nf * (len(kernels) + 1)

    def forward(self, x):
        z = self.bottleneck(x) if self.use_bottleneck else x
        outs = [c(z) for c in self.convs] + [self.pool_conv(x)]
        return F.relu(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    """Fawaz et al. 2020 style: stacked inception modules + residual shortcut every 3 modules."""
    def __init__(self, in_ch, n_classes=2):
        super().__init__()
        self.modules_, self.shortcuts = nn.ModuleList(), nn.ModuleDict()
        ch = in_ch
        res_in = in_ch
        for d in range(INC_DEPTH):
            m = InceptionModule(ch, INC_FILTERS, INC_BOTTLENECK, INC_KERNELS)
            self.modules_.append(m)
            ch = m.out_ch
            if d % 3 == 2:
                self.shortcuts[str(d)] = nn.Sequential(
                    nn.Conv1d(res_in, ch, 1, bias=False), nn.BatchNorm1d(ch))
                res_in = ch
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):                # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        res = x
        for d, m in enumerate(self.modules_):
            x = m(x)
            if d % 3 == 2:
                x = F.relu(x + self.shortcuts[str(d)](res))
                res = x
        return self.fc(x.mean(dim=-1))   # global average pooling


class LSTMClassifier(nn.Module):
    def __init__(self, in_ch, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, LSTM_HIDDEN, num_layers=LSTM_LAYERS, batch_first=True,
                            dropout=LSTM_DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(LSTM_DROPOUT), nn.Linear(LSTM_HIDDEN // 2, n_classes))

    def forward(self, x):                # x: (B, T, C)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class GRUClassifier(nn.Module):
    def __init__(self, in_ch, n_classes=2):
        super().__init__()
        self.gru = nn.GRU(in_ch, LSTM_HIDDEN, num_layers=LSTM_LAYERS, batch_first=True,
                          dropout=LSTM_DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2), nn.ReLU(),
            nn.Dropout(LSTM_DROPOUT), nn.Linear(LSTM_HIDDEN // 2, n_classes))

    def forward(self, x):                # x: (B, T, C)
        _, h = self.gru(x)
        return self.head(h[-1])


# ──────────────────────── HELPERS ────────────────────────
def metrics(y_true, y_pred, y_prob):
    return dict(
        acc=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        auc=roc_auc_score(y_true, y_prob),
        cm=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


@torch.no_grad()
def predict_dl(model, X):
    model.eval()
    preds, probs = [], []
    for i in range(0, len(X), BATCH_SIZE):
        p = F.softmax(model(X[i:i + BATCH_SIZE]), dim=1)
        preds.append(torch.argmax(p, 1).cpu().numpy())
        probs.append(p[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def train_dl(name, model, Xtr, ytr, Xte, yte, class_weights):
    set_seed(SEED)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    g = torch.Generator()
    g.manual_seed(SEED)

    yte_np = yte.cpu().numpy()
    best_f1, best_state, no_improve, epochs_run = 0.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        loss_sum = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(b)
        scheduler.step()
        loss_sum /= len(Xtr)

        vp, _ = predict_dl(model, Xte)
        v_acc = accuracy_score(yte_np, vp)
        v_f1  = f1_score(yte_np, vp, average="macro", zero_division=0)
        print(f"  Epoch {epoch:03d} | Loss: {loss_sum:.6f} | Eval Acc: {v_acc:.5f} | Eval F1: {v_f1:.5f}")

        epochs_run = epoch
        no_improve = 0 if v_f1 > best_f1 else no_improve + 1
        if v_f1 >= best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if EARLY_STOP_PATIENCE is not None and no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    preds, probs = predict_dl(model, Xte)
    res = metrics(yte_np, preds, probs)
    res.update(params=n_params, time=train_time, epochs=epochs_run)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def save_confusion_matrix(cm, title, key):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["NON-VPN", "VPN"]); ax.set_yticklabels(["NON-VPN", "VPN"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {title}")
    thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"cm_{key}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def report(title, key, res):
    print(f"  -> Accuracy {res['acc']:.6f} | Macro-F1 {res['f1']:.6f} | ROC-AUC {res['auc']:.6f} "
          f"| train time {res['time']:.0f}s")
    print(f"  Confusion matrix ({title}):")
    print(f"                    Pred NON-VPN   Pred VPN")
    print(f"    Actual NON-VPN  {res['cm'][0][0]:<14} {res['cm'][0][1]}")
    print(f"    Actual VPN      {res['cm'][1][0]:<14} {res['cm'][1][1]}")
    print(f"  Saved -> {save_confusion_matrix(res['cm'], title, key)}\n")


# ──────────────────────── MAIN ────────────────────────
def main():
    global OUT_DIR
    train_path, test_path = _resolve(TRAIN_PATH), _resolve(TEST_PATH)
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
    except OSError:
        OUT_DIR = os.path.join(os.getcwd(), "ablation3_outputs")
        os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("BASELINE COMPARISON  (InceptionTime, LSTM, GRU, Random Forest)")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Train CSV  : {train_path}")
    print(f"Test CSV   : {test_path}")
    print(f"Output dir : {OUT_DIR}")
    for p, n in [(train_path, "train.csv"), (test_path, "test.csv")]:
        if not os.path.exists(p):
            fail(f"{n} not found at {p}")

    train_df = pd.read_csv(train_path).fillna(0)
    test_df  = pd.read_csv(test_path).fillna(0)
    for nm, d in [("train", train_df), ("test", test_df)]:
        d["label"] = d["label"].astype(int)
        if set(d["label"].unique()) != {0, 1}:
            fail(f"{nm} labels are {sorted(d['label'].unique())}, expected {{0, 1}}.")

    # Feature columns, kept in the CSV's feature-major order: <name>_t0 .. <name>_t4
    names = []
    for c in train_df.columns:
        if c in EXCLUDE:
            continue
        m = re.fullmatch(r"(.+)_t(\d+)", c)
        if m is None:
            fail(f"Unexpected column '{c}'.")
        if m.group(1) not in names:
            names.append(m.group(1))
    if len(names) != ALL_FEATURES:
        fail(f"Expected {ALL_FEATURES} per-second features in the CSV, found {len(names)}.")
    for r in REMOVED_FEATURES:
        if r not in names:
            fail(f"Feature '{r}' to be removed was not found in the CSV: {names}")
    names = [n for n in names if n not in REMOVED_FEATURES]
    print(f"Removed features -> {', '.join(REMOVED_FEATURES)}")
    print(f"Baseline feature count -> {len(names)}")
    if len(names) != N_FEATURES:
        fail(f"Expected {N_FEATURES} features after removal, got {len(names)}.")
    cols = [f"{n}_t{j}" for n in names for j in range(WINDOW)]
    missing = [c for c in cols if c not in train_df.columns or c not in test_df.columns]
    if missing:
        fail(f"Missing columns: {missing[:5]} ...")
    if len(cols) != N_FEATURES * WINDOW:
        fail("Column count mismatch.")

    ytr_np = train_df["label"].values.astype(np.int64)
    yte_np = test_df["label"].values.astype(np.int64)
    print(f"Train rows {len(train_df)} | Test rows {len(test_df)} | features {len(cols)} "
          f"({N_FEATURES} per second x {WINDOW} s)")
    print(f"Train VPN/NonVPN: {(ytr_np==1).sum()}/{(ytr_np==0).sum()} | "
          f"Test VPN/NonVPN: {(yte_np==1).sum()}/{(yte_np==0).sum()}")

    # Scaler fitted on TRAIN only
    scaler = StandardScaler()
    Xtr_flat = scaler.fit_transform(train_df[cols].values.astype(np.float32)).astype(np.float32)
    Xte_flat = scaler.transform(test_df[cols].values.astype(np.float32)).astype(np.float32)
    del train_df, test_df
    gc.collect()

    # (N, 36, 5) -> (N, 5, 36): each time step is one second of 36 features
    to_seq = lambda a: np.ascontiguousarray(a.reshape(-1, N_FEATURES, WINDOW).transpose(0, 2, 1))
    Xtr_seq = torch.tensor(to_seq(Xtr_flat), device=device)
    Xte_seq = torch.tensor(to_seq(Xte_flat), device=device)
    ytr_t = torch.tensor(ytr_np, device=device)
    yte_t = torch.tensor(yte_np, device=device)

    vpn_n, nonvpn_n = int((ytr_np == 1).sum()), int((ytr_np == 0).sum())
    tot = vpn_n + nonvpn_n
    class_weights = torch.tensor([tot / (2.0 * nonvpn_n), tot / (2.0 * vpn_n)],
                                 dtype=torch.float32, device=device)

    # Sanity checks
    print("\nSanity check")
    if tuple(Xtr_seq.shape[1:]) != (WINDOW, N_FEATURES):
        fail(f"Bad sequence shape {tuple(Xtr_seq.shape)}.")
    for nm, ctor in [("InceptionTime", InceptionTime), ("LSTM", LSTMClassifier), ("GRU", GRUClassifier)]:
        set_seed(SEED)
        m = ctor(N_FEATURES).to(device).eval()
        with torch.no_grad():
            out = m(Xtr_seq[:8])
        if tuple(out.shape) != (8, 2) or not torch.isfinite(out).all():
            fail(f"{nm}: bad forward-pass output {tuple(out.shape)}.")
        print(f"  {nm:<14} input={tuple(Xtr_seq[:8].shape)} logits={tuple(out.shape)} OK")
        del m
    print(f"  ML models      input=({len(Xtr_flat)}, {Xtr_flat.shape[1]}) OK")
    print("All sanity checks passed.\n")

    rows = []

    def add_row(model_name, kind, res, params):
        rows.append({
            "Model": model_name, "Type": kind, "Parameters": params,
            "Accuracy": round(res["acc"], 6), "Macro-F1": round(res["f1"], 6),
            "ROC-AUC": round(res["auc"], 6), "Train time (s)": round(res["time"], 1),
        })

    # 1/4 InceptionTime
    print("=" * 70); print("[1/4] InceptionTime"); print("=" * 70)
    res = train_dl("InceptionTime", InceptionTime(N_FEATURES), Xtr_seq, ytr_t, Xte_seq, yte_t, class_weights)
    report("InceptionTime", "inceptiontime", res)
    add_row("InceptionTime", "Deep (CNN)", res, f"{res['params']:,}")

    # 2/4 LSTM
    print("=" * 70); print("[2/4] LSTM"); print("=" * 70)
    res = train_dl("LSTM", LSTMClassifier(N_FEATURES), Xtr_seq, ytr_t, Xte_seq, yte_t, class_weights)
    report("LSTM", "lstm", res)
    add_row("LSTM", "Deep (RNN)", res, f"{res['params']:,}")

    # 3/4 GRU
    print("=" * 70); print("[3/4] GRU"); print("=" * 70)
    res = train_dl("GRU", GRUClassifier(N_FEATURES), Xtr_seq, ytr_t, Xte_seq, yte_t, class_weights)
    report("GRU", "gru", res)
    add_row("GRU", "Deep (RNN)", res, f"{res['params']:,}")

    del Xtr_seq, Xte_seq
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4/4 Random Forest
    print("=" * 70); print("[4/4] Random Forest"); print("=" * 70)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=-1, class_weight="balanced",
                                random_state=SEED)
    rf.fit(Xtr_flat, ytr_np)
    t = time.time() - t0
    res = metrics(yte_np, rf.predict(Xte_flat), rf.predict_proba(Xte_flat)[:, 1])
    res["time"] = t
    report("Random Forest", "random_forest", res)
    add_row("Random Forest", "ML", res, f"{RF_TREES} trees")

    # Optional PatchTST row (only the numbers YOU provide)
    if all(v is not None for v in PATCHTST_RESULT.values()):
        rows.append({
            "Model": "PatchTST (ours)", "Type": "Deep (Transformer)", "Parameters": "-",
            "Accuracy": PATCHTST_RESULT["Accuracy"], "Macro-F1": PATCHTST_RESULT["Macro-F1"],
            "ROC-AUC": PATCHTST_RESULT["ROC-AUC"], "Train time (s)": "-",
        })

    table = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "baseline_comparison.csv")
    table.to_csv(csv_path, index=False)

    print("=" * 70)
    print("FINAL BASELINE COMPARISON TABLE")
    print("=" * 70)
    print("| " + " | ".join(table.columns) + " |")
    print("|" + "|".join("-" * (len(c) + 2) for c in table.columns) + "|")
    for _, row in table.iterrows():
        print("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values) + " |")
    if not all(v is not None for v in PATCHTST_RESULT.values()):
        print("\n(PatchTST row not shown: fill PATCHTST_RESULT at the top of the script with your measured results.)")
    print(f"\nTable saved -> {csv_path}")
    print(f"Confusion matrices saved in -> {OUT_DIR}")


if __name__ == "__main__":
    main()