#!/usr/bin/env python3
"""
ablation2.py  --  Window-size ablation (5 s vs 15 s vs 30 s vs 60 s) on VNAT with PatchTST

Question answered: how does the length of the sliding window (in seconds) affect
VPN / Non-VPN detection?   Stride is 1 second for every window size.

How the data is built (no PCAP re-processing needed, no change of split):
  * train.csv / test.csv hold 5-second windows with 1-second stride, one file_id
    per capture.  Consecutive rows of a file overlap by 4 seconds, so the original
    per-second feature sequence of each capture can be recovered exactly
    (this is verified against every row; inconsistent files are skipped and reported).
  * Using the SAME file-level train/test split (split.py), per-second sequences are
    re-windowed with W = 5, 15, 30, 60 seconds.  Captures shorter than W seconds
    cannot form a window and are dropped for that W (counts are printed).
  * StandardScaler is fitted on TRAIN seconds only and applied to test.

Everything else is identical for every W (same as vvtrain.py / ablation_study.py):
  all 39 features, PatchTST (D_MODEL=128, 4 heads, 3 layers, FF=256, dropout 0.1),
  AdamW lr 1e-4 / wd 1e-4, cosine schedule over 50 epochs, batch 512, class-weighted
  cross-entropy, grad-clip 1.0, early stopping (patience 10), seed 42, best-F1 checkpoint.
  Only the sequence length (number of tokens = W seconds) changes.

NOTE: here each token is ONE SECOND of 39 features (a proper (W, 39) sequence).
The original CSV-reshape in vvtrain.py groups columns differently, so the 5 s row
here may differ slightly from your original full-model result.  All W values are
run with the same code, so the comparison between them is fair.

Outputs:  window_ablation_table.csv (+ printed table) and one confusion matrix PNG per W.

Usage:  python ablation2.py
"""

import gc
import os
import re
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────── CONFIG ────────────────────────
WINDOW_SIZES = [5, 15, 30, 60]     # seconds (stride = 1 s)
SRC_WINDOW   = 5                   # window length stored in train.csv / test.csv
N_FEATURES   = 39                  # per-second features

D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 3
D_FF      = 256
DROPOUT   = 0.1

BATCH_SIZE    = 512
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOP_PATIENCE = 10           # None to disable; same for every W
SEED = 42

TRAIN_PATH = r"D:\sample_pcap_tor\CODE\take2\take2\train.csv"
TEST_PATH  = r"D:\sample_pcap_tor\CODE\take2\take2\test.csv"
OUT_DIR    = r"D:\sample_pcap_tor\CODE\take2\take2\ablation2"

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


# ──────────────────────── MODEL (same as vvtrain.py; sequence length is a parameter) ───
class PatchTSTClassifier(nn.Module):
    def __init__(self, n_features, window):
        super().__init__()
        self.patch_embed = nn.Linear(n_features, D_MODEL)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed   = nn.Parameter(torch.zeros(1, window + 1, D_MODEL))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS,
                                                 enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 2),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.transformer(x)
        return self.head(self.norm(x[:, 0]))


# ──────────────────────── PER-SECOND RECONSTRUCTION ────────────────────────
def base_feature_names(columns):
    names = []
    for c in columns:
        if c in EXCLUDE:
            continue
        m = re.fullmatch(r"(.+)_t(\d+)", c)
        if m is None:
            fail(f"Unexpected column '{c}' (expected <feature>_t<j>).")
        if m.group(1) not in names:
            names.append(m.group(1))
    return names


def rebuild_sequences(df, names, split_name):
    """
    Recover per-second sequences for every capture (file_id) from stride-1,
    5-second window rows.  Returns (list of (L, 39) arrays, list of labels, n_skipped).
    """
    seqs, labels, skipped = [], [], 0
    cols = [[f"{n}_t{j}" for j in range(SRC_WINDOW)] for n in names]
    for fid, g in df.groupby("file_id", sort=False):
        n = len(g)
        A = np.stack([g[c].values.astype(np.float64) for c in cols], axis=1)   # (n, 39, 5)
        # consistency: window k, second j  ==  window k+j, second 0
        ok = True
        for j in range(1, SRC_WINDOW):
            if n - j > 0 and not np.allclose(A[:n - j, :, j], A[j:, :, 0], rtol=1e-6, atol=1e-9):
                ok = False
                break
        if not ok or g["label"].nunique() != 1:
            skipped += 1
            continue
        seq = np.concatenate([A[:, :, 0], A[-1, :, 1:].T], axis=0)             # (n + 4, 39)
        seqs.append(seq.astype(np.float32))
        labels.append(int(g["label"].iloc[0]))
    total_files = df["file_id"].nunique()
    print(f"  {split_name}: {len(seqs)}/{total_files} captures rebuilt "
          f"({skipped} skipped: non-contiguous rows or mixed labels)")
    if skipped > 0.02 * total_files:
        fail(f"{split_name}: too many captures ({skipped}) failed the contiguity check; "
             f"rows are not in the original window order.")
    return seqs, labels


class SeqStore:
    """All seconds of one split on the device + window index (no windows stored in memory)."""
    def __init__(self, seqs, labels, scaler):
        lens = np.array([len(s) for s in seqs])
        self.offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        self.lens, self.file_labels = lens, np.array(labels)
        S = np.concatenate(seqs, axis=0)
        self.S = torch.tensor(scaler.transform(S).astype(np.float32), device=device)

    def windows(self, W):
        starts, labs = [], []
        for off, L, lab in zip(self.offsets, self.lens, self.file_labels):
            if L >= W:
                starts.append(off + np.arange(L - W + 1))
                labs.append(np.full(L - W + 1, lab))
        if not starts:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0
        n_used = len(starts)
        return np.concatenate(starts), np.concatenate(labs), n_used

    def batch(self, starts_t, W):
        idx = starts_t.unsqueeze(1) + torch.arange(W, device=device).unsqueeze(0)
        return self.S[idx]                                                      # (B, W, 39)


# ──────────────────────── TRAIN / EVAL ONE WINDOW SIZE ────────────────────────
@torch.no_grad()
def predict(model, store, starts, W):
    model.eval()
    preds, probs = [], []
    st = torch.tensor(starts, device=device)
    for i in range(0, len(st), BATCH_SIZE):
        p = F.softmax(model(store.batch(st[i:i + BATCH_SIZE], W)), dim=1)
        preds.append(torch.argmax(p, 1).cpu().numpy())
        probs.append(p[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def run_window(W, idx, total, tr_store, te_store):
    print("=" * 70)
    print(f"[{idx}/{total}] Window = {W} seconds (stride 1 s)")
    print("=" * 70)
    set_seed(SEED)

    tr_starts, tr_labels, tr_files = tr_store.windows(W)
    te_starts, te_labels, te_files = te_store.windows(W)
    print(f"  Train: {len(tr_starts)} windows from {tr_files} captures | "
          f"Test: {len(te_starts)} windows from {te_files} captures")

    vpn_n    = int((tr_labels == 1).sum())
    nonvpn_n = int((tr_labels == 0).sum())
    tot      = vpn_n + nonvpn_n
    class_weights = torch.tensor([tot / (2.0 * nonvpn_n), tot / (2.0 * vpn_n)],
                                 dtype=torch.float32).to(device)

    model     = PatchTSTClassifier(N_FEATURES, W).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    tr_st = torch.tensor(tr_starts, device=device)
    tr_y  = torch.tensor(tr_labels, dtype=torch.long, device=device)
    g = torch.Generator()
    g.manual_seed(SEED)

    best_f1, best_state, no_improve, epochs_run = 0.0, None, 0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr_st), generator=g).to(device)
        loss_sum = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            X, y = tr_store.batch(tr_st[b], W), tr_y[b]
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
        scheduler.step()
        loss_sum /= len(tr_st)

        vp, _ = predict(model, te_store, te_starts, W)
        v_acc = accuracy_score(te_labels, vp)
        v_f1  = f1_score(te_labels, vp, average="macro", zero_division=0)
        print(f"  Epoch {epoch:03d} | Loss: {loss_sum:.6f} | Eval Acc: {v_acc:.5f} | Eval F1: {v_f1:.5f}")

        epochs_run = epoch
        if v_f1 > best_f1:
            no_improve = 0
        else:
            no_improve += 1
        if v_f1 >= best_f1:                       # same checkpoint rule as vvtrain.py
            best_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if EARLY_STOP_PATIENCE is not None and no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no F1 improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    preds, probs = predict(model, te_store, te_starts, W)
    res = dict(
        n_train=len(tr_starts), n_test=len(te_starts), epochs=epochs_run,
        acc=accuracy_score(te_labels, preds),
        f1=f1_score(te_labels, preds, average="macro", zero_division=0),
        auc=roc_auc_score(te_labels, probs),
        cm=confusion_matrix(te_labels, preds, labels=[0, 1]),
    )
    print(f"  -> Accuracy {res['acc']:.6f} | Macro-F1 {res['f1']:.6f} | ROC-AUC {res['auc']:.6f} "
          f"| epochs run {epochs_run}\n")
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def save_confusion_matrix(cm, W):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["NON-VPN", "VPN"]); ax.set_yticklabels(["NON-VPN", "VPN"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {W}-second window")
    thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"cm_window_{W}s.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ──────────────────────── MAIN ────────────────────────
def main():
    global OUT_DIR
    train_path, test_path = _resolve(TRAIN_PATH), _resolve(TEST_PATH)
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_dir = OUT_DIR
    except OSError:
        out_dir = os.path.join(os.getcwd(), "ablation2_outputs")
        os.makedirs(out_dir, exist_ok=True)
    OUT_DIR = out_dir

    print("=" * 70)
    print("WINDOW-SIZE ABLATION  (PatchTST, VNAT)  --  windows: " + ", ".join(f"{w}s" for w in WINDOW_SIZES))
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
        if "file_id" not in d.columns:
            fail(f"{nm}.csv has no file_id column (needed to keep the original file-level split).")
        d["label"] = d["label"].astype(int)
        if set(d["label"].unique()) != {0, 1}:
            fail(f"{nm} labels are {sorted(d['label'].unique())}, expected {{0, 1}}.")

    names = base_feature_names(train_df.columns)
    if len(names) != N_FEATURES:
        fail(f"Expected {N_FEATURES} per-second features, found {len(names)}.")
    if base_feature_names(test_df.columns) != names:
        fail("train.csv and test.csv have different feature columns.")
    shared = set(train_df["file_id"].unique()) & set(test_df["file_id"].unique())
    if shared:
        fail(f"{len(shared)} captures appear in both train and test (split leakage).")
    print(f"Features per second: {len(names)} | train rows {len(train_df)} | test rows {len(test_df)}")
    print("Train/test captures are disjoint: OK")

    print("\nRebuilding per-second sequences from stride-1 windows ...")
    tr_seqs, tr_lab = rebuild_sequences(train_df, names, "train")
    te_seqs, te_lab = rebuild_sequences(test_df, names, "test")
    del train_df, test_df
    gc.collect()

    # Scaler: fitted on TRAIN seconds only
    scaler = StandardScaler().fit(np.concatenate(tr_seqs, axis=0))
    tr_store = SeqStore(tr_seqs, tr_lab, scaler)
    te_store = SeqStore(te_seqs, te_lab, scaler)

    # Sanity checks before expensive training
    print("\nSanity check (windows per size + one forward pass)")
    for W in WINDOW_SIZES:
        trs, trl, _ = tr_store.windows(W)
        tes, tel, _ = te_store.windows(W)
        if len(trs) == 0 or len(tes) == 0:
            fail(f"No windows for W={W}.")
        if set(np.unique(trl)) != {0, 1} or set(np.unique(tel)) != {0, 1}:
            fail(f"W={W}: a class is missing in train or test windows.")
        x = tr_store.batch(torch.tensor(trs[:8], device=device), W)
        if tuple(x.shape) != (8, W, N_FEATURES):
            fail(f"W={W}: bad tensor shape {tuple(x.shape)}.")
        set_seed(SEED)
        m = PatchTSTClassifier(N_FEATURES, W).to(device).eval()
        with torch.no_grad():
            out = m(x)
        if tuple(out.shape) != (8, 2) or not torch.isfinite(out).all():
            fail(f"W={W}: bad forward-pass output {tuple(out.shape)}.")
        print(f"  W={W:>2}s  train windows={len(trs):>7}  test windows={len(tes):>6}  "
              f"input={tuple(x.shape)}  logits={tuple(out.shape)}  OK")
        del m
    print("All sanity checks passed.\n")

    rows = []
    for i, W in enumerate(WINDOW_SIZES, 1):
        r = run_window(W, i, len(WINDOW_SIZES), tr_store, te_store)
        cm_path = save_confusion_matrix(r["cm"], W)
        print(f"  Confusion matrix ({W}s):")
        print(f"                    Pred NON-VPN   Pred VPN")
        print(f"    Actual NON-VPN  {r['cm'][0][0]:<14} {r['cm'][0][1]}")
        print(f"    Actual VPN      {r['cm'][1][0]:<14} {r['cm'][1][1]}")
        print(f"  Saved -> {cm_path}\n")
        rows.append({
            "Window (s)": W,
            "Train windows": r["n_train"],
            "Test windows": r["n_test"],
            "Epochs run": r["epochs"],
            "Accuracy": round(r["acc"], 6),
            "Macro-F1": round(r["f1"], 6),
            "ROC-AUC": round(r["auc"], 6),
        })

    table = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "window_ablation_table.csv")
    table.to_csv(csv_path, index=False)

    print("=" * 70)
    print("FINAL WINDOW-SIZE ABLATION TABLE")
    print("=" * 70)
    print("| " + " | ".join(table.columns) + " |")
    print("|" + "|".join("-" * (len(c) + 2) for c in table.columns) + "|")
    for _, row in table.iterrows():
        print("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values) + " |")
    best = table.loc[table["Macro-F1"].idxmax()]
    print(f"\nBest Macro-F1: {int(best['Window (s)'])}-second window ({best['Macro-F1']:.4f})")
    print(f"Table saved -> {csv_path}")
    print(f"Confusion matrices saved in -> {OUT_DIR}")


if __name__ == "__main__":
    main()
