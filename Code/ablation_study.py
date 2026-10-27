#!/usr/bin/env python3
"""
ablation_study.py  --  Feature ablation on VNAT with PatchTST

Runs exactly FOUR experiments (the full model is intentionally NOT run):
    A. Baseline                (no PTR, no PSCR, no Direction-Switch)
    B. Baseline + PTR
    C. Baseline + PSCR
    D. Baseline + Direction-Switch

Everything is reused from the existing pipeline (vvtrain.py / vvtest.py):
  * same train.csv / test.csv (file-level split from split.py, untouched)
  * StandardScaler fitted on TRAIN only, applied to test
  * same PatchTST architecture (only the input feature dim changes)
  * same optimizer / LR / scheduler / batch size / epochs / weighted CE loss
  * same checkpoint rule (best macro-F1 on the eval split, `>=`)
  * same evaluation (accuracy, macro-F1, ROC-AUC, confusion matrix)

Outputs (only these):
    ablation_table.csv + printed table
    cm_baseline.png, cm_ptr.png, cm_pscr.png, cm_direction_switch.png

Usage:  python ablation_study.py
"""

import gc
import os
import re
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────── CONFIG (identical to vvtrain.py) ────────────────────────
WINDOW        = 5      # 5-second window, 1-second stride (built by datap2.py)
ALL_FEATURES  = 39     # per-second features in the full pipeline

D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 3
D_FF      = 256
DROPOUT   = 0.1

BATCH_SIZE    = 512
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0      # data is already in memory; 0 avoids Windows multiprocessing issues (no effect on results)

EARLY_STOP_PATIENCE = 10   # stop if eval macro-F1 has not strictly improved for this many epochs
                           # (same value for all 4 experiments; set to None to disable)

SEED = 42   # same seed as split.py; re-applied before every experiment

TRAIN_PATH = r"D:\sample_pcap_tor\CODE\take2\take2\train.csv"
TEST_PATH  = r"D:\sample_pcap_tor\CODE\take2\take2\test.csv"
OUT_DIR    = r"D:\sample_pcap_tor\CODE\take2\take2\ablation"

EXCLUDE = {"label", "time_idx", "group_id", "file_id"}

# Actual feature names, as produced by datap2.py compute_features():
#   row["ptr"], row["pscr"], row["direction_switch"]
# (windowed columns are named <name>_t0 ... <name>_t4)
PTR_NAME  = "ptr"                # Packet Timing Ratio   (std(IAT)/mean(IAT))
PSCR_NAME = "pscr"               # Packet Size Change Ratio (mean |diff(sizes)|)
DS_NAME   = "direction_switch"   # Direction-Switch      (# in/out flips per second)

# (title, short key for filenames, include PTR, include PSCR, include DS)
EXPERIMENTS = [
    ("Baseline",           "baseline",         False, False, False),
    ("+ PTR",              "ptr",              True,  False, False),
    ("+ PSCR",             "pscr",             False, True,  False),
    ("+ Direction-Switch", "direction_switch", False, False, True),
]

# Resolve data paths: fall back to CWD / script dir if the original paths are absent
def _resolve(path):
    if os.path.exists(path):
        return path
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        alt = os.path.join(base, os.path.basename(path))
        if os.path.exists(alt):
            return alt
    return path

TRAIN_PATH = _resolve(TRAIN_PATH)
TEST_PATH  = _resolve(TEST_PATH)
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except OSError:
    OUT_DIR = os.path.join(os.getcwd(), "ablation_outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fail(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────── DATASET (same as vvtrain.py) ────────────────────────
class VPNDataset(Dataset):
    def __init__(self, df, feature_cols, n_features):
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        # Same reshape as the original pipeline: .view(-1, WINDOW, n_features)
        self.X = torch.tensor(X).view(-1, WINDOW, n_features)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ──────────────────────── MODEL (same as vvtrain.py; only input dim is a parameter) ───
class PatchTSTClassifier(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.patch_embed = nn.Linear(n_features, D_MODEL)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed   = nn.Parameter(torch.zeros(1, WINDOW + 1, D_MODEL))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
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


# ──────────────────────── FEATURE MAPPING ────────────────────────
def base_feature_names(columns):
    """Per-second feature names in original column order (strip the _t<j> suffix)."""
    names = []
    for c in columns:
        if c in EXCLUDE:
            continue
        m = re.fullmatch(r"(.+)_t(\d+)", c)
        if m is None:
            fail(f"Unexpected column name '{c}' (expected <feature>_t<j>).")
        if m.group(1) not in names:
            names.append(m.group(1))
    return names


def columns_for(feature_names):
    """Windowed columns, kept in the original feature-major order of the CSV."""
    return [f"{n}_t{j}" for n in feature_names for j in range(WINDOW)]


# ──────────────────────── RUN ONE EXPERIMENT ────────────────────────
def run_experiment(plan, idx, total, train_raw, test_raw):
    title, cols, nf = plan["title"], plan["cols"], plan["n_features"]
    print("=" * 70)
    print(f"[{idx}/{total}] {title}   ({nf} features, {len(cols)} columns)")
    print("=" * 70)
    set_seed(SEED)

    tr = train_raw[cols + ["label"]].copy()
    te = test_raw[cols + ["label"]].copy()

    # Class weights from TRAIN only (as in vvtrain.py)
    vpn_n    = int((tr["label"] == 1).sum())
    nonvpn_n = int((tr["label"] == 0).sum())
    tot      = vpn_n + nonvpn_n
    class_weights = torch.tensor([tot / (2.0 * nonvpn_n), tot / (2.0 * vpn_n)],
                                 dtype=torch.float32).to(device)

    # Scaler: fit on TRAIN only, transform test (no leakage)
    scaler = StandardScaler()
    tr[cols] = scaler.fit_transform(tr[cols])
    te[cols] = scaler.transform(te[cols])

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(VPNDataset(tr, cols, nf), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, generator=g,
                              persistent_workers=NUM_WORKERS > 0)
    test_loader  = DataLoader(VPNDataset(te, cols, nf), batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)

    model     = PatchTSTClassifier(nf).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state = 0.0, None
    no_improve, epochs_run = 0, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for X, y in tqdm(train_loader, desc=f"[{idx}/{total}] Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
        scheduler.step()
        loss_sum /= len(train_loader.dataset)

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for X, y in test_loader:
                vp.extend(torch.argmax(model(X.to(device)), 1).cpu().numpy())
                vl.extend(y.numpy())
        v_acc = accuracy_score(vl, vp)
        v_f1  = f1_score(vl, vp, average="macro", zero_division=0)
        print(f"  Epoch {epoch:03d} | Loss: {loss_sum:.6f} | Eval Acc: {v_acc:.5f} | Eval F1: {v_f1:.5f}")

        # Same checkpoint rule as vvtrain.py (>=), kept in memory so the
        # original model/scaler files are never overwritten.
        epochs_run = epoch
        if v_f1 > best_f1:
            no_improve = 0
        else:
            no_improve += 1
        if v_f1 >= best_f1:
            best_f1 = v_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        gc.collect()

        if EARLY_STOP_PATIENCE is not None and no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no F1 improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"  Epochs run: {epochs_run}/{NUM_EPOCHS}  | best eval macro-F1: {best_f1:.5f}")

    # Final evaluation with the best checkpoint (as vvtest.py does)
    model.load_state_dict(best_state)
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            p = F.softmax(model(X.to(device)), dim=1)
            preds.extend(torch.argmax(p, 1).cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(p[:, 1].cpu().numpy())

    res = dict(
        acc=accuracy_score(labels, preds),
        f1=f1_score(labels, preds, average="macro", zero_division=0),
        auc=roc_auc_score(labels, probs),
        cm=confusion_matrix(labels, preds, labels=[0, 1]),
    )
    print(f"  -> Accuracy {res['acc']:.6f} | Macro-F1 {res['f1']:.6f} | ROC-AUC {res['auc']:.6f}\n")
    del model, optimizer, train_loader, test_loader
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



def main():
    # ──────────────────────── LOAD DATA / SANITY ────────────────────────
    print("=" * 70)
    print("FEATURE ABLATION STUDY  (PatchTST, VNAT)  --  4 experiments, no full model")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Train CSV  : {TRAIN_PATH}")
    print(f"Test CSV   : {TEST_PATH}")
    print(f"Output dir : {OUT_DIR}")

    for p, n in [(TRAIN_PATH, "train.csv"), (TEST_PATH, "test.csv")]:
        if not os.path.exists(p):
            fail(f"{n} not found at {p}")

    train_raw = pd.read_csv(TRAIN_PATH).fillna(0)
    test_raw  = pd.read_csv(TEST_PATH).fillna(0)
    train_raw["label"] = train_raw["label"].astype(int)
    test_raw["label"]  = test_raw["label"].astype(int)
    print(f"Train rows : {len(train_raw)}  |  Test rows : {len(test_raw)}")

    all_names = base_feature_names(train_raw.columns)
    if len(all_names) != ALL_FEATURES:
        fail(f"Expected {ALL_FEATURES} per-second features, found {len(all_names)}.")
    if list(base_feature_names(test_raw.columns)) != all_names:
        fail("train.csv and test.csv have different feature columns.")

    print("\nFeature mapping")
    print(f"PTR -> {PTR_NAME}")
    print(f"PSCR -> {PSCR_NAME}")
    print(f"Direction-Switch -> {DS_NAME}")

    ABLATED = [PTR_NAME, PSCR_NAME, DS_NAME]
    for n in ABLATED:
        if n not in all_names:
            fail(f"Feature '{n}' not found among per-second features: {all_names}")
        missing = [c for c in columns_for([n]) if c not in train_raw.columns or c not in test_raw.columns]
        if missing:
            fail(f"Columns missing for feature '{n}': {missing}")

    baseline_names = [n for n in all_names if n not in ABLATED]
    print(f"Baseline feature count -> {len(baseline_names)}")
    if len(baseline_names) != ALL_FEATURES - 3:
        fail(f"Baseline should have {ALL_FEATURES - 3} features, got {len(baseline_names)}.")
    for n in ABLATED:
        if n in baseline_names:
            fail(f"Ablated feature '{n}' leaked into baseline.")

    # Labels
    for nm, d in [("train", train_raw), ("test", test_raw)]:
        if set(d["label"].unique()) != {0, 1}:
            fail(f"{nm} labels are {sorted(d['label'].unique())}, expected {{0, 1}}.")
    print(f"Labels OK  | train VPN/NonVPN: {(train_raw.label==1).sum()}/{(train_raw.label==0).sum()}"
          f"  | test VPN/NonVPN: {(test_raw.label==1).sum()}/{(test_raw.label==0).sum()}")

    # Build each experiment's feature list and verify
    plans = []
    for title, key, use_ptr, use_pscr, use_ds in EXPERIMENTS:
        names = list(baseline_names)
        if use_ptr:  names.append(PTR_NAME)
        if use_pscr: names.append(PSCR_NAME)
        if use_ds:   names.append(DS_NAME)
        # keep original feature ordering
        names = [n for n in all_names if n in names]
        cols = columns_for(names)
        expected = len(baseline_names) + int(use_ptr) + int(use_pscr) + int(use_ds)
        if len(names) != expected or len(cols) != expected * WINDOW:
            fail(f"[{title}] feature count mismatch: {len(names)} vs expected {expected}.")
        for n, flag in [(PTR_NAME, use_ptr), (PSCR_NAME, use_pscr), (DS_NAME, use_ds)]:
            if (n in names) != flag:
                fail(f"[{title}] wrong inclusion of '{n}'.")
        plans.append(dict(title=title, key=key, ptr=use_ptr, pscr=use_pscr, ds=use_ds,
                          names=names, cols=cols, n_features=len(names)))

    # Lightweight tensor / forward-pass check for every configuration
    print("\nSanity check (tensor shapes + one forward pass)")
    for p in plans:
        probe = train_raw[p["cols"]].head(8).values.astype(np.float32)
        t = torch.tensor(probe).view(-1, WINDOW, p["n_features"])
        if tuple(t.shape) != (8, WINDOW, p["n_features"]):
            fail(f"[{p['title']}] bad tensor shape {tuple(t.shape)}.")
        set_seed(SEED)
        m = PatchTSTClassifier(p["n_features"]).to(device).eval()
        with torch.no_grad():
            out = m(t.to(device))
        if tuple(out.shape) != (8, 2) or not torch.isfinite(out).all():
            fail(f"[{p['title']}] bad forward-pass output {tuple(out.shape)}.")
        print(f"  {p['title']:<20} features={p['n_features']:>2}  input={tuple(t.shape)}  logits={tuple(out.shape)}  OK")
        del m
    print("All sanity checks passed.\n")


    # ──────────────────────── MAIN ────────────────────────
    rows = []
    for i, plan in enumerate(plans, 1):
        r = run_experiment(plan, i, len(plans), train_raw, test_raw)
        cm_path = save_confusion_matrix(r["cm"], plan["title"], plan["key"])
        print(f"  Confusion matrix ({plan['title']}):")
        print(f"                    Pred NON-VPN   Pred VPN")
        print(f"    Actual NON-VPN  {r['cm'][0][0]:<14} {r['cm'][0][1]}")
        print(f"    Actual VPN      {r['cm'][1][0]:<14} {r['cm'][1][1]}")
        print(f"  Saved -> {cm_path}\n")
        rows.append({
            "Experiment": plan["title"],
            "PTR": "Yes" if plan["ptr"] else "No",
            "PSCR": "Yes" if plan["pscr"] else "No",
            "Direction-Switch": "Yes" if plan["ds"] else "No",
            "# Features": plan["n_features"],
            "Accuracy": round(r["acc"], 6),
            "Macro-F1": round(r["f1"], 6),
            "ROC-AUC": round(r["auc"], 6),
        })

    table = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "ablation_table.csv")
    table.to_csv(csv_path, index=False)

    print("=" * 70)
    print("FINAL ABLATION TABLE")
    print("=" * 70)
    print("| " + " | ".join(table.columns) + " |")
    print("|" + "|".join("-" * (len(c) + 2) for c in table.columns) + "|")
    for _, row in table.iterrows():
        print("| " + " | ".join(
            f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values) + " |")
    print(f"\nTable saved -> {csv_path}")
    print(f"Confusion matrices saved in -> {OUT_DIR}")


if __name__ == "__main__":
    main()
