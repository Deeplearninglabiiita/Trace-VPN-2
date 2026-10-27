#!/usr/bin/env python3
"""
vvtrain.py  —  VPN / Non-VPN classifier using PatchTST
               (Patch Time Series Transformer — Nie et al., 2023)

Why PatchTST instead of TFT:
  TFT is a FORECASTING model — it predicts future values, not class labels.
  Using it for classification required hacks (CrossEntropy on top of a
  forecasting head) that caused architectural data leakage.

  PatchTST is a pure DISCRIMINATIVE transformer:
    - No forecasting concepts (no encoder length, no prediction length)
    - No group_id / time_idx bookkeeping
    - Each CSV row = one independent sample
    - Patch the time series like ViT patches images
    - CLS token → classification head

Architecture for your data:
  Each sample = 5-second window, 39 features/second = 195 total features
  Reshape to (5 patches × 39 features) → PatchTST → binary label
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ──────────────────────── CONFIG ────────────────────────
WINDOW        = 5    # sliding window seconds — must match datap2.py
BASE_FEATURES = 39   # features per second  — total_cols / WINDOW = 195/5

D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 3
D_FF      = 256
DROPOUT   = 0.1

BATCH_SIZE    = 512
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 8

TRAIN_PATH  = "/home/ashok/agrim/dataset/take2/train.csv"
VAL_PATH    = "/home/ashok/agrim/dataset/take2/test.csv"
MODEL_PATH  = "/home/ashok/agrim/dataset/take2/models/patchtst_model.pt"
SCALER_PATH = "/home/ashok/agrim/dataset/take2/models/scaler.pkl"

EXCLUDE = {"label", "time_idx", "group_id", "file_id"}

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────── DATASET ───────────────────────
class VPNDataset(Dataset):
    def __init__(self, df, feature_cols):
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        self.X = torch.tensor(X).view(-1, WINDOW, BASE_FEATURES)
        self.y = torch.tensor(y)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]

# ──────────────────────── MODEL ─────────────────────────
class PatchTSTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(BASE_FEATURES, D_MODEL)
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

# ──────────────────────── LOAD DATA ─────────────────────
print("\nLoading data...")
train_df = pd.read_csv(TRAIN_PATH).fillna(0)
val_df   = pd.read_csv(VAL_PATH).fillna(0)
train_df["label"] = train_df["label"].astype(int)
val_df["label"]   = val_df["label"].astype(int)
print(f"Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")

feature_cols = [c for c in train_df.columns if c not in EXCLUDE]
print(f"Feature columns: {len(feature_cols)}")

assert len(feature_cols) == WINDOW * BASE_FEATURES, (
    f"Expected {WINDOW}×{BASE_FEATURES}={WINDOW*BASE_FEATURES} cols, got {len(feature_cols)}.\n"
    f"Update BASE_FEATURES in CONFIG. Hint: BASE_FEATURES = {len(feature_cols)} / {WINDOW} = {len(feature_cols)//WINDOW}"
)

# ──────────────────────── CLASS WEIGHTS ─────────────────
vpn_n    = int((train_df["label"] == 1).sum())
nonvpn_n = int((train_df["label"] == 0).sum())
total    = vpn_n + nonvpn_n
class_weights = torch.tensor(
    [total / (2.0 * nonvpn_n), total / (2.0 * vpn_n)],
    dtype=torch.float32
).to(device)
print(f"Class weights — NonVPN: {class_weights[0]:.4f}  VPN: {class_weights[1]:.4f}")

# ──────────────────────── SCALE ─────────────────────────
scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved → {SCALER_PATH}")

# ──────────────────────── DATALOADERS ───────────────────
train_loader = DataLoader(VPNDataset(train_df, feature_cols),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(VPNDataset(val_df, feature_cols),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# ──────────────────────── TRAIN ─────────────────────────
model     = PatchTSTClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

best_val_f1 = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss, preds_all, labels_all = 0.0, [], []

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(y)
        preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_acc   = accuracy_score(labels_all, preds_all)
    train_f1    = f1_score(labels_all, preds_all, average="macro", zero_division=0)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            logits = model(X.to(device))
            val_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            val_labels.extend(y.numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    print(f"Epoch {epoch:03d} | LR: {scheduler.get_last_lr()[0]:.2e} | "
          f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.5f} | Train F1: {train_f1:.5f} | "
          f"Val Acc: {val_acc:.5f} | Val F1: {val_f1:.5f}")

    if val_f1 >= best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  Best model saved (Val F1: {best_val_f1:.5f})")

    gc.collect()

print(f"\nTraining complete. Best Val F1: {best_val_f1:.5f}")
print(f"Model  → {MODEL_PATH}")
print(f"Scaler → {SCALER_PATH}")
