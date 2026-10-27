#!/usr/bin/env python3
"""
vvtest.py  —  Evaluate PatchTST VPN classifier on test.csv
No TFT / pytorch-forecasting. Pure PyTorch.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from torch.utils.data import Dataset, DataLoader

# ─────────────────────── CONFIG ────────────────────────
WINDOW        = 5
BASE_FEATURES = 39

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 256
DROPOUT  = 0.1

BATCH_SIZE  = 512
NUM_WORKERS = 8

TEST_PATH   = "/home/ashok/agrim/dataset/take2/test.csv"
MODEL_PATH  = "/home/ashok/agrim/dataset/take2/models/patchtst_model.pt"
SCALER_PATH = "/home/ashok/agrim/dataset/take2/models/scaler.pkl"
ROC_OUTPUT  = "/home/ashok/agrim/dataset/take2/roc_curve.png"

EXCLUDE = {"label", "time_idx", "group_id", "file_id"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

for path, name in [(TEST_PATH, "test.csv"), (MODEL_PATH, "model"), (SCALER_PATH, "scaler")]:
    if not os.path.exists(path):
        print(f"ERROR: {name} not found at {path}")
        sys.exit(1)

# ─────────────────────── DATASET ────────────────────────
class VPNDataset(Dataset):
    def __init__(self, df, feature_cols):
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        self.X = torch.tensor(X).view(-1, WINDOW, BASE_FEATURES)
        self.y = torch.tensor(y)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]

# ─────────────────────── MODEL ──────────────────────────
class PatchTSTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(BASE_FEATURES, D_MODEL)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed   = nn.Parameter(torch.zeros(1, WINDOW + 1, D_MODEL))
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

# ─────────────────────── LOAD DATA ──────────────────────
df = pd.read_csv(TEST_PATH).fillna(0)
df["label"] = df["label"].astype(int)
print(f"Total test samples: {len(df)}")

scaler       = joblib.load(SCALER_PATH)
feature_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") \
               else [c for c in df.columns if c not in EXCLUDE]

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

df[feature_cols] = scaler.transform(df[feature_cols])

loader = DataLoader(VPNDataset(df, feature_cols),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"Total batches: {len(loader)}")

# ─────────────────────── LOAD MODEL ─────────────────────
model = PatchTSTClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.\n")

# ─────────────────────── INFERENCE ──────────────────────
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for X, y in tqdm(loader, desc="Inference"):
        logits = model(X.to(device))
        probs  = F.softmax(logits, dim=1)
        all_preds.extend(torch.argmax(probs, 1).cpu().numpy())
        all_labels.extend(y.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

# ─────────────────────── METRICS ────────────────────────
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall    = recall_score(all_labels, all_preds, zero_division=0)
f1        = f1_score(all_labels, all_preds, zero_division=0)
cm        = confusion_matrix(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc   = auc(fpr, tpr)

print("\n========== TEST RESULTS ==========\n")
print(f"Total samples  : {len(all_labels)}")
print(f"\nAccuracy       : {accuracy:.6f}")
print(f"Precision      : {precision:.6f}")
print(f"Recall         : {recall:.6f}")
print(f"F1 Score       : {f1:.6f}")
print(f"ROC AUC        : {roc_auc:.6f}")
print("\nConfusion Matrix")
print(f"                 Pred NON-VPN   Pred VPN")
print(f"Actual NON-VPN   {cm[0][0]:<14} {cm[0][1]}")
print(f"Actual VPN       {cm[1][0]:<14} {cm[1][1]}")
print("\n--- Per-class breakdown ---")
print(classification_report(all_labels, all_preds,
                             target_names=["NON-VPN", "VPN"], zero_division=0))

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VPN Detection — ROC Curve (PatchTST)")
plt.legend(loc="lower right")
plt.savefig(ROC_OUTPUT)
print(f"ROC curve saved → {ROC_OUTPUT}")
