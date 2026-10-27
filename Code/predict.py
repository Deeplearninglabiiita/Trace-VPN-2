#!/usr/bin/env python3
"""
predict.py  —  Predict VPN / Non-VPN from any raw PCAP file using PatchTST

Usage:
    python predict.py --pcap /path/to/capture.pcap
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib

from scipy.stats import pearsonr
from scapy.all import PcapReader, IP
from torch.utils.data import Dataset, DataLoader

# ─────────────────────── CONFIG ────────────────────────
WINDOW        = 5
BASE_FEATURES = 39

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 256
DROPOUT  = 0.1

BATCH_SIZE  = 256
NUM_WORKERS = 4

MODEL_PATH  = "/home/ashok/agrim/dataset/take2/models/patchtst_model.pt"
SCALER_PATH = "/home/ashok/agrim/dataset/take2/models/scaler.pkl"

EXCLUDE = {"label", "time_idx", "group_id", "file_id"}

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

# ─────────────────────── FEATURE EXTRACTION ─────────────

def consecutive_ratio(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) < 2: return np.array([1.0])
    diff = np.diff(arr)
    prev = diff[:-1]; curr = diff[1:]
    prev[prev == 0] = 1e-6
    return curr / prev

def stats_1d(x):
    x = np.array(x, dtype=float)
    if len(x) == 0: return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}

def stats_2d(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    if len(x) == 0: x = np.array([0.0])
    if len(y) == 0: y = np.array([0.0])
    if np.std(x) == 0 or np.std(y) == 0: return {"cov": 0.0, "corr": 0.0}
    m = min(len(x), len(y))
    return {"cov": float(np.cov(x[:m], y[:m])[0,1] if m > 1 else 0.0),
            "corr": float(__import__("scipy.stats", fromlist=["pearsonr"]).pearsonr(x[:m], y[:m])[0])}

def size_bins(arr):
    arr = np.array(arr, dtype=float)
    bins = [250, 500, 750, 1000, 1250, 1500]
    return {f"bin_{b}": int(np.sum(arr <= b)) for b in bins} if len(arr) > 0 \
           else {f"bin_{b}": 0 for b in bins}

def safe_ptr(times):
    times = np.array(times, dtype=float)
    if len(times) < 2: return 0.0
    iat = np.diff(times); m = np.mean(iat) if np.mean(iat) != 0 else 1e-6
    return float(np.std(iat) / m)

def extract_packets(pcap_file):
    data = []; ref_ip = None
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if IP in pkt:
                data.append([float(pkt.time), len(pkt), pkt[IP].src, pkt[IP].dst])
                if ref_ip is None: ref_ip = pkt[IP].src
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=["timestamp","size","src","dst"])
    df["dir"]    = np.where(df["src"] == ref_ip, "out", "in")
    df["second"] = df["timestamp"].astype(int)
    return df

def compute_features(df):
    rows = []
    for _, g in df.groupby("second"):
        sizes = g["size"].values; times = g["timestamp"].values
        si = g[g["dir"]=="in"]["size"].values;  so = g[g["dir"]=="out"]["size"].values
        ti = g[g["dir"]=="in"]["timestamp"].values; to_ = g[g["dir"]=="out"]["timestamp"].values
        row = {}
        row.update({f"td_{k}":      v for k,v in stats_1d(consecutive_ratio(times)).items()})
        row.update({f"td_in_{k}":   v for k,v in stats_1d(consecutive_ratio(ti)).items()})
        row.update({f"td_out_{k}":  v for k,v in stats_1d(consecutive_ratio(to_)).items()})
        row.update({f"td_corr_{k}": v for k,v in stats_2d(consecutive_ratio(ti), consecutive_ratio(to_)).items()})
        row.update({f"sz_{k}":      v for k,v in stats_1d(consecutive_ratio(sizes)).items()})
        row.update({f"sz_in_{k}":   v for k,v in stats_1d(consecutive_ratio(si)).items()})
        row.update({f"sz_out_{k}":  v for k,v in stats_1d(consecutive_ratio(so)).items()})
        row.update({f"sz_corr_{k}": v for k,v in stats_2d(consecutive_ratio(si), consecutive_ratio(so)).items()})
        row.update({f"fw_{k}": v for k,v in size_bins(so).items()})
        row.update({f"bw_{k}": v for k,v in size_bins(si).items()})
        row.update({f"acc_{k}":      v for k,v in stats_1d(consecutive_ratio(np.cumsum(sizes))).items()})
        row.update({f"acc_in_{k}":   v for k,v in stats_1d(consecutive_ratio(np.cumsum(si))).items()})
        row.update({f"acc_out_{k}":  v for k,v in stats_1d(consecutive_ratio(np.cumsum(so))).items()})
        row.update({f"acc_corr_{k}": v for k,v in stats_2d(consecutive_ratio(np.cumsum(si)), consecutive_ratio(np.cumsum(so))).items()})
        row["ptr"]              = safe_ptr(times)
        row["pscr"]             = float(np.mean(np.abs(np.diff(sizes)))) if len(sizes) > 1 else 0.0
        row["direction_switch"] = int(np.sum(g["dir"].values[1:] != g["dir"].values[:-1])) if len(g) > 1 else 0
        rows.append(row)
    return pd.DataFrame(rows)

def sliding_windows(df):
    if df.empty or len(df) < WINDOW: return pd.DataFrame()
    windows = []
    for i in range(len(df) - WINDOW + 1):
        w = df.iloc[i:i+WINDOW]; row = {}
        for col in df.columns:
            for j in range(WINDOW): row[f"{col}_t{j}"] = w[col].iloc[j]
        windows.append(row)
    return pd.DataFrame(windows)

# ─────────────────────── INFERENCE DATASET ──────────────
class WindowDataset(Dataset):
    def __init__(self, X_tensor): self.X = X_tensor
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]

# ─────────────────────── MAIN ───────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap",    required=True)
    parser.add_argument("--model",   default=MODEL_PATH)
    parser.add_argument("--scaler",  default=SCALER_PATH)
    args = parser.parse_args()

    for path, name in [(args.pcap,"PCAP"),(args.model,"Model"),(args.scaler,"Scaler")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Extract features
    print(f"\nReading: {args.pcap}")
    raw = extract_packets(args.pcap)
    if raw.empty: print("ERROR: No IP packets found."); sys.exit(1)
    print(f"  Packets : {len(raw)}")

    features = compute_features(raw)
    print(f"  Seconds : {len(features)}")

    windows = sliding_windows(features)
    if windows.empty:
        print(f"ERROR: Need at least {WINDOW} seconds. Got {len(features)}."); sys.exit(1)
    print(f"  Windows : {len(windows)}")

    # Scale
    scaler = joblib.load(args.scaler)
    feature_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") \
                   else [c for c in windows.columns if c not in EXCLUDE]
    for col in feature_cols:
        if col not in windows.columns: windows[col] = 0.0
    windows[feature_cols] = scaler.transform(windows[feature_cols])

    # To tensor
    X = torch.tensor(
        windows[feature_cols].values.astype(np.float32)
    ).view(-1, WINDOW, BASE_FEATURES)
    loader = DataLoader(WindowDataset(X), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    model = PatchTSTClassifier().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Predict
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            probs = F.softmax(model(batch.to(device)), dim=1)
            all_preds.extend(torch.argmax(probs, 1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    vpn_wins = int(np.sum(np.array(all_preds) == 1))
    total    = len(all_preds)
    vpn_pct  = vpn_wins / total * 100
    avg_prob = float(np.mean(all_probs)) * 100

    print("\n========== PREDICTION RESULTS ==========\n")
    print(f"PCAP file       : {args.pcap}")
    print(f"Total windows   : {total}")
    print(f"VPN windows     : {vpn_wins}  ({vpn_pct:.1f}%)")
    print(f"Non-VPN windows : {total-vpn_wins}  ({100-vpn_pct:.1f}%)")
    print(f"Avg VPN prob    : {avg_prob:.2f}%")

    verdict    = "VPN"    if vpn_pct >= 50 else "NON-VPN"
    confidence = vpn_pct  if vpn_pct >= 50 else 100 - vpn_pct
    print(f"\n>>> VERDICT: {verdict}  (confidence: {confidence:.1f}%) <<<\n")

if __name__ == "__main__":
    main()
