#!/usr/bin/env python3
"""
datap2.py  —  PCAP → time-series feature CSV
Fixes applied:
  - Bug 5: unique group_id per PCAP file (not a single 0 for all)
  - Bug 4: class counts printed so you can spot imbalance before training
  - Sliding window includes 'second' column exclusion for cleanliness
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scapy.all import PcapReader, IP
from tqdm import tqdm
import warnings
import gc

# ---------------- CONFIG ----------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
DATA_DIR = "/home/ashok/agrim/dataset/vnat_dataset"
OUTPUT   = "/home/ashok/agrim/dataset/take2/vnat_timeseries.csv"
WINDOW   = 5   # sliding window size (seconds)

# ---------------- UTILITIES ----------------
def consecutive_ratio(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) < 2:
        return np.array([1.0])
    diff = np.diff(arr)
    prev = diff[:-1]
    curr = diff[1:]
    prev[prev == 0] = 1e-6
    return curr / prev

def stats_1d(x):
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(x)), "std": float(np.std(x))}

def stats_2d(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) == 0: x = np.array([0.0])
    if len(y) == 0: y = np.array([0.0])
    if np.std(x) == 0 or np.std(y) == 0:
        return {"cov": 0.0, "corr": 0.0}
    m = min(len(x), len(y))
    cov  = np.cov(x[:m], y[:m])[0, 1] if m > 1 else 0.0
    corr = pearsonr(x[:m], y[:m])[0]
    return {"cov": float(cov), "corr": float(corr)}

def size_bins(arr):
    arr  = np.array(arr, dtype=float)
    bins = [250, 500, 750, 1000, 1250, 1500]
    if len(arr) > 0:
        return {f"bin_{b}": int(np.sum(arr <= b)) for b in bins}
    return {f"bin_{b}": 0 for b in bins}

def safe_ptr(times):
    times = np.array(times, dtype=float)
    if len(times) < 2:
        return 0.0
    iat      = np.diff(times)
    mean_iat = np.mean(iat) if np.mean(iat) != 0 else 1e-6
    return float(np.std(iat) / mean_iat)

# ---------------- PACKET EXTRACTION ----------------
def extract_packets(pcap_file):
    data   = []
    ref_ip = None
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if IP in pkt:
                ts  = float(pkt.time)
                sz  = len(pkt)
                src = pkt[IP].src
                dst = pkt[IP].dst
                data.append([ts, sz, src, dst])
                if ref_ip is None:
                    ref_ip = src
    if not data:
        return pd.DataFrame()
    df         = pd.DataFrame(data, columns=["timestamp", "size", "src", "dst"])
    df["dir"]  = np.where(df["src"] == ref_ip, "out", "in")
    df["second"] = df["timestamp"].astype(int)
    return df

# ---------------- FEATURE COMPUTATION ----------------
def compute_features(df):
    rows = []
    for sec, g in df.groupby("second"):
        sizes      = g["size"].values
        times      = g["timestamp"].values
        sizes_in   = g[g["dir"] == "in"]["size"].values
        sizes_out  = g[g["dir"] == "out"]["size"].values
        times_in   = g[g["dir"] == "in"]["timestamp"].values
        times_out  = g[g["dir"] == "out"]["timestamp"].values

        dt      = consecutive_ratio(times)
        dt_in   = consecutive_ratio(times_in)
        dt_out  = consecutive_ratio(times_out)
        sz      = consecutive_ratio(sizes)
        sz_in   = consecutive_ratio(sizes_in)
        sz_out  = consecutive_ratio(sizes_out)
        acc     = consecutive_ratio(np.cumsum(sizes))
        acc_in  = consecutive_ratio(np.cumsum(sizes_in))
        acc_out = consecutive_ratio(np.cumsum(sizes_out))

        row = {"second": sec}

        row.update({f"td_{k}":      v for k, v in stats_1d(dt).items()})
        row.update({f"td_in_{k}":   v for k, v in stats_1d(dt_in).items()})
        row.update({f"td_out_{k}":  v for k, v in stats_1d(dt_out).items()})
        row.update({f"td_corr_{k}": v for k, v in stats_2d(dt_in, dt_out).items()})

        row.update({f"sz_{k}":      v for k, v in stats_1d(sz).items()})
        row.update({f"sz_in_{k}":   v for k, v in stats_1d(sz_in).items()})
        row.update({f"sz_out_{k}":  v for k, v in stats_1d(sz_out).items()})
        row.update({f"sz_corr_{k}": v for k, v in stats_2d(sz_in, sz_out).items()})

        row.update({f"fw_{k}": v for k, v in size_bins(sizes_out).items()})
        row.update({f"bw_{k}": v for k, v in size_bins(sizes_in).items()})

        row.update({f"acc_{k}":      v for k, v in stats_1d(acc).items()})
        row.update({f"acc_in_{k}":   v for k, v in stats_1d(acc_in).items()})
        row.update({f"acc_out_{k}":  v for k, v in stats_1d(acc_out).items()})
        row.update({f"acc_corr_{k}": v for k, v in stats_2d(acc_in, acc_out).items()})

        row["ptr"]              = safe_ptr(times)
        row["pscr"]             = float(np.mean(np.abs(np.diff(sizes)))) if len(sizes) > 1 else 0.0
        row["direction_switch"] = int(np.sum(g["dir"].values[1:] != g["dir"].values[:-1])) if len(g) > 1 else 0

        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- SLIDING WINDOWS ----------------
def sliding_windows(df):
    # Drop 'second' before windowing — it is an absolute timestamp, not a feature
    if "second" in df.columns:
        df = df.drop(columns=["second"])
    if df.empty or len(df) < WINDOW:
        return df
    windows = []
    for i in range(len(df) - WINDOW + 1):
        w   = df.iloc[i:i + WINDOW]
        row = {}
        for col in df.columns:
            for j in range(WINDOW):
                row[f"{col}_t{j}"] = w[col].iloc[j]
        windows.append(row)
    return pd.DataFrame(windows)

# ---------------- PROCESS ONE FILE ----------------
def process_file(pcap_file):
    basename = os.path.basename(pcap_file)
    label    = 1 if basename.startswith("vpn_") else 0

    df = extract_packets(pcap_file)
    if df.empty:
        return pd.DataFrame()

    features = compute_features(df)
    windows  = sliding_windows(features)
    if windows.empty:
        windows = features

    # FIX BUG 5: unique group_id per file so TFT treats each capture separately
    # Strip extension and sanitize for safe use as a categorical ID
    group_id = basename.replace(".pcap", "").replace(".", "_").replace("-", "_")

    windows["label"]    = label
    windows["group_id"] = group_id

    # Sequential time index within this file's windows
    windows["time_idx"] = range(len(windows))

    return windows

# ---------------- MAIN ----------------
def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pcap")))
    if not files:
        print(f"No .pcap files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} PCAP files")
    first_file  = True
    vpn_count   = 0
    nonvpn_count = 0

    for f in tqdm(files, desc="Processing PCAPs"):
        df = process_file(f)
        if not df.empty:
            vpn_count    += int((df["label"] == 1).sum())
            nonvpn_count += int((df["label"] == 0).sum())
            df.to_csv(OUTPUT, mode="w" if first_file else "a",
                      header=first_file, index=False)
            first_file = False
        del df
        gc.collect()

    print(f"\nDataset saved to: {OUTPUT}")
    print(f"VPN windows     : {vpn_count}")
    print(f"Non-VPN windows : {nonvpn_count}")
    ratio = vpn_count / nonvpn_count if nonvpn_count > 0 else float("inf")
    print(f"VPN/NonVPN ratio: {ratio:.2f}  (use this in train.py class weights)")

if __name__ == "__main__":
    main()
