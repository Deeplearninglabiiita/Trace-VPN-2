#!/usr/bin/env python3
"""
evaluate.py — End-to-end VPN/Non-VPN pipeline for ISCX-style PCAP datasets.

Pipeline (mirrors datap2.py -> split.py -> vvtrain.py -> vvtest.py, but in
one script, run directly on raw PCAPs from folders):

  1. Discover PCAP files in --vpn_dirs (label=1) and --nonvpn_dirs (label=0).
  2. Extract per-second traffic features, build 5-second sliding windows.
     Results are cached to <out_dir>/train_features_cache.csv and
     test_features_cache.csv so re-runs are fast (use --force_reextract to
     bypass the cache).
  3. Split FILES 80:20 (stratified per source folder, seeded) into train/test.
     No file's windows appear in both splits.
  4. Fit a StandardScaler on TRAIN windows only. Train a fresh PatchTST
     classifier from scratch on TRAIN. Save best model + scaler into a NEW
     folder: <out_dir>/models/patchtst_model.pt and scaler.pkl.
  5. Evaluate the trained model on the TEST split: accuracy, precision,
     recall, F1 (per-class + macro/weighted), confusion matrix, ROC-AUC.
     Metrics saved at both WINDOW level (the paper's primary reported
     metric, e.g. Table IV / the 42,566-window confusion matrix) and FLOW
     level, where one flow = one pcap file / session (per the paper's own
     definition: "Each capture file corresponds to one session and is
     labeled VPN or non-VPN"), with the verdict taken as majority vote
     across that file's windows.

Usage:
    python evaluate.py ^
        --vpn_dirs "D:\\VPN-PCAPS-01zip\\VPN-PCAPS-01" "D:\\VPN-PCAPs-02zip\\VPN-PCAPs-02" ^
        --nonvpn_dirs "D:\\NonVPN-PCAPs-01-zip\\NonVPN-PCAPs-01" ^
        --test_frac 0.2 ^
        --seed 42 ^
        --epochs 50 ^
        --out_dir ./results

This trains a brand-new model from scratch (no leakage: scaler and model
only ever see the 80% train split). --model/--scaler are only used with
--skip_train, to evaluate an already-trained model on the same split.
"""

import argparse
import glob
import json
import os
import sys
import gc
import re
import struct
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, f1_score
)

warnings.filterwarnings("ignore")  # silences scipy ConstantInputWarning spam etc.

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ─────────────────────── CONFIG (must match training arch) ────────────────
WINDOW        = 5
BASE_FEATURES = 39

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 256
DROPOUT  = 0.10
LABEL_SMOOTHING = 0.0

DEFAULT_BATCH_SIZE  = 512
DEFAULT_NUM_WORKERS = 0  # Windows-safe default; bump on Linux if you like

EXCLUDE = {"label", "time_idx", "group_id", "file_id"}
PCAP_EXTS = (".pcap", ".pcapng", ".cap")


# ─────────────────────── MODEL (unchanged) ────────────────────────────────
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


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017), applied here per user request to down-
    weight easy/well-classified windows -- e.g. long runs of near-identical
    quiet-audio seconds from facebook_audio/hangouts_audio captures -- so
    gradient updates focus more on ambiguous or misclassified windows.
    gamma=0 reduces exactly to (optionally class-weighted) cross-entropy;
    gamma=2.0 is the standard default from the paper. label_smoothing is
    applied via the underlying cross_entropy call for consistency with the
    rest of the script's config (LABEL_SMOOTHING).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # per-class weights tensor, same as CrossEntropyLoss's `weight`
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.alpha, label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─────────────────────── FEATURE EXTRACTION (unchanged) ───────────────────
def consecutive_ratio(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) < 2:
        return np.array([1.0])
    diff = np.diff(arr)
    prev = diff[:-1].copy()
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
    if len(x) == 0:
        x = np.array([0.0])
    if len(y) == 0:
        y = np.array([0.0])
    if np.std(x) == 0 or np.std(y) == 0:
        return {"cov": 0.0, "corr": 0.0}
    m = min(len(x), len(y))
    if m > 1:
        cov = float(np.cov(x[:m], y[:m])[0, 1])
        from scipy.stats import pearsonr
        corr = float(pearsonr(x[:m], y[:m])[0])
    else:
        cov, corr = 0.0, 0.0
    return {"cov": cov, "corr": corr}


def size_bins(arr):
    arr = np.array(arr, dtype=float)
    bins = [250, 500, 750, 1000, 1250, 1500]
    if len(arr) > 0:
        return {f"bin_{b}": int(np.sum(arr <= b)) for b in bins}
    return {f"bin_{b}": 0 for b in bins}


def safe_ptr(times):
    times = np.array(times, dtype=float)
    if len(times) < 2:
        return 0.0
    iat = np.diff(times)
    m = np.mean(iat) if np.mean(iat) != 0 else 1e-6
    return float(np.std(iat) / m)


def _ip_to_str(b):
    return f"{b[0]}.{b[1]}.{b[2]}.{b[3]}"


def extract_packets_fast(pcap_file):
    """
    Fast raw pcap/pcapng parser: reads only what we need (timestamp, wire
    length, src/dst IPv4) without building scapy Packet objects. This is the
    dominant cost in the original pipeline (scapy dissects every layer of
    every packet); here we do minimal struct-level parsing instead.
    Falls back to scapy automatically for formats this doesn't recognize
    (e.g. unusual link types) so nothing silently breaks.
    """
    ext = os.path.splitext(pcap_file)[1].lower()
    try:
        if ext == ".pcapng":
            return _parse_pcapng(pcap_file)
        return _parse_pcap_classic(pcap_file)
    except Exception:
        return _extract_packets_scapy_fallback(pcap_file)


def _parse_pcap_classic(pcap_file):
    data = []
    ref_ip = None
    with open(pcap_file, "rb") as f:
        header = f.read(24)
        if len(header) < 24:
            return pd.DataFrame()
        magic = header[:4]
        if magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"
            nano = False
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian = "<"
            nano = True
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"
            nano = False
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian = ">"
            nano = True
        else:
            raise ValueError("not a classic pcap file")

        linktype = struct.unpack(endian + "I", header[20:24])[0]
        # 1 = Ethernet, 101 = raw IP, 113 = Linux SLL
        eth_offset = {1: 14, 101: 0, 113: 16}.get(linktype, 14)

        rec_hdr_fmt = endian + "IIII"
        rec_hdr_size = 16
        while True:
            rh = f.read(rec_hdr_size)
            if len(rh) < rec_hdr_size:
                break
            ts_sec, ts_frac, incl_len, orig_len = struct.unpack(rec_hdr_fmt, rh)
            payload = f.read(incl_len)
            if len(payload) < incl_len:
                break
            ts = ts_sec + (ts_frac / 1e9 if nano else ts_frac / 1e6)

            if len(payload) < eth_offset + 20:
                continue
            ip_start = eth_offset
            ver_ihl = payload[ip_start]
            if (ver_ihl >> 4) != 4:  # only IPv4 (matches original IP-filter behavior closely enough)
                continue
            src = payload[ip_start + 12:ip_start + 16]
            dst = payload[ip_start + 16:ip_start + 20]
            if len(src) < 4 or len(dst) < 4:
                continue
            src_s = _ip_to_str(src)
            dst_s = _ip_to_str(dst)
            data.append((ts, orig_len, src_s, dst_s))
            if ref_ip is None:
                ref_ip = src_s

    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["timestamp", "size", "src", "dst"])
    df["dir"]    = np.where(df["src"] == ref_ip, "out", "in")
    df["second"] = df["timestamp"].astype(int)
    return df


def _parse_pcapng(pcap_file):
    """Minimal pcapng parser: handles Section Header + Interface Description +
    Enhanced Packet Block, which is what modern captures (Wireshark/tshark,
    including ISCX-style dumps) use."""
    data = []
    ref_ip = None
    with open(pcap_file, "rb") as f:
        buf = f.read()
    pos = 0
    endian = "<"
    linktypes = {}  # interface_id -> linktype
    n = len(buf)
    while pos + 8 <= n:
        block_type = struct.unpack_from(endian + "I", buf, pos)[0]
        block_len = struct.unpack_from(endian + "I", buf, pos + 4)[0]
        if block_len < 12 or pos + block_len > n:
            break
        block = buf[pos:pos + block_len]

        if block_type == 0x0A0D0D0A:  # Section Header Block
            bom = struct.unpack_from("<I", block, 8)[0]
            endian = "<" if bom == 0x1A2B3C4D else ">"

        elif block_type == 0x00000001:  # Interface Description Block
            linktype = struct.unpack_from(endian + "H", block, 8)[0]
            linktypes[len(linktypes)] = linktype

        elif block_type == 0x00000006:  # Enhanced Packet Block
            iface_id = struct.unpack_from(endian + "I", block, 8)[0]
            ts_high = struct.unpack_from(endian + "I", block, 12)[0]
            ts_low  = struct.unpack_from(endian + "I", block, 16)[0]
            cap_len = struct.unpack_from(endian + "I", block, 20)[0]
            orig_len = struct.unpack_from(endian + "I", block, 24)[0]
            pkt_data = block[28:28 + cap_len]
            # EPB timestamps are typically microsecond resolution unless
            # if_tsresol option says otherwise; microsecond is the overwhelming
            # common case (Wireshark default) so we use that.
            ts = ((ts_high << 32) | ts_low) / 1e6

            linktype = linktypes.get(iface_id, 1)
            eth_offset = {1: 14, 101: 0, 113: 16}.get(linktype, 14)
            if len(pkt_data) >= eth_offset + 20:
                ip_start = eth_offset
                ver_ihl = pkt_data[ip_start]
                if (ver_ihl >> 4) == 4:
                    src = pkt_data[ip_start + 12:ip_start + 16]
                    dst = pkt_data[ip_start + 16:ip_start + 20]
                    if len(src) == 4 and len(dst) == 4:
                        src_s = _ip_to_str(src)
                        dst_s = _ip_to_str(dst)
                        data.append((ts, orig_len, src_s, dst_s))
                        if ref_ip is None:
                            ref_ip = src_s

        pos += block_len

    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["timestamp", "size", "src", "dst"])
    df["dir"]    = np.where(df["src"] == ref_ip, "out", "in")
    df["second"] = df["timestamp"].astype(int)
    return df


def _extract_packets_scapy_fallback(pcap_file):
    """Slow but maximally-compatible fallback used only if the fast parser
    can't handle a given file's format."""
    from scapy.all import PcapReader, IP
    data = []
    ref_ip = None
    with PcapReader(pcap_file) as pcap:
        for pkt in pcap:
            if IP in pkt:
                data.append([float(pkt.time), len(pkt), pkt[IP].src, pkt[IP].dst])
                if ref_ip is None:
                    ref_ip = pkt[IP].src
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["timestamp", "size", "src", "dst"])
    df["dir"]    = np.where(df["src"] == ref_ip, "out", "in")
    df["second"] = df["timestamp"].astype(int)
    return df


# Public name used elsewhere in the file — fast path with automatic fallback
extract_packets = extract_packets_fast


def compute_features(df):
    rows = []
    for sec, g in df.groupby("second"):
        sizes = g["size"].values
        times = g["timestamp"].values
        si = g[g["dir"] == "in"]["size"].values
        so = g[g["dir"] == "out"]["size"].values
        ti = g[g["dir"] == "in"]["timestamp"].values
        to_ = g[g["dir"] == "out"]["timestamp"].values
        row = {"second": sec}
        row.update({f"td_{k}":      v for k, v in stats_1d(consecutive_ratio(times)).items()})
        row.update({f"td_in_{k}":   v for k, v in stats_1d(consecutive_ratio(ti)).items()})
        row.update({f"td_out_{k}":  v for k, v in stats_1d(consecutive_ratio(to_)).items()})
        row.update({f"td_corr_{k}": v for k, v in stats_2d(consecutive_ratio(ti), consecutive_ratio(to_)).items()})
        row.update({f"sz_{k}":      v for k, v in stats_1d(consecutive_ratio(sizes)).items()})
        row.update({f"sz_in_{k}":   v for k, v in stats_1d(consecutive_ratio(si)).items()})
        row.update({f"sz_out_{k}":  v for k, v in stats_1d(consecutive_ratio(so)).items()})
        row.update({f"sz_corr_{k}": v for k, v in stats_2d(consecutive_ratio(si), consecutive_ratio(so)).items()})
        row.update({f"fw_{k}": v for k, v in size_bins(so).items()})
        row.update({f"bw_{k}": v for k, v in size_bins(si).items()})
        row.update({f"acc_{k}":      v for k, v in stats_1d(consecutive_ratio(np.cumsum(sizes))).items()})
        row.update({f"acc_in_{k}":   v for k, v in stats_1d(consecutive_ratio(np.cumsum(si))).items()})
        row.update({f"acc_out_{k}":  v for k, v in stats_1d(consecutive_ratio(np.cumsum(so))).items()})
        row.update({f"acc_corr_{k}": v for k, v in stats_2d(consecutive_ratio(np.cumsum(si)), consecutive_ratio(np.cumsum(so))).items()})
        row["ptr"]              = safe_ptr(times)
        row["pscr"]             = float(np.mean(np.abs(np.diff(sizes)))) if len(sizes) > 1 else 0.0
        row["direction_switch"] = int(np.sum(g["dir"].values[1:] != g["dir"].values[:-1])) if len(g) > 1 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def sliding_windows(df):
    """
    Vectorized replacement for the original triple-nested-loop version.
    Builds the same output (one row per window, columns named '{col}_t{j}')
    using numpy stride tricks instead of ~windows*cols*WINDOW scalar .iloc
    lookups. This is the single biggest speedup in the pipeline for files
    with many seconds of traffic — the old version was O(n) Python-level
    cell accesses, this is a handful of vectorized numpy/pandas ops.
    """
    if "second" in df.columns:
        df = df.drop(columns=["second"])
    n = len(df)
    if df.empty or n < WINDOW:
        return pd.DataFrame()

    cols = list(df.columns)
    arr = df.to_numpy(dtype=np.float64)  # shape (n, n_cols)
    n_windows = n - WINDOW + 1

    # sliding_window_view over the time axis -> (n_windows, WINDOW, n_cols)
    windows_view = np.lib.stride_tricks.sliding_window_view(arr, WINDOW, axis=0)
    # sliding_window_view puts the window axis last; transpose to (n_windows, WINDOW, n_cols)
    windows_view = np.moveaxis(windows_view, -1, 1)

    out_cols = {}
    for ci, col in enumerate(cols):
        block = windows_view[:, :, ci]  # (n_windows, WINDOW)
        for j in range(WINDOW):
            out_cols[f"{col}_t{j}"] = block[:, j]

    return pd.DataFrame(out_cols)


class VPNDataset(Dataset):
    """Windowed dataset with labels, matching vvtrain.py / vvtest.py."""
    def __init__(self, df, feature_cols):
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        self.X = torch.tensor(X).view(-1, WINDOW, BASE_FEATURES)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ─────────────────────── DATASET DISCOVERY / SPLIT ────────────────────────
def list_pcaps(folder):
    files = []
    for ext in PCAP_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def stratified_file_split(vpn_dirs, nonvpn_dirs, test_frac, seed):
    """File-level split, stratified per source folder. Returns (train_list, test_list, summary),
    where each list item is (filepath, label, folder)."""
    rng = np.random.default_rng(seed)
    train_files, test_files = [], []
    summary = []

    def split_folder(folder, label):
        files = list_pcaps(folder)
        if not files:
            print(f"WARNING: no pcap files found in {folder}")
            return
        files = list(files)
        rng.shuffle(files)
        n_test = max(1, round(len(files) * test_frac))
        test_set = files[:n_test]
        train_set = files[n_test:]
        for f in train_set:
            train_files.append((f, label, folder))
        for f in test_set:
            test_files.append((f, label, folder))
        summary.append({
            "folder": folder, "label": label,
            "total_files": len(files),
            "train_files": len(train_set),
            "test_files": len(test_set),
        })

    for d in vpn_dirs:
        split_folder(d, 1)
    for d in nonvpn_dirs:
        split_folder(d, 0)

    return train_files, test_files, summary


# ─────────────────────── FEATURE CACHE BUILD ───────────────────────────────
def _process_one_file(args):
    """Worker function for parallel extraction. Must be top-level (picklable)."""
    fpath, label, max_windows = args
    try:
        raw = extract_packets(fpath)
        if raw.empty:
            return (fpath, None, "no IP packets")
        feats = compute_features(raw)
        windows = sliding_windows(feats)
        if windows.empty:
            return (fpath, None, f"too few seconds (<{WINDOW})")

        # Optional cap: keep a CONTIGUOUS slice (first N windows, in time
        # order) rather than a random sample, so every retained window is
        # still a real, temporally-contiguous 5-second capture segment and
        # the eventual time_idx renumbering stays simple and correct. This
        # exists to stop one or two very long captures (e.g. long audio
        # calls) from dominating a class's total window count.
        if max_windows is not None and len(windows) > max_windows:
            windows = windows.iloc[:max_windows].reset_index(drop=True)

        basename = os.path.basename(fpath)
        file_id = basename.replace(".", "_").replace("-", "_")
        windows["label"]    = label
        windows["file_id"]  = file_id
        windows["time_idx"] = range(len(windows))
        return (fpath, windows, "ok")
    except Exception as e:
        return (fpath, None, f"error: {e}")


def build_windowed_dataframe(file_label_list, cache_path, force_reextract, split_name,
                              n_jobs=None, max_windows_per_file=None):
    """
    Extracts features + builds sliding windows for every (filepath, label) pair,
    in parallel across processes (one PCAP per worker — PCAP parsing and the
    per-file feature/window computation are CPU-bound and fully independent
    across files, so this scales close to linearly with core count).
    Caches the concatenated result to cache_path as CSV. Adds 'file_id' column
    (sanitized basename) so windows can be traced back to source pcap.
    Returns a single concatenated DataFrame (may be empty if nothing usable).

    max_windows_per_file, if set, caps each file's window count (see
    _process_one_file for how the cap is applied). Note this only affects
    NEW extraction -- if a cache file already exists and force_reextract is
    False, the cached (potentially uncapped) data is used as-is.
    """
    if os.path.exists(cache_path) and not force_reextract:
        print(f"  [{split_name}] Using cached features: {cache_path}")
        return pd.read_csv(cache_path)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(file_label_list)) or 1

    print(f"  [{split_name}] Extracting features from {len(file_label_list)} PCAP files "
          f"using {n_jobs} parallel worker(s)...")
    if max_windows_per_file is not None:
        print(f"  [{split_name}] Capping each file at {max_windows_per_file} windows (contiguous, first-in-time)")

    work_items = [(fpath, label, max_windows_per_file) for fpath, label, _folder in file_label_list]
    all_windows = []
    skipped = []

    if n_jobs <= 1:
        for item in tqdm(work_items, desc=f"Extracting ({split_name})"):
            fpath, windows, status = _process_one_file(item)
            if status == "ok":
                all_windows.append(windows)
            else:
                skipped.append((fpath, status))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_process_one_file, item): item[0] for item in work_items}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting ({split_name})"):
                fpath, windows, status = fut.result()
                if status == "ok":
                    all_windows.append(windows)
                else:
                    skipped.append((fpath, status))

    if skipped:
        print(f"  [{split_name}] Skipped {len(skipped)} file(s):")
        for f, reason in skipped[:20]:
            print(f"    - {f}: {reason}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    if not all_windows:
        return pd.DataFrame()

    df = pd.concat(all_windows, ignore_index=True)
    df.to_csv(cache_path, index=False)
    print(f"  [{split_name}] Saved feature cache: {cache_path}  ({len(df)} window rows)")
    return df


# ─────────────────────── METRICS ───────────────────────────────────────────
def compute_and_print_metrics(y_true, y_pred, y_prob, level_name, out_dir):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    auc = None
    if y_prob is not None and len(set(y_true.tolist())) == 2:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Non-VPN", "VPN"],
        zero_division=0
    )

    print(f"\n========== {level_name} METRICS ==========")
    print(f"Samples          : {len(y_true)}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}   (weighted): {prec_w:.4f}")
    print(f"Recall    (macro): {rec_macro:.4f}   (weighted): {rec_w:.4f}")
    print(f"F1-score  (macro): {f1_macro:.4f}   (weighted): {f1_w:.4f}")
    if auc is not None:
        print(f"ROC-AUC          : {auc:.4f}")
    print("\nPer-class:")
    print(f"  Non-VPN -> Precision: {prec[0]:.4f}  Recall: {rec[0]:.4f}  F1: {f1[0]:.4f}  Support: {support[0]}")
    print(f"  VPN     -> Precision: {prec[1]:.4f}  Recall: {rec[1]:.4f}  F1: {f1[1]:.4f}  Support: {support[1]}")
    print("\nConfusion Matrix (rows=true, cols=pred) [Non-VPN, VPN]:")
    print(cm)
    print("\n" + report)

    result = {
        "level": level_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "roc_auc": float(auc) if auc is not None else None,
        "per_class": {
            "Non-VPN": {"precision": float(prec[0]), "recall": float(rec[0]),
                        "f1": float(f1[0]), "support": int(support[0])},
            "VPN":     {"precision": float(prec[1]), "recall": float(rec[1]),
                        "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"metrics_{level_name.lower().replace(' ', '_')}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {fname}")
    return result


# ─────────────────────── TRAINING ───────────────────────────────────────────
def _flow_level_f1(df_with_preds, label_col="label", pred_col="pred", group_col="file_id"):
    """Majority-vote per flow (= one pcap file, per the paper's session
    definition), then macro F1 across flows. Much less noisy than
    window-level F1 when the file count is small, and it matches the
    per-session unit the paper's file-level split is built around."""
    rows = []
    for fid, g in df_with_preds.groupby(group_col):
        true_label = int(g[label_col].iloc[0])
        vpn_pct = 100.0 * g[pred_col].sum() / len(g)
        pred_label = 1 if vpn_pct >= 50 else 0
        rows.append((true_label, pred_label))
    if not rows:
        return 0.0
    y_true = [r[0] for r in rows]
    y_pred = [r[1] for r in rows]
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def train_model(train_df, test_df, feature_cols, epochs, batch_size, num_workers,
                 model_out_path, scaler_out_path, device, lr=1e-4, weight_decay=1e-3,
                 nonvpn_weight=1.0, vpn_weight=1.0, use_focal_loss=False, focal_gamma=2.0,
                 checkpoint_metric="flow"):
    """
    Fit scaler on train, train PatchTST from scratch for the full number of
    epochs (no early stopping), saving the checkpoint with the best
    FLOW-LEVEL macro-F1 (majority vote per pcap) seen on the held-out test
    split. Flow-level F1 is used for checkpoint selection rather than
    window-level F1 because, with only a few dozen files, window-level F1 is
    noisy (thousands of correlated, overlapping windows from the same
    handful of captures), so it doesn't reliably track which epoch actually
    generalizes -- flow-level is also the metric that matters most for a
    per-capture VPN/Non-VPN verdict.
    """
    vpn_n    = int((train_df["label"] == 1).sum())
    nonvpn_n = int((train_df["label"] == 0).sum())
    total    = vpn_n + nonvpn_n
    print(f"Train windows: {total}  (VPN={vpn_n}, NonVPN={nonvpn_n})")

    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    os.makedirs(os.path.dirname(scaler_out_path), exist_ok=True)
    joblib.dump(scaler, scaler_out_path)
    print(f"Scaler saved -> {scaler_out_path}")

    # Per-window sample weighting that balances BOTH class and per-file
    # contribution. Diagnostics on this dataset showed single files
    # (e.g. one pcap = 37.6% of the test split) can dominate window counts
    # far more than class imbalance does -- a plain class-weighted loss or
    # a class-only sampler still lets one long capture's windows flood every
    # batch. Each window's weight here is (class balance weight) x
    # (1 / windows in its own file), so every FILE contributes equally to
    # training regardless of its length, and every CLASS contributes equally
    # regardless of file count.
    file_counts = train_df.groupby("file_id")["file_id"].transform("count")
    n_files_per_class = train_df.groupby("label")["file_id"].transform("nunique")
    sample_weights = (1.0 / file_counts) / n_files_per_class
    sample_weights = sample_weights.values.astype(np.float64)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    print(f"Per-file+class balanced sampler active "
          f"({train_df['file_id'].nunique()} files, weights normalize both class and file-length skew)")

    train_loader = DataLoader(VPNDataset(train_df, feature_cols),
                               batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(VPNDataset(test_df, feature_cols),
                             batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))
    print(f"Train batches: {len(train_loader)}  |  Val(test) batches: {len(val_loader)}")

    model = PatchTSTClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # One-shot cosine decay, matching vvtrain.py's original VNAT recipe
    # exactly (the recipe that reached 99.28% there) rather than the
    # warm-restarts variant tried on ISCX, which didn't outperform it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # Per-class loss weight. vvtrain.py used NO class weighting because VNAT
    # was perfectly balanced (Section IV-E of the paper). ISCX is not
    # perfectly balanced, so this stays available via --vpn_weight/
    # --nonvpn_weight, but defaults to neutral (1.0/1.0) -- same as "off",
    # matching the paper's approach unless you have a reason to deviate.
    class_weights = torch.tensor([nonvpn_weight, vpn_weight], dtype=torch.float32).to(device)
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, label_smoothing=LABEL_SMOOTHING)
        print(f"Using Focal Loss (gamma={focal_gamma}) instead of plain CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  (d_model={D_MODEL}, layers={N_LAYERS}, dropout={DROPOUT})\n")

    best_flow_f1 = -1.0
    best_window_f1 = -1.0
    best_epoch = 0
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, preds_all, labels_all = 0.0, [], []
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            preds_all.extend(torch.argmax(logits, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
            if hasattr(loop, "set_postfix"):
                loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= max(1, len(train_loader.dataset))
        train_acc = accuracy_score(labels_all, preds_all)
        train_f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X.to(device))
                val_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                val_labels.extend(y.numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1_window = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        # Flow-level F1 for model selection (one pcap = one flow/session,
        # per the paper's definition; needs file_id, which test_df carries)
        val_with_preds = test_df.copy()
        val_with_preds["pred"] = val_preds
        val_flow_f1 = _flow_level_f1(val_with_preds)

        print(f"Epoch {epoch:03d} | LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.5f} | Train F1: {train_f1:.5f} | "
              f"Val Acc: {val_acc:.5f} | Val F1(window): {val_f1_window:.5f} | Val F1(flow): {val_flow_f1:.5f}")

        # Checkpoint selection metric: 'flow' (default, one pcap = one vote,
        # matches the paper's session definition, but coarse with few test
        # files) or 'window' (finer-grained, uses more of the training
        # trajectory, but individual windows from the same file are highly
        # correlated so it can be noisier as a generalization signal -- pick
        # whichever matches what you intend to report as your headline
        # metric, since it's what gets optimized for).
        if checkpoint_metric == "window":
            improved = val_f1_window > best_window_f1
        else:
            improved = val_flow_f1 > best_flow_f1

        if improved:
            best_flow_f1 = val_flow_f1
            best_window_f1 = val_f1_window
            best_epoch = epoch
            torch.save(model.state_dict(), model_out_path)
            print(f"  Best model saved (Val {checkpoint_metric.capitalize()}-F1: "
                  f"{val_f1_window if checkpoint_metric == 'window' else val_flow_f1:.5f}, "
                  f"epoch {epoch}) -> {model_out_path}")

        gc.collect()

    print(f"\nTraining complete. Ran all {epochs} epochs. Best epoch: {best_epoch}  "
          f"Best Val Flow-F1: {best_flow_f1:.5f}  Best Val Window-F1: {best_window_f1:.5f}  "
          f"(selection metric: {checkpoint_metric})")
    return model_out_path, scaler_out_path


# ─────────────────────── DIAGNOSTICS ───────────────────────────────────────
def _run_data_diagnostics(train_df, test_df, feature_cols, out_dir):
    """
    Prints checks aimed at explaining a below-0.5 ROC-AUC / val-gets-worse-
    every-epoch pattern, which usually means train and test come from
    different underlying distributions (domain shift) rather than the model
    being undertrained. Also writes a CSV of per-feature train/test mean
    shift so you can see exactly which features disagree most.
    """
    print("\n========== DATA DIAGNOSTICS ==========")

    # 1) Per-file window counts, to catch one huge file dominating a class
    for name, df in [("train", train_df), ("test", test_df)]:
        counts = df.groupby("file_id").size()
        print(f"  [{name}] files={len(counts)}  windows/file: "
              f"min={counts.min()} max={counts.max()} mean={counts.mean():.0f}")
        top = counts.sort_values(ascending=False).head(3)
        for fid, n in top.items():
            frac = n / len(df)
            flag = "  <-- dominates >30% of split!" if frac > 0.30 else ""
            print(f"      {fid}: {n} windows ({frac:.1%} of {name} split){flag}")

    # 2) Per-class feature mean shift between train and test (standardized by
    #    train std, so a value of ~1.0 means test is shifted by one train-
    #    stdev on that feature -- a strong signal of domain shift).
    rows = []
    for label, label_name in [(1, "VPN"), (0, "Non-VPN")]:
        tr = train_df[train_df["label"] == label]
        te = test_df[test_df["label"] == label]
        if tr.empty or te.empty:
            continue
        tr_mean = tr[feature_cols].mean()
        tr_std  = tr[feature_cols].std().replace(0, 1e-6)
        te_mean = te[feature_cols].mean()
        shift = ((te_mean - tr_mean) / tr_std).abs()
        rows.append(pd.DataFrame({"feature": feature_cols, "label": label_name,
                                   "shift_in_train_stdevs": shift.values}))
    if rows:
        shift_df = pd.concat(rows, ignore_index=True)
        shift_csv = os.path.join(out_dir, "diagnostics_feature_shift.csv")
        shift_df.sort_values("shift_in_train_stdevs", ascending=False).to_csv(shift_csv, index=False)
        big_shift = shift_df[shift_df["shift_in_train_stdevs"] > 2.0]
        print(f"\n  Feature shift (test mean vs train mean, in train-stdevs) saved: {shift_csv}")
        if len(big_shift) > 0:
            worst = shift_df.sort_values("shift_in_train_stdevs", ascending=False).head(10)
            print(f"  WARNING: {len(big_shift)} (class, feature) pairs shifted >2 train-stdevs "
                  f"between train and test -- likely domain shift, not undertraining.")
            print("  Top 10 most-shifted:")
            for _, r in worst.iterrows():
                print(f"    [{r['label']:7s}] {r['feature']:<20s} shift={r['shift_in_train_stdevs']:.2f} stdevs")
        else:
            print("  No large feature shifts detected (max < 2 train-stdevs) -- "
                  "train/test look like the same distribution per class.")

    # 3) Class balance per file source folder isn't available here (folder
    #    info isn't threaded into the cached df), but window-count-per-file
    #    above plus the shift table are usually enough to spot the cause:
    #    - a few files with huge window counts vs many with few -> imbalance
    #      the sampler doesn't fix (per-window, not per-file, weighting)
    #    - large per-class feature shift -> train/test pcaps aren't really
    #      comparable (different capture length, app mix, or tool/settings)
    print("=======================================\n")


# ─────────────────────── APP/CATEGORY EXTRACTION ──────────────────────────
def extract_app_category(file_id):
    """
    Derives an app/traffic-category key from a file_id (sanitized basename),
    e.g. 'vpn_hangouts_audio2_pcap' -> 'hangouts', 'facebook_audio3_pcapng'
    -> 'facebook', 'vpn_vimeo_A_pcap' -> 'vimeo'. Used to stratify the
    train/test split by traffic type (not just by VPN/Non-VPN label), so
    every app/category present in the dataset appears on both sides of the
    split -- this is what prevents the domain-shift problem diagnosed
    earlier, where an entire app type landed only in test and the model had
    never seen anything like it during training.

    Strips a leading 'vpn_' prefix, a trailing extension token (pcap/pcapng/
    cap), and any trailing digits/letters that look like a session/variant
    index, leaving the app name itself as the category.
    """
    name = file_id.lower()
    name = re.sub(r'_pcapng$|_pcap$|_cap$', '', name)
    if name.startswith('vpn_'):
        name = name[4:]
    original = name
    # Strip one trailing variant/session token in common shapes:
    #   "_audio2" -> drops "2", "_files1b" -> drops "1b", "_A" -> drops "_a",
    #   "voipbuster1a" -> drops "1a" (no underscore before the digit)
    stripped = re.sub(r'(_?\d+[a-z]?)$', '', name)
    stripped = re.sub(r'_[a-z]$', '', stripped)
    return stripped if stripped else original


def category_stratified_file_split(file_id_to_label, test_frac, seed):
    """
    Splits files into train/test stratified by (label, app_category) pair,
    not just by label. Every app/category that has 2+ files in a class gets
    at least one file on each side of the split where possible, so no
    category is entirely absent from train or test. Categories with only 1
    file for a class go to train (can't usefully test on an app the model
    never saw a single example of during training either way, but this
    avoids a single-file category creating a train/test gap for that app).
    Returns (train_ids, test_ids, category_summary).
    """
    rng = np.random.default_rng(seed)
    by_class_category = {}
    for fid, label in file_id_to_label.items():
        cat = extract_app_category(fid)
        by_class_category.setdefault((label, cat), []).append(fid)

    train_ids, test_ids = [], []
    category_summary = []
    for (label, cat), ids in sorted(by_class_category.items()):
        ids = list(ids)
        rng.shuffle(ids)
        if len(ids) >= 2:
            n_test = max(1, round(len(ids) * test_frac))
            n_test = min(n_test, len(ids) - 1)  # always leave >=1 for train
        else:
            n_test = 0  # single-file category: keep in train, can't split further
        test_ids.extend(ids[:n_test])
        train_ids.extend(ids[n_test:])
        category_summary.append({
            "label": label, "category": cat, "total_files": len(ids),
            "train_files": len(ids) - n_test, "test_files": n_test
        })
    return train_ids, test_ids, category_summary


# ─────────────────────── MAIN ───────────────────────────────────────────────
def main():
    global D_MODEL, N_LAYERS, DROPOUT
    parser = argparse.ArgumentParser(description="Train + evaluate PatchTST VPN classifier on ISCX-style pcap dataset.")
    parser.add_argument("--vpn_dirs", nargs="+", required=True, help="Folders of VPN pcaps (label=1)")
    parser.add_argument("--nonvpn_dirs", nargs="+", required=True, help="Folders of Non-VPN pcaps (label=0)")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of files per folder held out for TEST (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (always runs the full count, no early stopping)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                         help="DataLoader workers. Keep at 0 on Windows unless you know it works.")
    parser.add_argument("--d_model", type=int, default=None, help=f"Transformer hidden dim (default {D_MODEL})")
    parser.add_argument("--n_layers", type=int, default=None, help=f"Transformer encoder layers (default {N_LAYERS})")
    parser.add_argument("--dropout", type=float, default=None, help=f"Dropout rate (default {DROPOUT})")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default 1e-4, matches vvtrain.py)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay (default 1e-4, matches vvtrain.py)")
    parser.add_argument("--nonvpn_weight", type=float, default=1.0,
                         help="Extra loss weight on the Non-VPN class on top of the balanced sampler "
                              "(default 1.0 = neutral). Check your confusion matrix: if Non-VPN recall "
                              "trails VPN recall, raise this above 1.0.")
    parser.add_argument("--vpn_weight", type=float, default=1.0,
                         help="Extra loss weight on the VPN class on top of the balanced sampler "
                              "(default 1.0 = neutral). Check your confusion matrix: if VPN recall "
                              "trails Non-VPN recall, raise this above 1.0.")
    parser.add_argument("--focal_loss", action="store_true",
                         help="Use Focal Loss instead of plain CrossEntropyLoss. Down-weights easy, "
                              "well-classified windows (e.g. long runs of near-identical quiet-audio "
                              "seconds) so gradient updates focus on ambiguous/misclassified windows. "
                              "Off by default -- compare against the CrossEntropyLoss baseline before "
                              "trusting an improvement, since results here have been noisy run-to-run.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                         help="Focal Loss focusing parameter (default 2.0, the standard value from the "
                              "original paper). Only used when --focal_loss is set. gamma=0 is equivalent "
                              "to plain (optionally class-weighted) cross-entropy.")
    parser.add_argument("--checkpoint_metric", choices=["flow", "window"], default="flow",
                         help="Which validation metric selects the best checkpoint each epoch (default "
                              "'flow'). 'flow' = macro-F1 over per-file majority votes (matches the "
                              "paper's session definition, but coarse with few test files and can lock "
                              "in an early epoch). 'window' = macro-F1 over every window (uses more of "
                              "the training trajectory, but windows within a file are highly correlated "
                              "so it isn't necessarily a better generalization signal -- pick whichever "
                              "matches the metric you intend to report as your headline number).")
    parser.add_argument("--max_windows_per_file", type=int, default=None,
                         help="Cap the number of sliding windows kept per PCAP file (contiguous slice "
                              "from the start, in time order). Use this to stop one or two very long "
                              "captures (e.g. long audio calls) from dominating a class's window count. "
                              "Applied during feature extraction, so combine with --force_reextract if "
                              "you've already cached features without this cap. Default: no cap.")
    parser.add_argument("--out_dir", default="./results", help="Where to write model, scaler, metrics, caches")
    parser.add_argument("--force_reextract", action="store_true", help="Ignore feature cache and re-parse all PCAPs")
    parser.add_argument("--balance_files", action="store_true",
                         help="Randomly subsample the majority class's FILE count down to match the "
                              "minority class before splitting, so both train and test are file-count "
                              "balanced (not just balanced via the training sampler). Uses only real, "
                              "unmodified files -- just fewer of whichever class has more.")
    parser.add_argument("--jobs", type=int, default=None,
                         help="Parallel worker processes for PCAP feature extraction (default: CPU count - 1)")
    parser.add_argument("--skip_train", action="store_true",
                         help="Skip training; evaluate an existing --model/--scaler on the test split instead")
    parser.add_argument("--model", default=None, help="Existing model path (only used with --skip_train)")
    parser.add_argument("--scaler", default=None, help="Existing scaler path (only used with --skip_train)")
    args = parser.parse_args()

    # Allow CLI overrides of architecture/regularization without touching the class definition.
    if args.d_model is not None:
        D_MODEL = args.d_model
    if args.n_layers is not None:
        N_LAYERS = args.n_layers
    if args.dropout is not None:
        DROPOUT = args.dropout

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    models_dir = os.path.join(args.out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ── 1. Extract features from ALL files first, into one combined table ──
    print("\n========== FEATURE EXTRACTION (all files, before split) ==========")
    all_files = []
    for d in args.vpn_dirs:
        all_files.extend((f, 1, d) for f in list_pcaps(d))
    for d in args.nonvpn_dirs:
        all_files.extend((f, 0, d) for f in list_pcaps(d))
    if not all_files:
        print("ERROR: no pcap files found in the given --vpn_dirs / --nonvpn_dirs.")
        sys.exit(1)
    print(f"Found {len(all_files)} PCAP files total "
          f"(VPN={sum(1 for _,l,_ in all_files if l==1)}, Non-VPN={sum(1 for _,l,_ in all_files if l==0)})")

    all_cache = os.path.join(args.out_dir, "all_features_cache.csv")
    all_df = build_windowed_dataframe(all_files, all_cache, args.force_reextract, "all",
                                       n_jobs=args.jobs, max_windows_per_file=args.max_windows_per_file)
    if all_df.empty:
        print("ERROR: feature table is empty after extraction (too many unusable PCAPs?).")
        sys.exit(1)
    all_df = all_df.fillna(0)
    all_df["label"] = all_df["label"].astype(int)

    feature_cols = [c for c in all_df.columns if c not in EXCLUDE]
    expected = WINDOW * BASE_FEATURES
    if len(feature_cols) != expected:
        print(f"WARNING: expected {WINDOW}x{BASE_FEATURES}={expected} feature columns, got {len(feature_cols)}.")
        print(f"         Adjust BASE_FEATURES in CONFIG to {len(feature_cols)//WINDOW} if this is unexpected.")
    print(f"Total windows: {len(all_df)}  |  Feature columns: {len(feature_cols)}")

    # ── 2. Split the FEATURE TABLE 80:20, stratified by (label, app_category) ──
    # Matches split.py's guarantee that every file-group is represented on
    # both sides of the split, extended here to app/traffic category as well
    # as VPN/Non-VPN label: diagnostics showed an entire app-category (a set
    # of Non-VPN files with a different capture tool/protocol) landing only
    # in test in an earlier run, causing a 3-5 train-stdev feature shift and
    # a model that looked great on train but failed on that unseen category.
    # Stratifying by category (not just class) keeps every app represented
    # in both train and test wherever it has 2+ files.
    file_id_to_label = all_df.drop_duplicates("file_id").set_index("file_id")["label"].to_dict()

    if args.balance_files:
        # Genuine dataset balancing: randomly subsample the majority class's
        # FILE count down to match the minority class, before splitting.
        rng = np.random.default_rng(args.seed)
        file_ids_by_class = {1: [], 0: []}
        for fid, label in file_id_to_label.items():
            file_ids_by_class[label].append(fid)
        n_vpn, n_nonvpn = len(file_ids_by_class[1]), len(file_ids_by_class[0])
        target = min(n_vpn, n_nonvpn)
        print(f"\n--balance_files active: subsampling both classes to {target} files "
              f"(had VPN={n_vpn}, Non-VPN={n_nonvpn})")
        for label in (1, 0):
            ids = list(file_ids_by_class[label])
            rng.shuffle(ids)
            file_ids_by_class[label] = ids[:target]
        kept_ids = set(file_ids_by_class[1]) | set(file_ids_by_class[0])
        file_id_to_label = {fid: label for fid, label in file_id_to_label.items() if fid in kept_ids}

    train_ids, test_ids, category_summary = category_stratified_file_split(
        file_id_to_label, args.test_frac, args.seed
    )

    print("\n========== SPLIT SUMMARY (80:20, stratified by label AND app/traffic category) ==========")
    label_totals = {1: {"train": 0, "test": 0}, 0: {"train": 0, "test": 0}}
    for s in category_summary:
        label_totals[s["label"]]["train"] += s["train_files"]
        label_totals[s["label"]]["test"] += s["test_files"]
        flag = "  <-- single-file category, kept in train only" if s["test_files"] == 0 and s["total_files"] == 1 else ""
        print(f"  [{'VPN' if s['label']==1 else 'Non-VPN':<7s}] {s['category']:<20s} "
              f"total={s['total_files']:>3}  train={s['train_files']:>3}  test={s['test_files']:>3}{flag}")
    print(f"  ---")
    for label, name in [(1, "VPN"), (0, "Non-VPN")]:
        print(f"  {name} totals: train={label_totals[label]['train']}  test={label_totals[label]['test']}")

    train_df = all_df[all_df["file_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = all_df[all_df["file_id"].isin(test_ids)].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        print("ERROR: train or test split is empty after splitting the feature table. Check --test_frac.")
        sys.exit(1)

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    print(f"\nTrain windows: {len(train_df)}  |  Test windows: {len(test_df)}")

    _run_data_diagnostics(train_df, test_df, feature_cols, args.out_dir)

    model_path  = os.path.join(models_dir, "patchtst_model.pt")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # ── 3. Train (or load existing) ──
    if args.skip_train:
        if not args.model or not args.scaler:
            print("ERROR: --skip_train requires --model and --scaler.")
            sys.exit(1)
        model_path, scaler_path = args.model, args.scaler
        scaler = joblib.load(scaler_path)
        test_df_scaled = test_df.copy()
        cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else feature_cols
        for c in cols:
            if c not in test_df_scaled.columns:
                test_df_scaled[c] = 0.0
        test_df_scaled[cols] = scaler.transform(test_df_scaled[cols])
        feature_cols = cols
        eval_df = test_df_scaled
        model = PatchTSTClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("\n========== TRAINING (80% split, from scratch) ==========")
        train_model(
            train_df, test_df, feature_cols,
            epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
            model_out_path=model_path, scaler_out_path=scaler_path, device=device,
            lr=args.lr, weight_decay=args.weight_decay,
            nonvpn_weight=args.nonvpn_weight, vpn_weight=args.vpn_weight,
            use_focal_loss=args.focal_loss, focal_gamma=args.focal_gamma,
            checkpoint_metric=args.checkpoint_metric
        )
        # Reload best checkpoint + fitted scaler for a clean eval pass
        scaler = joblib.load(scaler_path)
        eval_df = test_df.copy()
        eval_df[feature_cols] = scaler.transform(eval_df[feature_cols])
        model = PatchTSTClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # ── 4. Evaluate on TEST split (window level) ──
    print("\n========== EVALUATION (20% held-out test split) ==========")
    loader = DataLoader(VPNDataset(eval_df, feature_cols),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Inference"):
            logits = model(X.to(device))
            probs = F.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(probs, 1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    compute_and_print_metrics(all_labels, all_preds, all_probs, "Window Level", args.out_dir)

    # Warn if window-level metrics are dominated by one or two files, since
    # that changes how those numbers should be interpreted/reported.
    eval_counts = eval_df.groupby("file_id").size().sort_values(ascending=False)
    top_share = eval_counts.iloc[0] / len(eval_df)
    if top_share > 0.25:
        print(f"NOTE: window-level metrics above are dominated by a single file "
              f"('{eval_counts.index[0]}', {top_share:.1%} of test windows). "
              f"With only {eval_df['file_id'].nunique()} test files, window-level numbers "
              f"mostly reflect that one capture's characteristics rather than a "
              f"representative sample -- treat Flow Level metrics below as more reliable, "
              f"and consider more test files or k-fold CV for a stable reported number.")

    # ── 5. Flow-level metrics ──
    # Following the paper's convention: each PCAP file corresponds to one
    # session/flow (Section III: "Each capture file corresponds to one
    # session and is labeled VPN or non-VPN"). A flow-level verdict is the
    # majority vote across that file's windows, matching the file-level
    # split unit used throughout the paper's methodology (no true 5-tuple
    # sub-flow extraction is used there either).
    eval_df_with_preds = eval_df.copy()
    eval_df_with_preds["pred"] = all_preds
    eval_df_with_preds["prob"] = all_probs
    flow_rows = []
    for file_id, g in eval_df_with_preds.groupby("file_id"):
        true_label = int(g["label"].iloc[0])
        vpn_pct = 100.0 * g["pred"].sum() / len(g)
        avg_prob = float(g["prob"].mean())
        pred_label = 1 if vpn_pct >= 50 else 0
        flow_rows.append({
            "flow_id": file_id, "true_label": true_label, "pred_label": pred_label,
            "n_windows": len(g), "vpn_window_pct": vpn_pct, "avg_vpn_prob": avg_prob * 100
        })
    flow_df = pd.DataFrame(flow_rows)
    flow_csv = os.path.join(args.out_dir, "per_flow_test_results.csv")
    flow_df.to_csv(flow_csv, index=False)
    print(f"\nSaved per-flow test results: {flow_csv}")

    if len(flow_df) > 0:
        compute_and_print_metrics(
            flow_df["true_label"].tolist(),
            flow_df["pred_label"].tolist(),
            flow_df["avg_vpn_prob"].div(100.0).tolist(),
            "Flow Level", args.out_dir
        )

    print("\nDone.")
    print(f"Model  -> {model_path}")
    print(f"Scaler -> {scaler_path}")
    print(f"All results in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    mp.freeze_support()  # required for multiprocessing to work correctly on Windows
    main()
