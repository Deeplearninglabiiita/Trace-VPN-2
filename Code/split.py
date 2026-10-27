#!/usr/bin/env python3
"""
split.py  —  Split vnat_timeseries.csv → train.csv + test.csv

Critical design decisions:
  1. file_id  = original per-file unique ID — used ONLY for group-aware splitting
  2. group_id = "vpn" or "nonvpn"          — 2 fixed values for TFT vocabulary
  3. time_idx = globally unique within each group_id, NON-OVERLAPPING between
                train and test — prevents TFT from seeing val samples in encoder
                context during training. Train gets 0..N-1, test gets N..N+M-1.
"""

import pandas as pd
import numpy as np

# ──────────────── CONFIG ────────────────
INPUT_CSV  = "/home/ashok/agrim/dataset/take2/vnat_timeseries.csv"
TRAIN_CSV  = "/home/ashok/agrim/dataset/take2/train.csv"
TEST_CSV   = "/home/ashok/agrim/dataset/take2/test.csv"
TEST_RATIO = 0.20
SEED       = 42

def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV).fillna(0)
    print(f"Total rows: {len(df)}")

    if "group_id" not in df.columns:
        raise ValueError("group_id column missing — re-run datap2.py first.")

    # Keep original per-file ID for splitting only
    df = df.rename(columns={"group_id": "file_id"})

    # TFT-safe group_id: exactly 2 known values in every split
    df["group_id"] = df["label"].map({1: "vpn", 0: "nonvpn"})

    # ── file-level split (no file straddles train and test) ──
    files = df["file_id"].unique()
    rng   = np.random.default_rng(SEED)
    rng.shuffle(files)

    n_test      = max(1, int(len(files) * TEST_RATIO))
    test_files  = set(files[:n_test])
    train_files = set(files[n_test:])

    train_df = df[df["file_id"].isin(train_files)].copy().reset_index(drop=True)
    test_df  = df[df["file_id"].isin(test_files)].copy().reset_index(drop=True)

    # ── NON-OVERLAPPING time_idx (the critical fix) ──────────
    # For each group_id ("vpn" / "nonvpn"):
    #   train gets  0 .. train_count-1
    #   test  gets  train_count .. train_count+test_count-1
    # This guarantees test indices are always > any train index,
    # so TFT's encoder window can never reach into the test set.
    train_df["time_idx"] = train_df.groupby("group_id").cumcount()

    # Count how many train rows each group has — test starts right after
    train_group_counts = train_df.groupby("group_id")["time_idx"].max() + 1  # {vpn: N, nonvpn: M}

    def assign_test_time_idx(group):
        gid    = group["group_id"].iloc[0]
        offset = int(train_group_counts.get(gid, 0))
        group  = group.copy()
        group["time_idx"] = range(offset, offset + len(group))
        return group

    test_df = test_df.groupby("group_id", group_keys=False).apply(assign_test_time_idx)
    test_df = test_df.reset_index(drop=True)

    # ── verify no overlap ────────────────────────────────────
    for gid in ["vpn", "nonvpn"]:
        tr_idx = set(train_df[train_df["group_id"] == gid]["time_idx"].tolist())
        te_idx = set(test_df[test_df["group_id"]   == gid]["time_idx"].tolist())
        overlap = tr_idx & te_idx
        if overlap:
            raise RuntimeError(f"time_idx overlap in group '{gid}': {len(overlap)} shared indices!")
    print("time_idx overlap check: PASSED (no overlap between train and test)")

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV,   index=False)

    print(f"\nSplit complete")
    print(f"  Train files : {len(train_files):>4}   rows: {len(train_df)}")
    print(f"  Test  files : {len(test_files):>4}   rows: {len(test_df)}")
    print(f"  Train VPN   : {(train_df['label']==1).sum()}   NonVPN: {(train_df['label']==0).sum()}")
    print(f"  Test  VPN   : {(test_df['label']==1).sum()}    NonVPN: {(test_df['label']==0).sum()}")

    # Show time_idx ranges for sanity check
    for gid in ["vpn", "nonvpn"]:
        tr = train_df[train_df["group_id"] == gid]["time_idx"]
        te = test_df[test_df["group_id"]   == gid]["time_idx"]
        print(f"  [{gid}] train time_idx: {tr.min()}..{tr.max()}  |  test time_idx: {te.min()}..{te.max()}")

    print(f"\nSaved: {TRAIN_CSV}")
    print(f"Saved: {TEST_CSV}")

if __name__ == "__main__":
    main()
