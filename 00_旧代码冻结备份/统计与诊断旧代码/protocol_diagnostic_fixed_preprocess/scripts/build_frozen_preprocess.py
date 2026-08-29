# -*- coding: utf-8 -*-
r"""
Builds the PREPROCESS_SEED=42 frozen preprocessing artifacts for the
fixed-preprocessing / training-seed-isolation diagnostic protocol.

Reuses `代码/main_experiment_3_fgds_psi_optimized.py` (== `base`) as a
READ-ONLY library, in the exact same way `代码/7.4对比实验.py` and
`baselines/htt_net/train.py` already do. Nothing in `代码/` is modified.
This script only ever runs ONCE (PREPROCESS_SEED is always 42) and its
outputs are then loaded, never recomputed, by every TRAIN_SEED run.

Usage:
    python build_frozen_preprocess.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DIAG_ROOT = THIS_DIR.parent
PROJECT_ROOT = DIAG_ROOT.parent
CODE_DIR = PROJECT_ROOT / "代码"
FROZEN_DIR = DIAG_ROOT / "frozen_preprocess"
FROZEN_DIR.mkdir(parents=True, exist_ok=True)

AUTHORITATIVE_FEATURE_FILE = PROJECT_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"
PREPROCESS_SEED = 42

# Redirect base's own disposable side-output directory (it writes some
# diagnostic CSVs as a side effect of some functions) so it never touches
# the manuscript's original output folder or the synced project tree.
os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "protocol_diagnostic_base_side_outputs"))

sys.path.insert(0, str(CODE_DIR))
import main_experiment_3_fgds_psi_optimized as base  # noqa: E402

base.FEATURE_FILE = AUTHORITATIVE_FEATURE_FILE
base.RANDOM_SEED = PREPROCESS_SEED


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    if not AUTHORITATIVE_FEATURE_FILE.exists():
        raise FileNotFoundError(f"Authoritative feature file missing: {AUTHORITATIVE_FEATURE_FILE}")

    print("=" * 100)
    print("BUILDING FROZEN PREPROCESSING ARTIFACTS (PREPROCESS_SEED=42)")
    print("=" * 100)

    # ---- 3.1 Authoritative raw feature table -----------------------------
    raw_bytes = AUTHORITATIVE_FEATURE_FILE.read_bytes()
    raw_sha256 = sha256_bytes(raw_bytes)
    raw_df_check = pd.read_csv(AUTHORITATIVE_FEATURE_FILE)
    cond_col = None
    for c in raw_df_check.columns:
        if str(c).strip().lower() == "condition":
            cond_col = c
            break
    cond_counts = raw_df_check[cond_col].astype(str).str.strip().str.upper().value_counts().to_dict() if cond_col else {}
    with open(FROZEN_DIR / "raw_feature_sha256.txt", "w", encoding="utf-8") as f:
        f.write(f"file: {AUTHORITATIVE_FEATURE_FILE}\n")
        f.write(f"sha256: {raw_sha256}\n")
        f.write(f"row_count: {len(raw_df_check)}\n")
        f.write(f"column_count: {len(raw_df_check.columns)}\n")
        for k, v in cond_counts.items():
            f.write(f"condition_count[{k}]: {v}\n")
    print(f"[3.1] raw feature file sha256={raw_sha256[:16]}... rows={len(raw_df_check)} cols={len(raw_df_check.columns)} conds={cond_counts}")

    # ---- Reproduce the exact preprocessing sequence from 7.4对比实验.py ---
    raw_df = base.load_feature_table()
    label_df, stage_thresholds = base.define_condition_relative_stages(raw_df)

    # ---- 3.2 Stage labels --------------------------------------------------
    stage_cols = ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true"]
    stage_labels_df = label_df[stage_cols].copy()
    stage_labels_df.to_csv(FROZEN_DIR / "stage_labels.csv", index=False, encoding="utf-8-sig")
    stage_thresholds.to_csv(FROZEN_DIR / "stage_thresholds.csv", index=False, encoding="utf-8-sig")
    print(f"[3.2] stage_labels.csv rows={len(stage_labels_df)} (Q_EARLY={base.Q_EARLY}, Q_LATE={base.Q_LATE}, RATE_LATE_Q={base.RATE_LATE_Q})")

    final_train_raw, final_val_raw, test_c6_raw = base.split_grouped_lifecycle(label_df)

    # ---- 3.3 Train / validation / test split -------------------------------
    split_rows = []
    for split_name, sub in [("train", final_train_raw), ("val", final_val_raw), ("test", test_c6_raw)]:
        for _, r in sub.iterrows():
            split_rows.append({
                "condition": r["condition"], "run_id": int(r["run_id"]),
                "stage": r["stage"], "split": split_name,
            })
    split_manifest = pd.DataFrame(split_rows).sort_values(["split", "condition", "run_id"]).reset_index(drop=True)
    split_manifest.to_csv(FROZEN_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    split_counts = split_manifest.groupby(["split", "condition"]).size().to_dict()
    print(f"[3.3] split_manifest.csv rows={len(split_manifest)} counts={split_counts}")

    # ---- Split-safe online features -----------------------------------------
    raw_cols = base.get_raw_numeric_sensor_cols(final_train_raw)
    split_feat = base.build_online_features_by_split({
        "final_train": final_train_raw,
        "final_internal_val": final_val_raw,
        "test_C6": test_c6_raw,
    }, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "final_train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "final_internal_val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test_C6"].copy()

    all_cols = base.feature_cols_from(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, all_cols)

    # ---- 3.4 Selected feature list (train-only MI + redundancy, seed=42) ----
    selected, selected_df = base.select_features_train_only(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, selected)

    with open(FROZEN_DIR / "selected_features_seed42.json", "w", encoding="utf-8") as f:
        json.dump({
            "preprocess_seed": PREPROCESS_SEED,
            "n_selected_features": len(selected),
            "selected_features_in_order": selected,
        }, f, ensure_ascii=False, indent=2)
    selected_df.to_csv(FROZEN_DIR / "selected_features_scored_seed42.csv", index=False, encoding="utf-8-sig")
    print(f"[3.4] selected {len(selected)} features (target N_SELECTED_FEATURES={base.N_SELECTED_FEATURES})")

    # ---- 3.6 GMM fine-state definition (fit once on feat_train, seed=42) ----
    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = base.assign_fine_states(feat_test, gmm, raw_to_order)

    with open(FROZEN_DIR / "gmm_seed42.pkl", "wb") as f:
        pickle.dump({"gmm": gmm, "raw_to_order": raw_to_order}, f)

    fine_rows = []
    for split_name, df in [("train", feat_train), ("val", feat_val), ("test", feat_test)]:
        for _, r in df.iterrows():
            fine_rows.append({
                "condition": r["condition"], "run_id": int(r["run_id"]), "split": split_name,
                "q_true": float(r["q_true"]), "rate_norm": float(r["rate_norm"]),
                "fine_state": int(r["fine_state_true"]),
            })
    fine_state_df = pd.DataFrame(fine_rows).sort_values(["split", "condition", "run_id"]).reset_index(drop=True)
    fine_state_df.to_csv(FROZEN_DIR / "fine_state_labels_seed42.csv", index=False, encoding="utf-8-sig")
    print(f"[3.6] GMM fit on {len(feat_train)} train rows; fine_state_labels_seed42.csv rows={len(fine_state_df)}")

    # ---- 3.5 StandardScaler (fit once on feat_train[selected]) --------------
    scaler = base.StandardScaler().fit(feat_train[selected].values)
    np.save(FROZEN_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(FROZEN_DIR / "scaler_scale.npy", scaler.scale_)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[3.5] StandardScaler fit on train[selected] shape={feat_train[selected].shape}")

    # Persist the fully-built, scaled feature tables themselves too, so
    # TRAIN_SEED runs never need to recompute ANY preprocessing step -- they
    # can just load these CSVs + selected_features_seed42.json directly.
    feat_train.to_csv(FROZEN_DIR / "feat_train_frozen.csv", index=False, encoding="utf-8-sig")
    feat_val.to_csv(FROZEN_DIR / "feat_val_frozen.csv", index=False, encoding="utf-8-sig")
    feat_test.to_csv(FROZEN_DIR / "feat_test_frozen.csv", index=False, encoding="utf-8-sig")

    # ---- 3.8 Sequence windows (L=12) -----------------------------------------
    L = base.BEST_ARCH["L"]
    window_rows = []
    for split_name, df_sub in [("train", feat_train), ("val", feat_val), ("test", feat_test)]:
        X, ys, yf, yq, meta = base.build_windows(df_sub, selected, L, split_name)
        for _, r in meta.iterrows():
            window_rows.append({
                "condition": r["condition"],
                "window_start": int(r["cut_index"]) - L + 1,
                "window_end": int(r["cut_index"]),
                "run_id_end": int(r["cut_index"]),
                "stage": r["stage_true"],
                "fine_state": int(r["fine_state_true"]),
                "q_target": float(r["q_true"]),
                "split": split_name,
            })
    window_manifest = pd.DataFrame(window_rows).sort_values(["split", "condition", "run_id_end"]).reset_index(drop=True)
    window_manifest.to_csv(FROZEN_DIR / "window_manifest.csv", index=False, encoding="utf-8-sig")
    n_test_windows = int((window_manifest["split"] == "test").sum())
    test_run_range = window_manifest[window_manifest["split"] == "test"]["run_id_end"]
    print(f"[3.8] window_manifest.csv rows={len(window_manifest)} (L={L})")
    print(f"      C6 test universe: {n_test_windows} windows, run_id_end {test_run_range.min()}-{test_run_range.max()}")

    # ---- manifest_hashes.json -------------------------------------------------
    artifact_files = [
        "raw_feature_sha256.txt", "stage_labels.csv", "stage_thresholds.csv",
        "split_manifest.csv", "selected_features_seed42.json", "selected_features_scored_seed42.csv",
        "gmm_seed42.pkl", "fine_state_labels_seed42.csv", "scaler_mean.npy", "scaler_scale.npy",
        "feat_train_frozen.csv", "feat_val_frozen.csv", "feat_test_frozen.csv", "window_manifest.csv",
    ]
    hashes = {"preprocess_seed": PREPROCESS_SEED, "raw_feature_file_sha256": raw_sha256, "files": {}}
    for name in artifact_files:
        p = FROZEN_DIR / name
        hashes["files"][name] = sha256_file(p)
    with open(FROZEN_DIR / "manifest_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print("=" * 100)
    print("FROZEN PREPROCESSING ARTIFACTS COMPLETE")
    print(f"  split_manifest sha256:       {hashes['files']['split_manifest.csv'][:16]}...")
    print(f"  selected_features sha256:    {hashes['files']['selected_features_seed42.json'][:16]}...")
    print(f"  gmm sha256:                  {hashes['files']['gmm_seed42.pkl'][:16]}...")
    print(f"  window_manifest sha256:      {hashes['files']['window_manifest.csv'][:16]}...")
    print("=" * 100)


if __name__ == "__main__":
    main()
