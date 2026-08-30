# -*- coding: utf-8 -*-
r"""
Builds task-specific PREPROCESS_SEED=42 frozen preprocessing artifacts for a
PHM2010 cross-cutter task (D2 or D3), mirroring exactly what
`protocol_diagnostic_fixed_preprocess/scripts/build_frozen_preprocess.py` did
for D1 -- except the train/test split is source-only per-task (never D1's
C1+C4->C6), using the already-validated generalized split logic:

  `final_statistical_evidence/scripts/methods/common_pipeline.py::split_by_conditions`
  (itself verbatim from `代码/7.7跨工况实验.py::split_train_val_by_conditions`,
  the reference implementation already used for the real D2/D3 runs in
  final_statistical_evidence/). That function raises ValueError on any
  train/test condition overlap -- the leakage guard is structural, not just
  documented.

Both `代码/main_experiment_3_fgds_psi_optimized.py` and
`final_statistical_evidence/scripts/methods/common_pipeline.py` are imported
READ-ONLY as libraries. Nothing in either original tree is modified.

Usage:
    python build_phm_task_frozen_preprocess.py --task D2 --train C1,C6 --test C4
    python build_phm_task_frozen_preprocess.py --task D3 --train C4,C6 --test C1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent                    # .../shared/reproducibility
SHARED_DIR = THIS_DIR.parent                                    # .../shared
EXPAND_ROOT = SHARED_DIR.parent                                 # .../扩充实验代码
PROJECT_ROOT = EXPAND_ROOT.parent                                # repo root (论文/)
CODE_DIR = PROJECT_ROOT / "代码"
COMMON_PIPELINE_PATH = PROJECT_ROOT / "final_statistical_evidence" / "scripts" / "methods" / "common_pipeline.py"

PREPROCESS_SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def import_common_pipeline():
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "phm_task_frozen_preprocess_side_outputs"))
    spec = importlib.util.spec_from_file_location("common_pipeline_frozen_build", COMMON_PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["D2", "D3"])
    ap.add_argument("--train", required=True, help="comma-separated train conditions, e.g. C1,C6")
    ap.add_argument("--test", required=True, help="single test condition, e.g. C4")
    args = ap.parse_args()

    train_conditions = [c.strip() for c in args.train.split(",")]
    test_condition = args.test.strip()

    if test_condition in train_conditions:
        raise ValueError(f"LEAKAGE: test condition {test_condition} is also in train conditions {train_conditions}")

    FROZEN_DIR = EXPAND_ROOT / "shared" / "reproducibility" / f"PHM2010_{args.task}_frozen_preprocess"
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"BUILDING FROZEN PREPROCESSING ARTIFACTS FOR PHM2010 {args.task} "
          f"(train={train_conditions}, test={test_condition}, PREPROCESS_SEED={PREPROCESS_SEED})")
    print("=" * 100)

    cp = import_common_pipeline()
    base = cp.import_base()
    base.RANDOM_SEED = PREPROCESS_SEED

    authoritative_feature_file = base.FEATURE_FILE
    if not Path(authoritative_feature_file).exists():
        raise FileNotFoundError(f"Authoritative feature file missing: {authoritative_feature_file}")
    raw_bytes = Path(authoritative_feature_file).read_bytes()
    raw_sha256 = sha256_bytes(raw_bytes)
    raw_df_check = pd.read_csv(authoritative_feature_file)
    with open(FROZEN_DIR / "raw_feature_sha256.txt", "w", encoding="utf-8") as f:
        f.write(f"file: {authoritative_feature_file}\n")
        f.write(f"sha256: {raw_sha256}\n")
        f.write(f"row_count: {len(raw_df_check)}\n")
    print(f"[1] raw feature file sha256={raw_sha256[:16]}... rows={len(raw_df_check)}")

    # ---- label df + stage thresholds (keep both, unlike common_pipeline.load_label_df which discards thresholds)
    raw_df = base.load_feature_table()
    label_df, stage_thresholds = base.define_condition_relative_stages(raw_df)

    stage_cols = ["condition", "run_id", "VB", "VB_smooth", "q_true", "rate_norm", "stage", "stage_id", "fine_state_true"]
    label_df[stage_cols].to_csv(FROZEN_DIR / "stage_labels.csv", index=False, encoding="utf-8-sig")
    stage_thresholds.to_csv(FROZEN_DIR / "stage_thresholds.csv", index=False, encoding="utf-8-sig")

    # ---- Leakage-checked, source-only split (verbatim reference logic) ----
    train_raw, val_raw, test_raw = cp.split_by_conditions(base, label_df, train_conditions, test_condition)
    assert set(train_raw["condition"].unique()) <= set(train_conditions), "train_raw contains unexpected condition"
    assert set(val_raw["condition"].unique()) <= set(train_conditions), "val_raw (source-internal) contains unexpected condition"
    assert set(test_raw["condition"].unique()) == {test_condition}, "test_raw must be exactly the held-out condition"
    print(f"[2] split OK: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)} "
          f"(conditions train={sorted(train_raw['condition'].unique())} val={sorted(val_raw['condition'].unique())} "
          f"test={sorted(test_raw['condition'].unique())})")

    split_rows = []
    for split_name, sub in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        for _, r in sub.iterrows():
            split_rows.append({"condition": r["condition"], "run_id": int(r["run_id"]), "stage": r["stage"], "split": split_name})
    split_manifest = pd.DataFrame(split_rows).sort_values(["split", "condition", "run_id"]).reset_index(drop=True)
    split_manifest.to_csv(FROZEN_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[3] split_manifest.csv rows={len(split_manifest)} counts={split_manifest.groupby(['split', 'condition']).size().to_dict()}")

    # ---- Split-safe online features (train-only fit) ----
    raw_cols = base.get_raw_numeric_sensor_cols(train_raw)
    split_feat = base.build_online_features_by_split({"train": train_raw, "val": val_raw, "test": test_raw}, raw_cols)
    feat_train = split_feat[split_feat["split_name_for_feature_build"] == "train"].copy()
    feat_val = split_feat[split_feat["split_name_for_feature_build"] == "val"].copy()
    feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test"].copy()

    all_cols = base.feature_cols_from(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, all_cols)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, all_cols)

    selected, selected_df = base.select_features_train_only(feat_train)
    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, selected)

    with open(FROZEN_DIR / "selected_features_seed42.json", "w", encoding="utf-8") as f:
        json.dump({"preprocess_seed": PREPROCESS_SEED, "task": args.task,
                    "train_conditions": train_conditions, "test_condition": test_condition,
                    "n_selected_features": len(selected), "selected_features_in_order": selected}, f, ensure_ascii=False, indent=2)
    selected_df.to_csv(FROZEN_DIR / "selected_features_scored_seed42.csv", index=False, encoding="utf-8-sig")
    print(f"[4] selected {len(selected)} features (train-only, {args.task})")

    # ---- GMM fine-state (fit on train only) ----
    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = base.assign_fine_states(feat_test, gmm, raw_to_order)
    with open(FROZEN_DIR / "gmm_seed42.pkl", "wb") as f:
        pickle.dump({"gmm": gmm, "raw_to_order": raw_to_order}, f)

    fine_rows = []
    for split_name, df in [("train", feat_train), ("val", feat_val), ("test", feat_test)]:
        for _, r in df.iterrows():
            fine_rows.append({"condition": r["condition"], "run_id": int(r["run_id"]), "split": split_name,
                               "q_true": float(r["q_true"]), "rate_norm": float(r["rate_norm"]), "fine_state": int(r["fine_state_true"])})
    pd.DataFrame(fine_rows).sort_values(["split", "condition", "run_id"]).reset_index(drop=True).to_csv(
        FROZEN_DIR / "fine_state_labels_seed42.csv", index=False, encoding="utf-8-sig")
    print(f"[5] GMM fit on {len(feat_train)} train rows")

    # ---- StandardScaler (fit on train only) ----
    scaler = base.StandardScaler().fit(feat_train[selected].values)
    np.save(FROZEN_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(FROZEN_DIR / "scaler_scale.npy", scaler.scale_)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[6] StandardScaler fit on train[selected] shape={feat_train[selected].shape}")

    feat_train.to_csv(FROZEN_DIR / "feat_train_frozen.csv", index=False, encoding="utf-8-sig")
    feat_val.to_csv(FROZEN_DIR / "feat_val_frozen.csv", index=False, encoding="utf-8-sig")
    feat_test.to_csv(FROZEN_DIR / "feat_test_frozen.csv", index=False, encoding="utf-8-sig")

    # ---- Windows (L=12) -- evaluation universe verified from actual data, never hardcoded ----
    L = base.BEST_ARCH["L"]
    window_rows = []
    for split_name, df_sub in [("train", feat_train), ("val", feat_val), ("test", feat_test)]:
        X, ys, yf, yq, meta = base.build_windows(df_sub, selected, L, split_name)
        for _, r in meta.iterrows():
            window_rows.append({"condition": r["condition"], "window_start": int(r["cut_index"]) - L + 1,
                                 "window_end": int(r["cut_index"]), "run_id_end": int(r["cut_index"]),
                                 "stage": r["stage_true"], "fine_state": int(r["fine_state_true"]),
                                 "q_target": float(r["q_true"]), "split": split_name})
    window_manifest = pd.DataFrame(window_rows).sort_values(["split", "condition", "run_id_end"]).reset_index(drop=True)
    window_manifest.to_csv(FROZEN_DIR / "window_manifest.csv", index=False, encoding="utf-8-sig")
    n_test_windows = int((window_manifest["split"] == "test").sum())
    test_run_range = window_manifest[window_manifest["split"] == "test"]["run_id_end"]
    print(f"[7] window_manifest.csv rows={len(window_manifest)} (L={L})")
    print(f"    {test_condition} test universe: {n_test_windows} windows, run_id_end {test_run_range.min()}-{test_run_range.max()}")

    # ---- Structural leakage re-verification (post-hoc, on the actual frozen artifacts) ----
    leak_report = {
        "task": args.task, "train_conditions": train_conditions, "test_condition": test_condition,
        "test_condition_in_train_split": bool(test_condition in set(split_manifest[split_manifest.split == "train"]["condition"])),
        "test_condition_in_val_split": bool(test_condition in set(split_manifest[split_manifest.split == "val"]["condition"])),
        "feature_selection_fit_on": "train only (base.select_features_train_only(feat_train))",
        "gmm_fit_on": "train only (base.fit_train_gmm(feat_train))",
        "scaler_fit_on": "train only (base.StandardScaler().fit(feat_train[selected]))",
        "n_test_windows": n_test_windows, "test_run_id_end_range": [int(test_run_range.min()), int(test_run_range.max())],
        "leakage_detected": bool(test_condition in set(split_manifest[split_manifest.split != "test"]["condition"])),
    }
    with open(FROZEN_DIR / "leakage_check.json", "w", encoding="utf-8") as f:
        json.dump(leak_report, f, indent=2)
    if leak_report["leakage_detected"]:
        raise RuntimeError(f"PROTOCOL_FAILED: {test_condition} leaked into train/val split for {args.task}")
    print(f"[8] leakage_check.json: leakage_detected={leak_report['leakage_detected']}")

    # ---- manifest_hashes.json ----
    artifact_files = ["raw_feature_sha256.txt", "stage_labels.csv", "stage_thresholds.csv", "split_manifest.csv",
                       "selected_features_seed42.json", "selected_features_scored_seed42.csv", "gmm_seed42.pkl",
                       "fine_state_labels_seed42.csv", "scaler_mean.npy", "scaler_scale.npy",
                       "feat_train_frozen.csv", "feat_val_frozen.csv", "feat_test_frozen.csv", "window_manifest.csv"]
    hashes = {"preprocess_seed": PREPROCESS_SEED, "task": args.task, "train_conditions": train_conditions,
              "test_condition": test_condition, "raw_feature_file_sha256": raw_sha256,
              "n_test_windows": n_test_windows, "files": {}}
    for name in artifact_files:
        hashes["files"][name] = sha256_file(FROZEN_DIR / name)
    with open(FROZEN_DIR / "manifest_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print("=" * 100)
    print(f"FROZEN PREPROCESSING ARTIFACTS COMPLETE FOR {args.task}")
    print(f"  split_hash:      {hashes['files']['split_manifest.csv'][:16]}...")
    print(f"  feature_hash:    {hashes['files']['selected_features_seed42.json'][:16]}...")
    print(f"  gmm_hash:        {hashes['files']['gmm_seed42.pkl'][:16]}...")
    print(f"  window_hash:     {hashes['files']['window_manifest.csv'][:16]}...")
    print(f"  n_test_windows:  {n_test_windows}")
    print("=" * 100)


if __name__ == "__main__":
    main()
