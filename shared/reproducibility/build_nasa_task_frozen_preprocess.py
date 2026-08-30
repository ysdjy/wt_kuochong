# -*- coding: utf-8 -*-
r"""
Builds task-specific PREPROCESS_SEED=42 frozen preprocessing artifacts for a
NASA Milling cross-case task (N1-N4), reusing the feature-engineering /
label-construction functions of `代码/9.1nasa数据实验.py` (read-only import;
see below for why a source-text patch, not a file edit, is needed).

Case-split source of truth: the N1-N4 defined DIRECTLY by the user in this
overnight task (validated leak-free, full 1-16 coverage; see
`05_统计检验/seed_statistics/NASA_TASK_AUDIT.md`), NOT the old script's own
`FIXED_TASKS` (which uses different case assignments) and NOT any of the old
"bestcase split selection" scripts (which chose splits based on target
performance -- explicitly not reused, per the audit doc).

Why a source-text patch instead of a plain import: `9.1nasa数据实验.py`
executes `OUT_DIR.mkdir(parents=True, exist_ok=True)` at MODULE IMPORT TIME
against a hardcoded path under `C:\Users\wangting\...` (the original
author's machine), which does not exist on this machine and cannot be
created (PermissionDenied, verified). Editing the original file is
forbidden by policy, so this script reads the file's TEXT, replaces only
the two hardcoded path literals (`MAT_FILE`, `OUT_DIR`) with paths valid on
this machine, and execs that in-memory copy -- the file on disk is never
touched. All other lines (every function body: feature engineering, stage
labels, GMM, model, DC-PSR inference) are executed byte-identical to the
original.

Usage:
    python build_nasa_task_frozen_preprocess.py --task N1
    python build_nasa_task_frozen_preprocess.py --task N2
    ... N3, N4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent                    # .../shared/reproducibility
EXPAND_ROOT = THIS_DIR.parent.parent                             # .../扩充实验代码
PROJECT_ROOT = EXPAND_ROOT.parent                                 # repo root (论文/)
CODE_DIR = PROJECT_ROOT / "代码"
NASA_SCRIPT = CODE_DIR / "9.1nasa数据实验.py"
MILL_MAT = PROJECT_ROOT / "mill" / "mill.mat"

PREPROCESS_SEED = 42

# User-given N1-N4 (validated leak-free; see NASA_TASK_AUDIT.md). Authoritative
# for this round -- overrides the old script's own FIXED_TASKS.
NASA_TASKS = {
    "N1": {"train": [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], "test": [3, 12, 15, 16]},
    "N2": {"train": [1, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16], "test": [2, 8, 9, 10]},
    "N3": {"train": [1, 2, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16], "test": [3, 5, 10, 12]},
    "N4": {"train": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15], "test": [8, 12, 13, 16]},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def import_nasa_module():
    """Read-only import of 代码/9.1nasa数据实验.py with only the two
    hardcoded, off-machine path literals patched in an in-memory copy --
    the file on disk is never written to."""
    src = NASA_SCRIPT.read_text(encoding="utf-8")
    safe_out_dir = Path(tempfile.gettempdir()) / "nasa_build_frozen_side_outputs"
    old_mat_line = 'MAT_FILE = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\mill.mat")'
    old_out_line = 'OUT_DIR = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\nasa_dcpsr_results_stageaware_opt")'
    assert old_mat_line in src, "MAT_FILE line not found -- old script changed unexpectedly, aborting"
    assert old_out_line in src, "OUT_DIR line not found -- old script changed unexpectedly, aborting"
    src = src.replace(old_mat_line, f'MAT_FILE = Path(r"{MILL_MAT}")')
    src = src.replace(old_out_line, f'OUT_DIR = Path(r"{safe_out_dir}")')

    import types
    mod = types.ModuleType("nasa_frozen_build_9_1")
    mod.__file__ = str(NASA_SCRIPT)
    sys.modules["nasa_frozen_build_9_1"] = mod
    exec(compile(src, str(NASA_SCRIPT), "exec"), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(NASA_TASKS.keys()))
    args = ap.parse_args()
    train_cases = NASA_TASKS[args.task]["train"]
    test_cases = NASA_TASKS[args.task]["test"]
    if set(train_cases) & set(test_cases):
        raise ValueError(f"LEAKAGE: task {args.task} train/test case overlap")

    FROZEN_DIR = EXPAND_ROOT / "shared" / "reproducibility" / f"NASA_{args.task}_frozen_preprocess"
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"BUILDING FROZEN PREPROCESSING ARTIFACTS FOR NASA {args.task} "
          f"(train_cases={train_cases}, test_cases={test_cases}, PREPROCESS_SEED={PREPROCESS_SEED})")
    print("=" * 100)

    if not MILL_MAT.exists():
        raise FileNotFoundError(f"mill.mat missing: {MILL_MAT}")
    raw_bytes = MILL_MAT.read_bytes()
    raw_sha256 = sha256_bytes(raw_bytes)
    with open(FROZEN_DIR / "raw_mat_sha256.txt", "w", encoding="utf-8") as f:
        f.write(f"file: {MILL_MAT}\nsha256: {raw_sha256}\n")
    print(f"[1] mill.mat sha256={raw_sha256[:16]}...")

    nasa = import_nasa_module()
    nasa.SEED = PREPROCESS_SEED

    # ---- Feature extraction + case-relative stage labels (all 167 runs, all cases) ----
    raw_df = nasa.extract_signal_features()
    label_df = nasa.build_case_relative_q_and_stage_labels(raw_df)
    label_df.to_csv(FROZEN_DIR / "all_cases_labeled.csv", index=False, encoding="utf-8-sig")
    print(f"[2] extract_signal_features + build_case_relative_q_and_stage_labels: {len(label_df)} rows, "
          f"{label_df['case'].nunique()} cases")

    # ---- Task split (case-based; source-only) ----
    task_df = label_df.copy()
    train_all = task_df[task_df["case"].isin(train_cases)].copy()
    test_df = task_df[task_df["case"].isin(test_cases)].copy()
    assert set(train_all["case"].unique()) <= set(train_cases)
    assert set(test_df["case"].unique()) <= set(test_cases)
    train_df, val_df = nasa.split_train_val_by_case(train_all)
    print(f"[3] split: train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(train cases={sorted(train_df['case'].unique().tolist())}, "
          f"val cases={sorted(val_df['case'].unique().tolist())}, "
          f"test cases={sorted(test_df['case'].unique().tolist())})")

    split_rows = []
    for split_name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, r in sub.iterrows():
            split_rows.append({"case": int(r["case"]), "run": int(r["run"]), "stage_label": r["stage_label"], "split": split_name})
    split_manifest = pd.DataFrame(split_rows).sort_values(["split", "case", "run"]).reset_index(drop=True)
    split_manifest.to_csv(FROZEN_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")

    # ---- Online relative features (case-wise expanding stats; train-only fit downstream) ----
    all_feat_cols = nasa.raw_feature_columns(label_df)
    online_all = nasa.build_online_relative_features(label_df, all_feat_cols)
    # Expanding-window stats in build_online_relative_features are computed per-case
    # using only that case's own history (causal, within-case) -- splitting by
    # case+run afterward introduces no train/test leakage.
    train_on = online_all.merge(train_df[["case", "run"]], on=["case", "run"], how="inner")
    val_on = online_all.merge(val_df[["case", "run"]], on=["case", "run"], how="inner")
    test_on = online_all.merge(test_df[["case", "run"]], on=["case", "run"], how="inner")

    online_cols = nasa.online_feature_cols(online_all)
    train_on, val_on = nasa.fill_by_train_median(train_on, val_on, online_cols)
    _, test_on = nasa.fill_by_train_median(train_on, test_on, online_cols)

    selected = nasa.select_features_train_only(train_on, online_cols, nasa.N_FEATURES)
    train_on, val_on = nasa.fill_by_train_median(train_on, val_on, selected)
    _, test_on = nasa.fill_by_train_median(train_on, test_on, selected)

    with open(FROZEN_DIR / "selected_features_seed42.json", "w", encoding="utf-8") as f:
        json.dump({"preprocess_seed": PREPROCESS_SEED, "task": args.task, "train_cases": train_cases,
                    "test_cases": test_cases, "n_selected_features": len(selected),
                    "selected_features_in_order": selected}, f, ensure_ascii=False, indent=2)
    print(f"[4] selected {len(selected)} features (train-only, {args.task})")

    # ---- GMM fine-state (train-only) ----
    gmm, raw_to_order = nasa.fit_gmm_fine_states(train_on)
    train_on = nasa.assign_fine_states(train_on, gmm, raw_to_order)
    val_on = nasa.assign_fine_states(val_on, gmm, raw_to_order)
    test_on = nasa.assign_fine_states(test_on, gmm, raw_to_order)
    with open(FROZEN_DIR / "gmm_seed42.pkl", "wb") as f:
        pickle.dump({"gmm": gmm, "raw_to_order": raw_to_order}, f)
    print(f"[5] GMM fit on {len(train_on)} train rows")

    # ---- StandardScaler (train-only) ----
    scaler = nasa.StandardScaler().fit(train_on[selected].values)
    np.save(FROZEN_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(FROZEN_DIR / "scaler_scale.npy", scaler.scale_)
    for d in [train_on, val_on, test_on]:
        d[selected] = np.nan_to_num(scaler.transform(d[selected].values), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[6] StandardScaler fit on train[selected] shape={train_on[selected].shape}")

    train_on.to_csv(FROZEN_DIR / "feat_train_frozen.csv", index=False, encoding="utf-8-sig")
    val_on.to_csv(FROZEN_DIR / "feat_val_frozen.csv", index=False, encoding="utf-8-sig")
    test_on.to_csv(FROZEN_DIR / "feat_test_frozen.csv", index=False, encoding="utf-8-sig")

    # ---- Windows (L=6, NASA's own pre-existing window length; case 6 has only 1 run < L and is skipped) ----
    L = nasa.L_DEFAULT
    window_rows = []
    for split_name, df_sub in [("train", train_on), ("val", val_on), ("test", test_on)]:
        pack = nasa.build_sliding_windows(df_sub, selected, L)
        for _, r in pack["meta"].iterrows():
            window_rows.append({"case": int(r["case"]), "run_end": int(r["run"]), "stage": r["true_stage"],
                                 "q_target": float(r["q_true"]), "split": split_name})
    window_manifest = pd.DataFrame(window_rows).sort_values(["split", "case", "run_end"]).reset_index(drop=True)
    window_manifest.to_csv(FROZEN_DIR / "window_manifest.csv", index=False, encoding="utf-8-sig")
    n_test_windows = int((window_manifest["split"] == "test").sum())
    print(f"[7] window_manifest.csv rows={len(window_manifest)} (L={L})")
    print(f"    {args.task} test universe: {n_test_windows} windows "
          f"(test cases with windows: {sorted(window_manifest[window_manifest.split=='test']['case'].unique().tolist())})")

    # ---- Leakage re-verification ----
    leak_report = {
        "task": args.task, "train_cases": train_cases, "test_cases": test_cases,
        "test_case_in_train_split": bool(set(test_cases) & set(split_manifest[split_manifest.split == "train"]["case"])),
        "test_case_in_val_split": bool(set(test_cases) & set(split_manifest[split_manifest.split == "val"]["case"])),
        "feature_selection_fit_on": "train only (select_features_train_only(train_on, ...))",
        "gmm_fit_on": "train only (fit_gmm_fine_states(train_on))",
        "scaler_fit_on": "train only (StandardScaler().fit(train_on[selected]))",
        "n_test_windows": n_test_windows,
    }
    leak_report["leakage_detected"] = bool(leak_report["test_case_in_train_split"] or leak_report["test_case_in_val_split"])
    with open(FROZEN_DIR / "leakage_check.json", "w", encoding="utf-8") as f:
        json.dump(leak_report, f, indent=2)
    if leak_report["leakage_detected"]:
        raise RuntimeError(f"PROTOCOL_FAILED: test cases leaked into train/val split for {args.task}")
    print(f"[8] leakage_check.json: leakage_detected={leak_report['leakage_detected']}")

    if n_test_windows == 0:
        raise RuntimeError(f"PROTOCOL_FAILED: {args.task} produced ZERO test windows (all test cases shorter than L={L}?)")

    # ---- manifest_hashes.json ----
    artifact_files = ["raw_mat_sha256.txt", "all_cases_labeled.csv", "split_manifest.csv",
                       "selected_features_seed42.json", "gmm_seed42.pkl", "scaler_mean.npy", "scaler_scale.npy",
                       "feat_train_frozen.csv", "feat_val_frozen.csv", "feat_test_frozen.csv", "window_manifest.csv"]
    hashes = {"preprocess_seed": PREPROCESS_SEED, "task": args.task, "train_cases": train_cases,
              "test_cases": test_cases, "raw_mat_sha256": raw_sha256, "n_test_windows": n_test_windows,
              "window_length_L": L, "files": {}}
    for name in artifact_files:
        hashes["files"][name] = sha256_file(FROZEN_DIR / name)
    with open(FROZEN_DIR / "manifest_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print("=" * 100)
    print(f"FROZEN PREPROCESSING ARTIFACTS COMPLETE FOR NASA {args.task}")
    print(f"  n_test_windows: {n_test_windows}")
    print("=" * 100)


if __name__ == "__main__":
    main()
