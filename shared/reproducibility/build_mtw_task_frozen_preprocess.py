# -*- coding: utf-8 -*-
r"""
Builds task-specific PREPROCESS_SEED=42 frozen preprocessing artifacts for a
MTW-CM (Mendeley multi-machine tool wear) task (D1-M/D2-M/D3-M), reusing the
already-validated `experiments_mendeley/code/dcpsr` package (read-only
import, never modified) for: dataset loading (cached feature CSV),
condition-relative stage labels, causal online features, source-only train/
val/test split (`splits.make_splits`), train-only feature selection
(`selection.select_features`), StandardScaler, and the 5-state GMM
(`stages.FineStateModel`) -- exactly mirroring the PHM2010/NASA frozen-
preprocessing scripts, so only TRAIN_SEED varies downstream (0..100),
never the split/features/scaler/GMM.

D1-M/D2-M/D3-M come from `dcpsr.datasets.mendeley.MendeleyMachineToolWear
.task_definitions()` (machine-hold-out mapping hardcoded in that adapter:
D1-M holds out M3, D2-M holds out M2, D3-M holds out M1) -- confirmed
identical to `experiments_mendeley/02_protocols/task_definitions.json`
(the already-frozen protocol doc referenced in MTW_TASK_AUDIT.md). Not
redefined here.

Usage:
    python build_mtw_task_frozen_preprocess.py --task D1-M
    python build_mtw_task_frozen_preprocess.py --task D2-M
    python build_mtw_task_frozen_preprocess.py --task D3-M
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent                    # .../shared/reproducibility
EXPAND_ROOT = THIS_DIR.parent.parent                             # .../扩充实验代码
PROJECT_ROOT = EXPAND_ROOT.parent                                 # repo root (论文/)
DCPSR_CODE_DIR = PROJECT_ROOT / "experiments_mendeley" / "code"
MTW_RAW_DIR = PROJECT_ROOT / "Multivariate time series data of milling processes with varying tool wear and machine tools"
FEATURES_DIR = EXPAND_ROOT / "05_统计检验" / "seed_statistics" / "_mtw_run_root" / "01_features"

PREPROCESS_SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["D1-M", "D2-M", "D3-M"])
    args = ap.parse_args()
    task_name = args.task

    sys.path.insert(0, str(DCPSR_CODE_DIR))
    from dcpsr.datasets.mendeley import MendeleyMachineToolWear
    from dcpsr.stages import build_stage_labels, FineStateModel
    from dcpsr.online_features import build_online_features, online_feature_columns
    from dcpsr.splits import make_splits
    from dcpsr.selection import select_features
    from sklearn.preprocessing import StandardScaler

    FROZEN_DIR = EXPAND_ROOT / "shared" / "reproducibility" / f"MTW_{task_name}_frozen_preprocess"
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"BUILDING FROZEN PREPROCESSING ARTIFACTS FOR MTW-CM {task_name} (PREPROCESS_SEED={PREPROCESS_SEED})")
    print("=" * 100)

    ad = MendeleyMachineToolWear(MTW_RAW_DIR, FEATURES_DIR, channel_set="primary")
    table = ad.load_run_level_table()
    table_bytes = (FEATURES_DIR / "run_level_features_primary.csv").read_bytes()
    features_sha256 = sha256_bytes(table_bytes)
    print(f"[1] loaded cached run-level table: {len(table)} rows, sha256={features_sha256[:16]}...")

    labelled, stage_summary = build_stage_labels(table)
    stage_summary.to_csv(FROZEN_DIR / "stage_summary.csv", index=False, encoding="utf-8-sig")
    raw_cols = ad.feature_columns(table)
    online = build_online_features(labelled, raw_cols)
    candidates = online_feature_columns(online)
    print(f"[2] build_stage_labels + build_online_features: {len(raw_cols)} raw cols -> {len(candidates)} candidate online features")

    task = next(t for t in ad.task_definitions() if t.name == task_name)
    task.validate()
    print(f"[3] task={task.name} train_domains={task.train_domains} test_domains={task.test_domains} "
          f"train_sequences={task.train_sequences} test_sequences={task.test_sequences}")

    tr_lab, va_lab, te_lab, split_info = make_splits(labelled, task, seed=PREPROCESS_SEED)
    print(f"[4] make_splits(seed={PREPROCESS_SEED}): {split_info}")
    if set(task.test_sequences) & set(split_info["train_sequences"]) or set(task.test_sequences) & set(split_info["validation_sequences"]):
        raise RuntimeError(f"PROTOCOL_FAILED: {task_name} test sequences leaked into train/val")

    # `online` already carries every meta/derived label column (see
    # online_features.py's own `keep` list) alongside the __rel/__slope/__rank
    # candidate features, so a single merge on the row key is sufficient.
    key = ["sequence_id", "order_key"]
    tr = tr_lab[key].merge(online, on=key, how="left")
    va = va_lab[key].merge(online, on=key, how="left")
    te = te_lab[key].merge(online, on=key, how="left")

    med = tr[candidates].median()
    for f in (tr, va, te):
        f[candidates] = f[candidates].fillna(med).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    selected, sel_table = select_features(tr, candidates, seed=PREPROCESS_SEED)
    sel_table.to_csv(FROZEN_DIR / "selected_features_scored_seed42.csv", index=False, encoding="utf-8-sig")
    with open(FROZEN_DIR / "selected_features_seed42.json", "w", encoding="utf-8") as f:
        json.dump({"preprocess_seed": PREPROCESS_SEED, "task": task_name,
                    "train_sequences": split_info["train_sequences"],
                    "validation_sequences": split_info["validation_sequences"],
                    "test_sequences": split_info["test_sequences"],
                    "n_selected_features": len(selected), "selected_features_in_order": selected}, f, ensure_ascii=False, indent=2)
    print(f"[5] select_features(seed={PREPROCESS_SEED}): {len(selected)} features selected (train-only)")

    scaler = StandardScaler().fit(tr[selected].values)
    np.save(FROZEN_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(FROZEN_DIR / "scaler_scale.npy", scaler.scale_)
    for f in (tr, va, te):
        f[selected] = np.nan_to_num(scaler.transform(f[selected].values), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[6] StandardScaler fit on train[selected] shape={tr[selected].shape}")

    fsm = FineStateModel(seed=PREPROCESS_SEED).fit(tr)
    tr, va, te = fsm.assign(tr), fsm.assign(va), fsm.assign(te)
    fsm.mapping_table.to_csv(FROZEN_DIR / "gmm_fine_state_mapping.csv", index=False, encoding="utf-8-sig")
    with open(FROZEN_DIR / "gmm_seed42.pkl", "wb") as f:
        pickle.dump({"gmm": fsm.gmm, "order": fsm.order}, f)
    print(f"[7] FineStateModel fit on {len(tr)} train rows")

    tr.to_csv(FROZEN_DIR / "feat_train_frozen.csv", index=False, encoding="utf-8-sig")
    va.to_csv(FROZEN_DIR / "feat_val_frozen.csv", index=False, encoding="utf-8-sig")
    te.to_csv(FROZEN_DIR / "feat_test_frozen.csv", index=False, encoding="utf-8-sig")

    # window universe check (verified from actual data, not hardcoded), L from dcpsr.config
    from dcpsr import config as C
    from dcpsr.model import build_windows
    L = C.ARCH["L"]
    _, _, _, _, _, _, meta_test = build_windows(te, selected, L, "test")
    n_test_windows = len(meta_test)
    print(f"[8] build_windows(L={L}) test universe: {n_test_windows} windows "
          f"(test sequences: {sorted(meta_test['sequence_id'].unique().tolist())})")
    if n_test_windows == 0:
        raise RuntimeError(f"PROTOCOL_FAILED: {task_name} produced ZERO test windows")

    with open(FROZEN_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, default=str)

    artifact_files = ["stage_summary.csv", "selected_features_seed42.json", "selected_features_scored_seed42.csv",
                       "gmm_seed42.pkl", "gmm_fine_state_mapping.csv", "scaler_mean.npy", "scaler_scale.npy",
                       "feat_train_frozen.csv", "feat_val_frozen.csv", "feat_test_frozen.csv", "split_info.json"]
    hashes = {"preprocess_seed": PREPROCESS_SEED, "task": task_name,
              "train_sequences": split_info["train_sequences"], "validation_sequences": split_info["validation_sequences"],
              "test_sequences": split_info["test_sequences"], "features_table_sha256": features_sha256,
              "window_length_L": L, "n_test_windows": n_test_windows, "files": {}}
    for name in artifact_files:
        hashes["files"][name] = sha256_file(FROZEN_DIR / name)
    with open(FROZEN_DIR / "manifest_hashes.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    print("=" * 100)
    print(f"FROZEN PREPROCESSING ARTIFACTS COMPLETE FOR MTW-CM {task_name}")
    print(f"  n_test_windows: {n_test_windows}")
    print("=" * 100)


if __name__ == "__main__":
    main()
