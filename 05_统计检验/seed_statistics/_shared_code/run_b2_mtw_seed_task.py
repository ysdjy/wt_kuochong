# -*- coding: utf-8 -*-
r"""
B2 (Relative-stage TCN-GRU, stage-only; dcpsr package's own id "B10" --
DEEP_SINGLE = {"B7":"mlp","B8":"tcn","B9":"gru","B10":"tcngru"}), MTW-CM
D1-M/D2-M/D3-M, strict TRAIN_SEED isolation.

Sibling to run_mtw_seed_task.py (B9+B3) and run_b2_seed_task.py (B2,
PHM2010) -- same frozen-preprocessing source and output convention. Reuses
`experiments_mendeley/code/dcpsr`'s own native single-task baseline path
(`ExperimentRunner._train_stage_only(packs, input_dim, kind="tcngru", seed,
device)` + `_predict_stage_only(net, pack, device, mid="B10")`, called via a
throwaway ExperimentRunner instance exactly as run_mtw_seed_task.py already
does for the multitask backbone) instead of building anything fresh --
`StageOnlyNet(kind="tcngru")` and this training path already exist in
dcpsr/runner.py specifically for this baseline. No probability
post-processing (B9-only step); q-MAE/q-RMSE/q-R2/Spearman are legitimately
NaN (this model has no continuous degradation-index head, confirmed via
`evaluate(p, "b10", with_q=False)` -- the same call dcpsr's own runner uses
for this baseline).

Usage:
    python run_b2_mtw_seed_task.py --task D1-M --train_seed 0 \
        --results_root ../B2_MTW_D1M_seed_landscape/results \
        --backbone_root ../B2_MTW_D1M_seed_landscape/backbone_checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

SEED_STATS_DIR = Path(__file__).resolve().parent.parent
EXPAND_ROOT = SEED_STATS_DIR.parents[1]
PROJECT_ROOT = EXPAND_ROOT.parent
# Self-contained: vendored copy inside this repo (git-clone portable), NOT the
# outside parent project -- see _shared_code/vendored_legacy/README.md.
DCPSR_CODE_DIR = SEED_STATS_DIR / "_shared_code" / "vendored_legacy"

STAGE_ORDER = ["E", "M", "L"]
DCPSR_STAGE_NAMES = ["early", "middle", "late"]  # dcpsr.config.STAGE_NAMES -- column-name source, not our own convention


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir_for(task: str) -> Path:
    return EXPAND_ROOT / "shared" / "reproducibility" / f"MTW_{task}_frozen_preprocess"


def assert_frozen_artifacts_unchanged(frozen_dir: Path):
    with open(frozen_dir / "manifest_hashes.json", "r", encoding="utf-8") as f:
        recorded = json.load(f)
    mismatches = []
    for name, expected in recorded["files"].items():
        actual = sha256_file(frozen_dir / name)
        if actual != expected:
            mismatches.append((name, expected, actual))
    if mismatches:
        lines = [f"  {n}: expected {e[:16]}... got {a[:16]}..." for n, e, a in mismatches]
        raise RuntimeError("FROZEN PREPROCESSING ARTIFACTS CHANGED -- protocol violated:\n" + "\n".join(lines))
    return recorded


def set_train_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"[set_train_seed] WARNING: use_deterministic_algorithms unavailable: {e}")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path):
    sys.path.insert(0, str(DCPSR_CODE_DIR))
    from dcpsr import config as C
    from dcpsr.model import build_windows
    from dcpsr.metrics import evaluate
    from dcpsr.runner import ExperimentRunner

    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")

    out_dir = results_root / f"seed{train_seed}"
    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if (prior.get("split_hash") == hashes["files"]["split_info.json"]
                    and prior.get("feature_hash") == hashes["files"]["selected_features_seed42.json"]
                    and prior.get("gmm_hash") == hashes["files"]["gmm_seed42.pkl"]
                    and prior.get("train_seed") == train_seed and prior.get("task") == task):
                print(f"[resume] task={task} seed={train_seed} already DONE with matching hashes -- skipping")
                return "skipped"
        except Exception:
            pass

    with open(frozen_dir / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        selected = json.load(f)["selected_features_in_order"]
    feat_train = pd.read_csv(frozen_dir / "feat_train_frozen.csv")
    feat_val = pd.read_csv(frozen_dir / "feat_val_frozen.csv")
    feat_test = pd.read_csv(frozen_dir / "feat_test_frozen.csv")
    L = C.ARCH["L"]

    set_train_seed(train_seed)
    Xtr, ystr, yftr, yqtr, sitr, oktr, meta_tr = build_windows(feat_train, selected, L, "train")
    Xva, ysva, yfva, yqva, siva, okva, meta_va = build_windows(feat_val, selected, L, "val")
    Xte, yste, yfte, yqte, site, okte, meta_te = build_windows(feat_test, selected, L, "test")
    packs = {
        "train": dict(X=Xtr, ys=ystr, yf=yftr, yq=yqtr, seq_index=sitr, order_key=oktr, meta=meta_tr),
        "val": dict(X=Xva, ys=ysva, yf=yfva, yq=yqva, seq_index=siva, order_key=okva, meta=meta_va),
        "test": dict(X=Xte, ys=yste, yf=yfte, yq=yqte, seq_index=site, order_key=okte, meta=meta_te),
    }
    y_true = meta_te["stage_true_id"].values.astype(int)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = ExperimentRunner(labelled=None, online=None, candidate_features=None,
                               out_root=results_root, dataset_name="MTW_CM")

    t0 = time.time()
    net, best_val_loss = runner._train_stage_only(packs, len(selected), "tcngru", train_seed, device)
    t1 = time.time()
    backbone_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = backbone_root / f"b2_backbone_{task}_seed{train_seed}.pth"
    torch.save(net.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    p = ExperimentRunner._predict_stage_only(net, packs["test"], device, "B10")
    m = evaluate(p, "b10", with_q=False)

    y_pred = p["stage_pred_b10"].values.astype(int)
    prob = p[[f"b10_prob_{s}" for s in DCPSR_STAGE_NAMES]].values.astype(float)

    seed_summary = {
        "Method": "B2_TCN_GRU", "Dataset": "MTW_CM", "Task": task, "Seed": train_seed,
        "Acc": m["Acc"], "Macro-F1": m["Macro_F1"], "E-F1": m["E_F1"], "M-F1": m["M_F1"], "L-F1": m["L_F1"],
        "M-Precision": m["M_Pre"], "M-Recall": m["M_Rec"], "M_to_E": m["M_to_E"], "M_to_L": m["M_to_L"],
        "Rev": m["Rev"], "Jump": m["Jump"], "Smooth": m["Smooth"],
        "q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan, "Spearman": np.nan,
        "best_val_loss": float(best_val_loss), "n_test": len(p), "training_seconds": t1 - t0,
    }

    pred_df = p[["sequence_id", "domain_id", "order_key", "stage_true", "q_true"]].copy()
    pred_df["q_hat"] = np.nan  # non-multitask model has no continuous degradation-index head
    pred_df["pred"] = [STAGE_ORDER[i] for i in y_pred]
    for eng, short in zip(DCPSR_STAGE_NAMES, STAGE_ORDER):
        pred_df[f"prob_{short}"] = p[f"b10_prob_{eng}"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    config_hash = hashlib.sha256(json.dumps({
        "task": task, "arch": C.ARCH, "epochs": C.EPOCHS, "patience": C.PATIENCE, "kind": "tcngru",
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = {
        "method": "B2_TCN_GRU", "dataset": "MTW_CM", "task": task,
        "preprocess_seed": hashes["preprocess_seed"], "train_seed": train_seed,
        "architecture": C.ARCH, "kind": "tcngru",
    }
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    meta_json = {
        "method": "B2_TCN_GRU", "dataset": "MTW_CM", "task": task,
        "preprocess_seed": hashes["preprocess_seed"], "train_seed": train_seed,
        "train_sequences": hashes.get("train_sequences"), "validation_sequences": hashes.get("validation_sequences"),
        "test_sequences": hashes.get("test_sequences"), "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_info.json"], "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "best_val_loss": float(best_val_loss), "n_test": len(p),
        "backbone_checkpoint_hash": backbone_checkpoint_hash, "backbone_checkpoint_path": str(ckpt_path),
        "config_hash": config_hash,
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"note": "_train_stage_only() returns best val loss, not a per-epoch curve", "best_val_loss": float(best_val_loss), "training_seconds": t1 - t0}]).to_csv(
        out_dir / "training_log.csv", index=False, encoding="utf-8-sig"
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, default=str)
    (out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
    print(f"[save_run] task={task} seed={train_seed} Acc={seed_summary['Acc']:.4f} "
          f"MacroF1={seed_summary['Macro-F1']:.4f} MF1={seed_summary['M-F1']:.4f} "
          f"MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} n_test={len(p)}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["D1-M", "D2-M", "D3-M"])
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--backbone_root", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.task, args.train_seed, args.results_root, args.backbone_root)
        print(f"[protocol] DONE({status}): task={args.task} train_seed={args.train_seed}")
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "error.log", "w", encoding="utf-8") as f:
            import traceback
            f.write(f"task={args.task} seed={args.train_seed}\n")
            traceback.print_exc(file=f)
        (out_dir / "FAILED.flag").write_text(f"failed: {e}\n", encoding="utf-8")
        print(f"[protocol] FAILED: task={args.task} train_seed={args.train_seed}: {e}")
        raise


if __name__ == "__main__":
    main()
