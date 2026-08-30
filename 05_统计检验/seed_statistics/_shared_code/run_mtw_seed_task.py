# -*- coding: utf-8 -*-
r"""
B9 (DC-PHSR, legacy id DC-PSR/B12=A6/final) + paired B3 (Multi-task TCN-GRU,
legacy id B11=raw) for MTW-CM (D1-M/D2-M/D3-M), strict TRAIN_SEED isolation.

Reuses `experiments_mendeley/code/dcpsr` (read-only import, never modified):
model.build_windows/TCNGRUMultiTask/class_weights/make_loader, inference.
apply_inference (called with the FIXED C.FUSION_DEFAULT -- never the grid
search in ExperimentRunner.run_unit's default path, so params are never
tuned per seed or per task), metrics.evaluate. Training loop reuses
ExperimentRunner._train_multitask/_predict_multitask/_epoch directly (same
already-validated code as the original pipeline) via a throwaway instance
(those methods don't touch self.labelled/online/candidates).

B3 = evaluate(inf_test, "raw")   (== dcpsr's B11, the raw pre-fusion backbone output)
B9 = evaluate(inf_test, "final") (== dcpsr's B12/A6, the full DC-PSR fusion)
Both come from the SAME trained backbone in this same process -- B3 is never
trained separately.

Usage:
    python run_mtw_seed_task.py --task D1-M --train_seed 0 \
        --results_root ../B9_MTW_D1M_seed_landscape/results \
        --backbone_root ../B9_MTW_D1M_seed_landscape/backbone_checkpoints \
        --b3_results_root ../B3_MTW_D1M_seed_landscape/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED_STATS_DIR = Path(__file__).resolve().parent.parent
EXPAND_ROOT = SEED_STATS_DIR.parents[1]
PROJECT_ROOT = EXPAND_ROOT.parent
DCPSR_CODE_DIR = PROJECT_ROOT / "experiments_mendeley" / "code"


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


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path, b3_results_root: Path):
    sys.path.insert(0, str(DCPSR_CODE_DIR))
    from dcpsr import config as C
    from dcpsr.model import build_windows
    from dcpsr.inference import apply_inference
    from dcpsr.metrics import evaluate
    from dcpsr.runner import ExperimentRunner

    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")

    out_dir = results_root / f"seed{train_seed}"
    b3_out_dir = b3_results_root / f"seed{train_seed}"
    if (out_dir / "DONE.flag").exists() and (b3_out_dir / "DONE.flag").exists():
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
                               out_root=results_root, dataset_name="MTW_CM")  # throwaway: only its
    # _train_multitask/_predict_multitask/_epoch static/instance methods are used below (no self.* access)

    t0 = time.time()
    model, hist, best_epoch = runner._train_multitask(packs, len(selected), train_seed, device)
    t1 = time.time()
    backbone_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = backbone_root / f"mtw_backbone_{task}_seed{train_seed}.pth"
    torch.save(model.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    pred_test = runner._predict_multitask(model, packs["test"], device)
    # FIXED fusion params -- never searched/tuned per seed or per task (dataset-agnostic
    # default identical to PHM2010's B12_PARAMS / NASA's PROB_PARAMS).
    fusion_params = dict(C.FUSION_DEFAULT)
    inf_test = apply_inference(pred_test, fusion_params)

    m9 = evaluate(inf_test, "final")   # B12/DC-PSR/DC-PHSR = B9
    m3 = evaluate(inf_test, "raw")     # B11/Multi-task TCN-GRU = B3

    def to_row(m, method_full):
        return {
            "Method": method_full, "Dataset": "MTW_CM", "Task": task, "Seed": train_seed,
            "Acc": m["Acc"], "Macro-F1": m["Macro_F1"], "E-F1": m["E_F1"], "M-F1": m["M_F1"], "L-F1": m["L_F1"],
            "M-Precision": m["M_Pre"], "M-Recall": m["M_Rec"], "M_to_E": m["M_to_E"], "M_to_L": m["M_to_L"],
            "Rev": m["Rev"], "Jump": m["Jump"], "Smooth": m["Smooth"],
            "q-MAE": m.get("q_MAE", np.nan), "q-RMSE": m.get("q_RMSE", np.nan), "q-R2": m.get("q_R2", np.nan),
            "Spearman": m.get("Spearman", np.nan),
            "best_epoch": int(best_epoch), "n_test": len(inf_test), "training_seconds": t1 - t0,
        }

    row9 = to_row(m9, "B9_DC_PHSR")
    row3 = to_row(m3, "B3_Multitask_TCN_GRU")

    from sklearn.metrics import confusion_matrix
    y_pred9 = inf_test["stage_pred_final"].values.astype(int)
    y_pred3 = inf_test["stage_pred_raw"].values.astype(int)
    stage_order = ["E", "M", "L"]
    cm9 = confusion_matrix(y_true, y_pred9, labels=[0, 1, 2])
    cm3 = confusion_matrix(y_true, y_pred3, labels=[0, 1, 2])
    cm9_df = pd.DataFrame(cm9, index=[f"true_{s}" for s in stage_order], columns=[f"pred_{s}" for s in stage_order])
    cm3_df = pd.DataFrame(cm3, index=[f"true_{s}" for s in stage_order], columns=[f"pred_{s}" for s in stage_order])

    pred_df9 = meta_te[["sequence_id", "domain_id", "order_key", "stage_true", "q_true"]].copy()
    pred_df9["pred"] = inf_test["stage_pred_final_name"].values
    pred_df9["q_hat"] = inf_test["q_hat"].values
    for i, s in enumerate(["E", "M", "L"]):
        pred_df9[f"prob_{s}"] = inf_test[[f"final_prob_early", f"final_prob_middle", f"final_prob_late"][i]].values

    pred_df3 = meta_te[["sequence_id", "domain_id", "order_key", "stage_true", "q_true"]].copy()
    pred_df3["pred"] = inf_test["stage_pred_raw_name"].values
    pred_df3["q_hat"] = inf_test["q_hat"].values
    for i, s in enumerate(["E", "M", "L"]):
        pred_df3[f"prob_{s}"] = inf_test[[f"raw_prob_early", f"raw_prob_middle", f"raw_prob_late"][i]].values

    config_hash = hashlib.sha256(json.dumps({
        "task": task, "arch": C.ARCH, "epochs": C.EPOCHS, "patience": C.PATIENCE,
        "fusion_params": fusion_params,
    }, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    pairing = {
        "backbone_checkpoint_hash": backbone_checkpoint_hash, "backbone_checkpoint_path": str(ckpt_path),
        "config_hash": config_hash,
        "paired_method": "B9_DC_PHSR<->B3_Multitask_TCN_GRU (same backbone, same seed, no separate B3 training)",
    }
    common_meta = {
        "dataset": "MTW_CM", "task": task, "preprocess_seed": hashes["preprocess_seed"], "train_seed": train_seed,
        "train_sequences": hashes["train_sequences"], "validation_sequences": hashes["validation_sequences"],
        "test_sequences": hashes["test_sequences"], "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_info.json"], "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "fusion_params": fusion_params, "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "best_epoch": int(best_epoch), "n_test": len(inf_test), **pairing,
    }

    for out_d, row, pred_df, cm_df, method in [
        (out_dir, row9, pred_df9, cm9_df, "B9_DC_PHSR"),
        (b3_out_dir, row3, pred_df3, cm3_df, "B3_Multitask_TCN_GRU"),
    ]:
        out_d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out_d / "metrics.csv", index=False, encoding="utf-8-sig")
        json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
        with open(out_d / "metrics.json", "w", encoding="utf-8") as f:
            json.dump([json_row], f, indent=2, default=str)
        pred_df.to_csv(out_d / "predictions.csv", index=False, encoding="utf-8-sig")
        cm_df.to_csv(out_d / "confusion_matrix.csv", encoding="utf-8-sig")
        meta_json = {"method": method, **common_meta}
        with open(out_d / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2, default=str)
        if method == "B9_DC_PHSR":
            hist.to_csv(out_d / "training_log.csv", index=False, encoding="utf-8-sig")
            hist.to_csv(out_d / "validation_history.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame([{"note": "shared training run -- see paired B9 dir's training_log.csv"}]).to_csv(
                out_d / "training_log.csv", index=False, encoding="utf-8-sig")
        (out_d / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")

    print(f"[save_run] task={task} seed={train_seed} B9 Acc={row9['Acc']:.4f} MacroF1={row9['Macro-F1']:.4f} "
          f"MF1={row9['M-F1']:.4f} MRec={row9['M-Recall']:.4f} Smooth={row9['Smooth']:.4f} "
          f"| B3 Acc={row3['Acc']:.4f} MacroF1={row3['Macro-F1']:.4f} n_test={len(inf_test)}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["D1-M", "D2-M", "D3-M"])
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--backbone_root", required=True, type=Path)
    ap.add_argument("--b3_results_root", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.task, args.train_seed, args.results_root, args.backbone_root, args.b3_results_root)
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
