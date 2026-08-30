# -*- coding: utf-8 -*-
r"""
Backfill paired B3 for already-completed B9_NASA_{task}_seed_landscape seeds
that predate the B3-pairing patch, WITHOUT retraining -- loads the
already-saved backbone checkpoint and does a forward-pass-only
reconstruction (same pattern as backfill_b3_phm_task.py).

Usage:
    python backfill_b3_nasa_task.py --task N1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

warnings.filterwarnings("ignore")

SEED_STATS_DIR = Path(__file__).resolve().parent.parent
EXPAND_ROOT = SEED_STATS_DIR.parents[1]
PROJECT_ROOT = EXPAND_ROOT.parent
CODE_DIR = PROJECT_ROOT / "代码"
NASA_SCRIPT = CODE_DIR / "9.1nasa数据实验.py"
MILL_MAT = PROJECT_ROOT / "mill" / "mill.mat"
COMMON_PIPELINE_PATH = PROJECT_ROOT / "final_statistical_evidence" / "scripts" / "methods" / "common_pipeline.py"
STAGE_ORDER = ["E", "M", "L"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_nasa_module():
    src = NASA_SCRIPT.read_text(encoding="utf-8")
    safe_out_dir = Path(tempfile.gettempdir()) / "nasa_b3_backfill_side_outputs"
    old_mat_line = 'MAT_FILE = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\mill.mat")'
    old_out_line = 'OUT_DIR = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\nasa_dcpsr_results_stageaware_opt")'
    src = src.replace(old_mat_line, f'MAT_FILE = Path(r"{MILL_MAT}")')
    src = src.replace(old_out_line, f'OUT_DIR = Path(r"{safe_out_dir}")')
    mod = types.ModuleType("nasa_b3_backfill_9_1")
    mod.__file__ = str(NASA_SCRIPT)
    sys.modules["nasa_b3_backfill_9_1"] = mod
    exec(compile(src, str(NASA_SCRIPT), "exec"), mod.__dict__)
    return mod


def import_common_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location("common_pipeline_nasa_b3_backfill", COMMON_PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["N1", "N2", "N3", "N4"])
    args = ap.parse_args()
    task = args.task

    B9_DIR = SEED_STATS_DIR / f"B9_NASA_{task}_seed_landscape"
    B3_DIR = SEED_STATS_DIR / f"B3_NASA_{task}_seed_landscape"
    FROZEN_DIR = EXPAND_ROOT / "shared" / "reproducibility" / f"NASA_{task}_frozen_preprocess"

    nasa = import_nasa_module()
    cp = import_common_pipeline()
    with open(FROZEN_DIR / "manifest_hashes.json", "r", encoding="utf-8") as f:
        hashes = json.load(f)
    L = hashes["window_length_L"]
    with open(FROZEN_DIR / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        selected = json.load(f)["selected_features_in_order"]
    feat_test = pd.read_csv(FROZEN_DIR / "feat_test_frozen.csv")
    te_pack = nasa.build_sliding_windows(feat_test, selected, L)
    y_true = te_pack["meta"]["stage_id"].values.astype(int) if "stage_id" in te_pack["meta"].columns else None

    seed_dirs = sorted((B9_DIR / "results").glob("seed*"), key=lambda p: int(p.name.replace("seed", "")))
    n_backfilled = n_skipped = n_missing = 0
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))
        if not (sd / "DONE.flag").exists():
            continue
        b3_out = B3_DIR / "results" / f"seed{seed}"
        if (b3_out / "DONE.flag").exists():
            n_skipped += 1
            continue
        ckpt_path = B9_DIR / "backbone_checkpoints" / f"nasa_backbone_{task}_seed{seed}.pth"
        if not ckpt_path.exists():
            print(f"[MISSING] seed{seed}: no checkpoint at {ckpt_path}")
            n_missing += 1
            continue

        with open(sd / "run_meta.json", "r", encoding="utf-8") as f:
            b9_meta = json.load(f)

        import torch
        model = nasa.TCNGRUMultiTaskModel(len(selected))
        state = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state)
        model = model.to(nasa.DEVICE)

        pred_raw = nasa.predict_model(model, te_pack, multitask=True)
        pred = nasa.apply_dcpsr_inference(pred_raw, nasa.PROB_PARAMS)

        y_true_local = pred["stage_id"].values.astype(int)
        prob_b3 = pred[["p_raw_E", "p_raw_M", "p_raw_L"]].values.astype(float)
        y_pred_b3 = prob_b3.argmax(axis=1)
        m3 = cp.full_metrics(y_true_local, y_pred_b3, prob_b3)

        q_true = pred["q_true"].values.astype(float)
        q_hat = pred["q_hat"].values.astype(float)
        q_mae = mean_absolute_error(q_true, q_hat)
        q_rmse = float(np.sqrt(mean_squared_error(q_true, q_hat)))
        q_r2 = r2_score(q_true, q_hat) if len(set(q_true.round(6))) > 1 else np.nan
        spearman = np.nan
        if spearmanr is not None:
            try:
                spearman = float(spearmanr(q_true, q_hat).correlation)
            except Exception:
                spearman = np.nan

        b3_summary = {
            "Method": "B3_Multitask_TCN_GRU", "Dataset": "NASA_Milling", "Task": task, "Seed": seed,
            "Acc": m3["Acc"], "Macro-F1": m3["MacroF1"], "E-F1": m3["E_F1"], "M-F1": m3["M_F1"], "L-F1": m3["L_F1"],
            "M-Precision": m3["M_Precision"], "M-Recall": m3["M_Recall"], "M_to_E": m3["M_to_E"], "M_to_L": m3["M_to_L"],
            "Rev": m3["Rev"], "Jump": m3["Jump"], "Smooth": m3["Smooth"],
            "q-MAE": q_mae, "q-RMSE": q_rmse, "q-R2": q_r2, "Spearman": spearman,
            "best_epoch": b9_meta.get("best_epoch"), "backfilled": True,
        }
        pred_df_b3 = pred[["case", "run", "stage_id", "q_true", "q_hat"]].copy()
        pred_df_b3["pred"] = [STAGE_ORDER[i] for i in y_pred_b3]
        pred_df_b3["prob_E"], pred_df_b3["prob_M"], pred_df_b3["prob_L"] = prob_b3[:, 0], prob_b3[:, 1], prob_b3[:, 2]

        cm3 = confusion_matrix(y_true_local, y_pred_b3, labels=[0, 1, 2])
        cm3_df = pd.DataFrame(cm3, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

        b3_out.mkdir(parents=True, exist_ok=True)
        b3_meta_json = {
            "method": "B3_Multitask_TCN_GRU", "dataset": "NASA_Milling", "task": task, "train_seed": seed,
            "preprocess_seed": 42, "feature_hash": b9_meta.get("feature_hash"),
            "split_hash": b9_meta.get("split_hash"), "gmm_hash": b9_meta.get("gmm_hash"),
            "window_hash": b9_meta.get("window_hash"),
            "backbone_checkpoint_hash": sha256_file(ckpt_path), "backbone_checkpoint_path": str(ckpt_path),
            "paired_method": "B9_DC_PHSR<->B3_Multitask_TCN_GRU (same backbone, same seed)",
            "note": "BACKFILLED from already-saved checkpoint via forward-pass only; NOT retrained.",
        }
        pd.DataFrame([b3_summary]).to_csv(b3_out / "metrics.csv", index=False, encoding="utf-8-sig")
        json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in b3_summary.items()}
        with open(b3_out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump([json_row], f, indent=2, default=str)
        pred_df_b3.to_csv(b3_out / "predictions.csv", index=False, encoding="utf-8-sig")
        cm3_df.to_csv(b3_out / "confusion_matrix.csv", encoding="utf-8-sig")
        with open(b3_out / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(b3_meta_json, f, indent=2, default=str)
        (b3_out / "DONE.flag").write_text("backfilled\n", encoding="utf-8")
        print(f"[backfilled] {task} seed{seed} B3 Acc={b3_summary['Acc']:.4f} MacroF1={b3_summary['Macro-F1']:.4f}")
        n_backfilled += 1

    print(f"\nDone {task}. backfilled={n_backfilled} skipped(already done)={n_skipped} missing_checkpoint={n_missing}")


if __name__ == "__main__":
    main()
