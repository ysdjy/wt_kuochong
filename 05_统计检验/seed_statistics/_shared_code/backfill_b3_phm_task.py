# -*- coding: utf-8 -*-
r"""
Backfill paired B3 for already-completed B9_PHM2010_{task}_seed_landscape
seeds (D2/D3) that predate the B3-pairing patch, WITHOUT retraining --
loads the already-saved backbone checkpoint from backbone_checkpoints/ and
does a forward-pass-only reconstruction (same pattern as backfill_b3_phm_d1.py,
generalized to D2/D3's task-specific frozen preprocessing).

Usage:
    python backfill_b3_phm_task.py --task D2
    python backfill_b3_phm_task.py --task D3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
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
STAGE_ORDER = ["E", "M", "L"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_base():
    os.environ.setdefault("FGDS_RUN_DIR", str(Path(tempfile.gettempdir()) / "b3_backfill_task_base_side_outputs"))
    sys.path.insert(0, str(CODE_DIR))
    import main_experiment_3_fgds_psi_optimized as base  # noqa
    base.FEATURE_FILE = PROJECT_ROOT / "baselines" / "htt_net" / "data" / "run_level_features_all.csv"
    return base


def import_comparison_script(seed_tag: str):
    os.environ["COMPARISON_RECHECK_DIR"] = str(Path(tempfile.gettempdir()) / "b3_backfill_task_comparison_side_outputs" / seed_tag)
    script_path = CODE_DIR / "7.4对比实验.py"
    mod_name = f"b3_backfill_task_comparison_{seed_tag}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["D1", "D2", "D3"])
    args = ap.parse_args()
    task = args.task

    B9_DIR = SEED_STATS_DIR / f"B9_PHM2010_{task}_seed_landscape"
    B3_DIR = SEED_STATS_DIR / f"B3_PHM2010_{task}_seed_landscape"
    FROZEN_DIR = EXPAND_ROOT / "shared" / "reproducibility" / f"PHM2010_{task}_frozen_preprocess"

    base = import_base()
    with open(FROZEN_DIR / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        selected = json.load(f)["selected_features_in_order"]
    feat_test = pd.read_csv(FROZEN_DIR / "feat_test_frozen.csv")
    L = base.BEST_ARCH["L"]
    te_pack = base.make_pack(feat_test, selected, L, "test")
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)
    input_dim = len(selected)

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
        ckpt_path = B9_DIR / "backbone_checkpoints" / f"b3_backbone_{task}_seed{seed}.pth"
        if not ckpt_path.exists():
            print(f"[MISSING] seed{seed}: no checkpoint at {ckpt_path}")
            n_missing += 1
            continue

        with open(sd / "run_meta.json", "r", encoding="utf-8") as f:
            b9_meta = json.load(f)

        import torch
        model = base.TCNGRUMultiTask(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"])
        state = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(state)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        mod = import_comparison_script(f"{task}_seed{seed}")
        pred_test_raw = base.predict_model(model, te_pack)
        pred_test = base.apply_probability_inference(pred_test_raw, mod.B12_PARAMS)

        y_pred_b3 = pred_test["stage_pred_raw"].values.astype(int)
        prob_b3 = pred_test[[f"raw_prob_{s}" for s in base.STAGE_NAMES]].values

        row_b3 = mod.metrics_row("B3", f"Multi-task TCN-GRU (B9 backbone, raw pre-inference, BACKFILLED) -- {task}",
                                  "Relative-stage", "TCN-GRU + auxiliary heads", y_true, y_pred_b3, prob_b3)
        row3 = row_b3.iloc[0] if hasattr(row_b3, "iloc") else row_b3

        q_hat = q_true = None
        if "q_hat" in pred_test.columns and "q_true" in pred_test.columns:
            q_hat = pred_test["q_hat"].values.astype(float)
            q_true = pred_test["q_true"].values.astype(float)
        q_mae = q_rmse = q_r2 = spearman = np.nan
        if q_hat is not None:
            q_mae = mean_absolute_error(q_true, q_hat)
            q_rmse = np.sqrt(mean_squared_error(q_true, q_hat))
            q_r2 = r2_score(q_true, q_hat)
            if spearmanr is not None:
                try:
                    spearman = float(spearmanr(q_true, q_hat).correlation)
                except Exception:
                    spearman = np.nan

        b3_summary = {
            "Method": "B3_Multitask_TCN_GRU", "Dataset": "PHM2010", "Task": task, "Seed": seed,
            "Acc": row3["Acc"], "Macro-F1": row3["Macro-F1"], "E-F1": row3["E-F1"], "M-F1": row3["M-F1"],
            "L-F1": row3["L-F1"], "M-Precision": row3["M-Pre"], "M-Recall": row3["M-Rec"],
            "M_to_E": row3["M→E"], "M_to_L": row3["M→L"], "Rev": row3["Rev"], "Jump": row3["Jump"],
            "Smooth": row3["Smooth"], "q-MAE": q_mae, "q-RMSE": q_rmse, "q-R2": q_r2, "Spearman": spearman,
            "best_epoch": b9_meta.get("best_epoch"), "backfilled": True,
        }
        pred_df_b3 = meta[["condition", "cut_index", "stage_true"]].copy()
        pred_df_b3["pred"] = [base.ID_TO_STAGE[int(v)] for v in y_pred_b3]
        pred_df_b3["prob_E"], pred_df_b3["prob_M"], pred_df_b3["prob_L"] = prob_b3[:, 0], prob_b3[:, 1], prob_b3[:, 2]
        if q_hat is not None:
            pred_df_b3["q_hat"] = q_hat
            pred_df_b3["q_true"] = q_true

        cm3 = confusion_matrix(y_true, y_pred_b3, labels=[0, 1, 2])
        cm3_df = pd.DataFrame(cm3, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

        b3_out.mkdir(parents=True, exist_ok=True)
        b3_meta_json = {
            "method": "B3_Multitask_TCN_GRU", "dataset": "PHM2010", "task": task, "train_seed": seed,
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
