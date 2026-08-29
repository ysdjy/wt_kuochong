# -*- coding: utf-8 -*-
r"""
Standalone re-evaluation of a saved HTT-Net checkpoint on the D1 (C1+C4 -> C6)
test split, without retraining. Rebuilds the identical data pipeline used by
train.py (same seed, same feature selection, same scaler-fit-on-train), loads
the checkpoint, and regenerates test_predictions.csv / metrics.json /
confusion_matrix.csv.

Usage:
    python evaluate.py --checkpoint outputs/htt_net/D1_C1C4_to_C6/checkpoint_best.pth
    python evaluate.py --checkpoint <path> --feature-file <path to run_level_features_all.csv>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train as T  # reuse prepare_data / model config / metrics from train.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to the checkpoint's own directory")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    T.base.set_seed(T.TRAIN_CFG["seed"])

    feature_file = Path(args.feature_file) if args.feature_file else None
    if feature_file is None and not T.base.FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found at default path {T.base.FEATURE_FILE}. "
            "Pass --feature-file or set HTT_FEATURE_FILE."
        )

    tr_pack, va_pack, te_pack, selected, scaler = T.prepare_data(feature_file)

    model = T.HTTNet(input_dim=len(selected), num_classes=3, **T.HTT_ARCH)
    state = torch.load(ckpt_path, map_location=T.base.DEVICE)
    model.load_state_dict(state)
    model.to(T.base.DEVICE)

    y_pred, prob = T.predict_stage_model(model, te_pack)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = meta["stage_true_id"].values.astype(int)

    mrow = T.metrics_row(y_true, y_pred, prob)
    mrow["n_params"] = model.num_parameters()
    mrow["checkpoint"] = str(ckpt_path)

    pred_table = meta[["condition", "cut_index", "stage_true"]].rename(
        columns={"cut_index": "run_id_end", "stage_true": "true_stage"}
    ).copy()
    id_to_stage = T.base.ID_TO_STAGE
    pred_table["pred_stage"] = [id_to_stage[int(v)] for v in y_pred]
    pred_table["p_early"] = prob[:, 0]
    pred_table["p_middle"] = prob[:, 1]
    pred_table["p_late"] = prob[:, 2]

    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=["true_early", "true_middle", "true_late"], columns=["pred_early", "pred_middle", "pred_late"])

    pd.DataFrame([mrow]).to_csv(output_dir / "reeval_metrics_summary.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(output_dir / "reeval_test_predictions.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(output_dir / "reeval_confusion_matrix.csv", encoding="utf-8-sig")
    with open(output_dir / "reeval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in mrow.items()}, f, indent=2, ensure_ascii=False)

    print(f"Acc={mrow['Acc']:.4f}  Macro-F1={mrow['Macro-F1']:.4f}  M-F1={mrow['M-F1']:.4f}")
    print(f"Re-evaluation outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
