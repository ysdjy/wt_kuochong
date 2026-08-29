# -*- coding: utf-8 -*-
r"""
Extract raw relative features and shared latent representation h_{c,t}.

This script reuses the existing main experiment pipeline and checkpoint.
It does not retrain the model. It rebuilds the split-safe online features,
loads the trained multi-task TCN-GRU checkpoint, and exports:

- repr_raw_features.csv
- repr_hidden_hct.csv
- repr_dcpsr_final_state.csv

Default output:
    小论文/10_第五章顶刊风格可视化/figures_representation_space
"""



from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import main_experiment_3_fgds_psi_optimized as base  # noqa: E402


ROOT = Path(os.environ.get(
    "PAPER_ROOT",
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文",
))
OUT_ROOT = Path(os.environ.get(
    "CH5_VIS_OUT_ROOT",
    str(ROOT / "10_第五章顶刊风格可视化"),
))
OUT_DIR = OUT_ROOT / "figures_representation_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Redirect incidental CSV exports from reused base helpers to this folder so
# existing main-experiment outputs are not overwritten.
base.DIR_RESULT = OUT_DIR

MODEL_PATH = ROOT / "3_main_experiment_fgds_psi" / "2_models" / "fgds_psi_best_model.pth"
SELECTED_PATH = ROOT / "3_main_experiment_fgds_psi" / "1_results" / "selected_features.csv"
DCPSR_PROB_PATH = ROOT / "6_ablation_experiment" / "ablation_probabilities_test_C6.csv"


def read_selected_features() -> list[str]:
    if not SELECTED_PATH.exists():
        raise FileNotFoundError(f"Cannot find selected feature file: {SELECTED_PATH}")
    df = pd.read_csv(SELECTED_PATH)
    col = "feature" if "feature" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def build_feature_packs(selected: list[str]):
    """Rebuild split-safe feature tensors with the same preprocessing logic."""
    base.set_seed(base.RANDOM_SEED)
    raw_df = base.load_feature_table()
    label_df, _ = base.define_condition_relative_stages(raw_df)
    final_train_raw, final_val_raw, test_c6_raw = base.split_grouped_lifecycle(label_df)

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

    feat_train, feat_val = base.fill_by_train_median(feat_train, feat_val, selected)
    _, feat_test = base.fill_by_train_median(feat_train, feat_test, selected)

    gmm, raw_to_order = base.fit_train_gmm(feat_train)
    feat_train = base.assign_fine_states(feat_train, gmm, raw_to_order)
    feat_val = base.assign_fine_states(feat_val, gmm, raw_to_order)
    feat_test = base.assign_fine_states(feat_test, gmm, raw_to_order)

    scaler = base.StandardScaler().fit(feat_train[selected].values)
    for df in [feat_train, feat_val, feat_test]:
        df[selected] = np.nan_to_num(
            scaler.transform(df[selected].values),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    L = base.BEST_ARCH["L"]
    packs = {
        "final_train": base.make_pack(feat_train, selected, L, "final_train"),
        "final_internal_val": base.make_pack(feat_val, selected, L, "final_internal_val"),
        "test_C6": base.make_pack(feat_test, selected, L, "test_C6"),
    }
    return packs


def load_model(input_dim: int):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find trained model checkpoint: {MODEL_PATH}")
    model = base.TCNGRUMultiTask(
        input_dim,
        base.BEST_ARCH["channels"],
        base.BEST_ARCH["gru_hidden"],
        base.BEST_ARCH["dropout"],
    ).to(base.DEVICE)
    state = base.torch.load(MODEL_PATH, map_location=base.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_pack(model, pack, selected: list[str], split_name: str):
    import torch

    X = torch.tensor(pack["X"], dtype=torch.float32, device=base.DEVICE)
    rows = []
    hidden_rows = []
    raw_rows = []

    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            h = model.tcn(xb.transpose(1, 2)).transpose(1, 2)
            h, _ = model.gru(h)
            z = model.shared(h[:, -1, :])
            stage_logits = model.stage_head(z)
            q_hat = torch.sigmoid(model.q_head(z)).reshape(-1)
            prob = torch.softmax(stage_logits, dim=1)

            z_np = z.detach().cpu().numpy()
            prob_np = prob.detach().cpu().numpy()
            q_np = q_hat.detach().cpu().numpy()
            raw_last = xb[:, -1, :].detach().cpu().numpy()

            meta = pack["meta"].iloc[start:start + len(xb)].reset_index(drop=True)
            pred_id = prob_np.argmax(axis=1)

            for i in range(len(meta)):
                m = meta.iloc[i].to_dict()
                sample_id = f"{split_name}_{int(start+i):05d}"
                common = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "condition": m["condition"],
                    "run_id": int(m["cut_index"]),
                    "true_stage": m["stage_true"],
                    "true_stage_id": int(m["stage_true_id"]),
                    "pred_stage": base.ID_TO_STAGE[int(pred_id[i])],
                    "q_true": float(m["q_true"]),
                    "q_hat": float(q_np[i]),
                    "p_E": float(prob_np[i, 0]),
                    "p_M": float(prob_np[i, 1]),
                    "p_L": float(prob_np[i, 2]),
                    "uncertainty": float(1.0 - np.max(prob_np[i])),
                    "entropy": float(-(prob_np[i] * np.log(prob_np[i] + 1e-12)).sum() / np.log(3.0)),
                    "misclassified": int(base.ID_TO_STAGE[int(pred_id[i])] != str(m["stage_true"])),
                }
                rows.append(common)

                hrow = common.copy()
                for j in range(z_np.shape[1]):
                    hrow[f"h_{j:02d}"] = float(z_np[i, j])
                hidden_rows.append(hrow)

                rrow = common.copy()
                for j, feat in enumerate(selected):
                    rrow[feat] = float(raw_last[i, j])
                raw_rows.append(rrow)

    return pd.DataFrame(rows), pd.DataFrame(raw_rows), pd.DataFrame(hidden_rows)


def export_dcpsr_final_state():
    if not DCPSR_PROB_PATH.exists():
        print(f"Skip DC-PSR final state export; missing: {DCPSR_PROB_PATH}")
        return None
    df = pd.read_csv(DCPSR_PROB_PATH)
    required = ["run_id", "true_stage", "q_hat", "p_final_E", "p_final_M", "p_final_L", "pred_A6"]
    if any(c not in df.columns for c in required):
        print("Skip DC-PSR final state export; required columns missing.")
        return None
    out = pd.DataFrame({
        "sample_id": [f"test_C6_{i:05d}" for i in range(len(df))],
        "split": "test_C6",
        "condition": df.get("condition", "C6"),
        "run_id": pd.to_numeric(df["run_id"], errors="coerce"),
        "true_stage": df["true_stage"].astype(str),
        "pred_stage": df["pred_A6"].astype(str),
        "q_true": np.nan,
        "q_hat": pd.to_numeric(df["q_hat"], errors="coerce"),
        "p_E": pd.to_numeric(df["p_final_E"], errors="coerce"),
        "p_M": pd.to_numeric(df["p_final_M"], errors="coerce"),
        "p_L": pd.to_numeric(df["p_final_L"], errors="coerce"),
    })
    prob = out[["p_E", "p_M", "p_L"]].values
    out["uncertainty"] = 1.0 - np.max(prob, axis=1)
    out["entropy"] = -(prob * np.log(prob + 1e-12)).sum(axis=1) / np.log(3.0)
    out["misclassified"] = (out["true_stage"].str.lower() != out["pred_stage"].str.lower()).astype(int)
    path = OUT_DIR / "repr_dcpsr_final_state.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")
    return out


def main():
    print("=" * 100)
    print("Extract shared latent representation h_ct")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 100)

    selected = read_selected_features()
    packs = build_feature_packs(selected)
    model = load_model(len(selected))

    meta_parts, raw_parts, hidden_parts = [], [], []
    for split_name, pack in packs.items():
        meta, raw, hidden = infer_pack(model, pack, selected, split_name)
        meta_parts.append(meta)
        raw_parts.append(raw)
        hidden_parts.append(hidden)
        print(f"{split_name}: {len(meta)} samples")

    raw_df = pd.concat(raw_parts, ignore_index=True)
    hidden_df = pd.concat(hidden_parts, ignore_index=True)

    raw_path = OUT_DIR / "repr_raw_features.csv"
    hidden_path = OUT_DIR / "repr_hidden_hct.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    hidden_df.to_csv(hidden_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {raw_path}")
    print(f"Saved: {hidden_path}")

    export_dcpsr_final_state()

    try:
        shutil.copy2(Path(__file__).resolve(), OUT_DIR / "extract_hidden_representation.py")
    except Exception:
        pass

    print("=" * 100)
    print("Finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()
