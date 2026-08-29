# -*- coding: utf-8 -*-
"""
Build the common D1 (C1+C4 -> C6) test universe: 304 runs, run_id 12-315.

Reads each of the 9 methods' AUTHORITATIVE, already-frozen D1 prediction
file (the fixed official model used for the manuscript comparison table --
NOT any 5-seed-sweep artifact) and normalizes it to one shared schema:

    condition, run_id, true_stage, pred_stage, p_early, p_middle, p_late

then filters to run_id in [12, 315] (304 rows) and writes
predictions_common_universe/D1_<method>_304runs.csv.

Read-only on all source files. Does not train anything.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1] / "predictions_common_universe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_COLS = ["condition", "run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]

WINDOWED_SRC = REPO_ROOT / "补充材料/小论文/4_comparison_experiment_recheck/1_results/FINAL_comparison_predictions.csv"
HTT_SRC = REPO_ROOT / "outputs/htt_net/D1_C1C4_to_C6_SOURCE_ONLY_TUNED/test_predictions.csv"

NATIVE_SRC = {
    "multi_source_attention": REPO_ROOT / "outputs/multi_source_attention/unified_protocol/test_predictions.csv",
    "mtf_avitk": REPO_ROOT / "outputs/mtf_avitk/seed_sweep/seed42/unified_protocol/test_predictions.csv",
    "dynamic_gin_tgp": REPO_ROOT / "outputs/dynamic_gin_tgp/unified_protocol/seed42/run_predictions.csv",
    "dp2net_adapted": REPO_ROOT / "outputs/dp2net/unified_protocol_B-D1/seed42/predictions.csv",
}

# method -> B-slot in the windowed comparison file
WINDOWED_METHODS = {
    "rf": "B5",
    "tcn_gru": "B10",
    "multitask_tcn_gru": "B11",
    "dc_psr": "B12",
}


def load_windowed(df_raw: pd.DataFrame, b_slot: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "condition": df_raw["condition"],
        "run_id": df_raw["run_id_end"].astype(int),
        "true_stage": df_raw["true_stage"],
        "pred_stage": df_raw[f"pred_{b_slot}"],
        "p_early": df_raw[f"prob_E_{b_slot}"],
        "p_middle": df_raw[f"prob_M_{b_slot}"],
        "p_late": df_raw[f"prob_L_{b_slot}"],
    })
    return out


def load_native(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"run_id_end": "run_id"})
    return df[COMMON_COLS]


def main():
    manifest = []

    df_windowed_raw = pd.read_csv(WINDOWED_SRC)
    for method, b_slot in WINDOWED_METHODS.items():
        df = load_windowed(df_windowed_raw, b_slot)
        df = df.sort_values("run_id").reset_index(drop=True)
        assert df["run_id"].min() == 12 and df["run_id"].max() == 315 and len(df) == 304, (
            f"{method}: unexpected universe {df['run_id'].min()}-{df['run_id'].max()} n={len(df)}"
        )
        out_path = OUT_DIR / f"D1_{method}_304runs.csv"
        df.to_csv(out_path, index=False)
        manifest.append((method, str(out_path.relative_to(REPO_ROOT)), str(WINDOWED_SRC.relative_to(REPO_ROOT)), len(df)))

    df_htt = load_native(HTT_SRC)
    df_htt = df_htt[(df_htt["run_id"] >= 12) & (df_htt["run_id"] <= 315)].sort_values("run_id").reset_index(drop=True)
    assert len(df_htt) == 304, f"htt_net: n={len(df_htt)}"
    out_path = OUT_DIR / "D1_htt_net_304runs.csv"
    df_htt.to_csv(out_path, index=False)
    manifest.append(("htt_net", str(out_path.relative_to(REPO_ROOT)), str(HTT_SRC.relative_to(REPO_ROOT)), len(df_htt)))

    for method, src in NATIVE_SRC.items():
        df = load_native(src)
        assert df["run_id"].min() == 1 and df["run_id"].max() == 315 and len(df) == 315, (
            f"{method}: expected native 315-run universe, got {df['run_id'].min()}-{df['run_id'].max()} n={len(df)}"
        )
        df_common = df[(df["run_id"] >= 12) & (df["run_id"] <= 315)].sort_values("run_id").reset_index(drop=True)
        assert len(df_common) == 304, f"{method}: n={len(df_common)}"
        out_path = OUT_DIR / f"D1_{method}_304runs.csv"
        df_common.to_csv(out_path, index=False)
        manifest.append((method, str(out_path.relative_to(REPO_ROOT)), str(src.relative_to(REPO_ROOT)), len(df_common)))

    manifest_df = pd.DataFrame(manifest, columns=["method", "output_file", "source_file", "n_rows"])
    manifest_path = OUT_DIR / "D1_MANIFEST.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(manifest_df.to_string(index=False))
    print(f"\nWrote {len(manifest)} common-universe files + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
