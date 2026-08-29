# -*- coding: utf-8 -*-
"""
D1 (C1+C4 -> C6) moving-block bootstrap, common 304-run universe.

Answers: "how much sample-level statistical uncertainty does the FIXED
official model's test performance carry on this particular test set?"
It is NOT a training-seed variance study (see final_five_seed_sweep/ for
that). No model is trained here -- this script only resamples the frozen
model's existing per-run predictions.

Block design: the 304-run test sequence (run_id 12..315) is one ordered,
overlapping-window time series per condition (window length L=12), so
adjacent rows are strongly autocorrelated. A plain i.i.d. bootstrap would
underestimate variance. We use a moving-block bootstrap: resample
contiguous blocks of BLOCK_LENGTH consecutive rows (with replacement)
and concatenate until the resampled series reaches the original length,
repeated N_BOOTSTRAP times, with a fixed random seed for reproducibility.

Metrics bootstrapped (pointwise, well-defined on any resampled bag of
rows): Acc, Macro-F1, E-F1, M-F1, L-F1, M-Precision, M-Recall, M->E, M->L.

Metrics NOT bootstrapped (sequence-order metrics -- resampling blocks
would inject artificial boundaries at block joins): Rev, Jump, Smooth.
These are reported as point estimates on the original ordered sequence
only, per PROTOCOL.md.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
PRED_DIR = BASE_DIR / "predictions_common_universe"
BOOT_DIR = BASE_DIR / "bootstrap"
RESULTS_DIR = BASE_DIR / "results"
BOOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_LENGTH = 12
N_BOOTSTRAP = 5000
RANDOM_SEED = 20260820

STAGE_MAP = {"early": 0, "middle": 1, "late": 2}
POINTWISE_METRICS = [
    "Acc", "MacroF1", "E_F1", "M_F1", "L_F1", "M_Precision", "M_Recall", "M_to_E", "M_to_L",
]

METHODS = [
    "rf", "tcn_gru", "multitask_tcn_gru", "dc_psr", "htt_net",
    "multi_source_attention", "mtf_avitk", "dynamic_gin_tgp", "dp2net_adapted",
]
DISPLAY_NAME = {
    "rf": "RF",
    "tcn_gru": "TCN-GRU",
    "multitask_tcn_gru": "Multi-task TCN-GRU",
    "dc_psr": "DC-PSR",
    "htt_net": "HTT-Net (adapted)",
    "multi_source_attention": "Multi-source Attention",
    "mtf_avitk": "MTF-AViTK",
    "dynamic_gin_tgp": "Dynamic GIN + TGP",
    "dp2net_adapted": "DP2Net-adapted",
}


def pointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p_each, r_each, f1_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "MacroF1": macro_f1,
        "E_F1": f1_each[0],
        "M_F1": f1_each[1],
        "L_F1": f1_each[2],
        "M_Precision": p_each[1],
        "M_Recall": r_each[1],
        "M_to_E": cmn[1, 0],
        "M_to_L": cmn[1, 2],
    }


def sequence_metrics(y_pred: np.ndarray, prob: np.ndarray) -> dict:
    if len(y_pred) <= 1:
        return {"Rev": 0, "Jump": 0, "Smooth": float("nan")}
    dy = np.diff(y_pred)
    return {
        "Rev": int(np.sum(dy < 0)),
        "Jump": int(np.sum(np.abs(dy) >= 2)),
        "Smooth": float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))),
    }


def moving_block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks_needed = int(np.ceil(n / block_length))
    max_start = n - block_length
    starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
    return idx


def run_method(method: str) -> dict:
    df = pd.read_csv(PRED_DIR / f"D1_{method}_304runs.csv").sort_values("run_id").reset_index(drop=True)
    n = len(df)
    y_true = df["true_stage"].map(STAGE_MAP).to_numpy()
    y_pred = df["pred_stage"].map(STAGE_MAP).to_numpy()
    prob = df[["p_early", "p_middle", "p_late"]].to_numpy(dtype=float)

    point = pointwise_metrics(y_true, y_pred)
    point.update(sequence_metrics(y_pred, prob))

    rng = np.random.default_rng(RANDOM_SEED)
    boot_rows = []
    for b in range(N_BOOTSTRAP):
        idx = moving_block_bootstrap_indices(n, BLOCK_LENGTH, rng)
        m = pointwise_metrics(y_true[idx], y_pred[idx])
        m["bootstrap_id"] = b
        boot_rows.append(m)
    boot_df = pd.DataFrame(boot_rows).set_index("bootstrap_id")

    method_dir = BOOT_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)
    boot_df.to_csv(method_dir / "bootstrap_samples.csv")

    summary_rows = []
    for metric in POINTWISE_METRICS:
        vals = boot_df[metric].to_numpy()
        summary_rows.append({
            "metric": metric,
            "point_estimate": point[metric],
            "bootstrap_mean": float(np.mean(vals)),
            "bootstrap_std": float(np.std(vals, ddof=1)),
            "ci_2.5": float(np.percentile(vals, 2.5)),
            "ci_97.5": float(np.percentile(vals, 97.5)),
            "n_bootstrap": N_BOOTSTRAP,
            "block_length": BLOCK_LENGTH,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(method_dir / "bootstrap_summary.csv", index=False)

    config = {
        "method": method,
        "display_name": DISPLAY_NAME[method],
        "block_length": BLOCK_LENGTH,
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "n_test_runs": n,
        "source_prediction_file": str((PRED_DIR / f"D1_{method}_304runs.csv").relative_to(REPO_ROOT)),
        "point_estimate_sequence_metrics": {k: point[k] for k in ["Rev", "Jump", "Smooth"]},
        "note": "Rev/Jump/Smooth are point estimates only (no bootstrap CI) -- block resampling injects artificial sequence-boundary discontinuities into these order-dependent metrics; see PROTOCOL.md.",
    }
    (method_dir / "bootstrap_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {"method": method, "point": point, "summary": summary_df}


def main():
    all_results = {}
    for method in METHODS:
        print(f"Bootstrapping {method} ...")
        all_results[method] = run_method(method)

    main_rows = []
    for method in METHODS:
        r = all_results[method]
        point = r["point"]
        summ = r["summary"].set_index("metric")

        def ci(metric, col):
            return summ.loc[metric, col]

        main_rows.append({
            "Method": DISPLAY_NAME[method],
            "Acc": point["Acc"], "Acc_CI_low": ci("Acc", "ci_2.5"), "Acc_CI_high": ci("Acc", "ci_97.5"),
            "MacroF1": point["MacroF1"], "MacroF1_CI_low": ci("MacroF1", "ci_2.5"), "MacroF1_CI_high": ci("MacroF1", "ci_97.5"),
            "M_F1": point["M_F1"], "M_F1_CI_low": ci("M_F1", "ci_2.5"), "M_F1_CI_high": ci("M_F1", "ci_97.5"),
            "M_Rec": point["M_Recall"], "M_Rec_CI_low": ci("M_Recall", "ci_2.5"), "M_Rec_CI_high": ci("M_Recall", "ci_97.5"),
            "M_to_E": point["M_to_E"],
            "M_to_L": point["M_to_L"],
            "Rev": point["Rev"],
            "Jump": point["Jump"],
            "Smooth": point["Smooth"],
        })
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(RESULTS_DIR / "D1_MAIN_BOOTSTRAP_CI.csv", index=False)
    print(main_df.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'D1_MAIN_BOOTSTRAP_CI.csv'}")


if __name__ == "__main__":
    main()
