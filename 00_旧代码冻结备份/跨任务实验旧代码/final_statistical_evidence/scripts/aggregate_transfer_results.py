# -*- coding: utf-8 -*-
"""
Aggregate D1 (reused fixed-model point estimate, NOT retrained) + D2 + D3
(freshly trained under TRAIN_SEED=42, frozen config) into:

    results/TRANSFER_TASKS_D1_D2_D3.csv   -- long format, one row per (method, task)
    results/TRANSFER_TASKS_MEAN_STD.csv   -- one row per method, mean+-std (ddof=1) across the 3 tasks

D1's numbers are the SAME point estimates already computed for the
bootstrap table (results/D1_MAIN_BOOTSTRAP_CI.csv) -- D1 is never retrained
here, per the task's explicit instruction. D2/D3 numbers come from
transfer_tasks/D{2,3}/<method>/metrics.json, written by
run_internal_methods_transfer_task.py or a published-method launcher.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FSE_ROOT = Path(__file__).resolve().parents[1]
BOOT_DIR = FSE_ROOT / "bootstrap"
TRANSFER_ROOT = FSE_ROOT / "transfer_tasks"
RESULTS_DIR = FSE_ROOT / "results"

METHODS = ["rf", "tcn_gru", "multitask_tcn_gru", "dc_psr", "htt_net",
           "multi_source_attention", "mtf_avitk", "dynamic_gin_tgp", "dp2net_adapted"]
DISPLAY_NAME = {
    "rf": "RF", "tcn_gru": "TCN-GRU", "multitask_tcn_gru": "Multi-task TCN-GRU",
    "dc_psr": "DC-PSR", "htt_net": "HTT-Net (adapted)",
    "multi_source_attention": "Multi-source Attention", "mtf_avitk": "MTF-AViTK",
    "dynamic_gin_tgp": "Dynamic GIN + TGP", "dp2net_adapted": "DP2Net-adapted",
}
METRICS = ["Acc", "MacroF1", "E_F1", "M_F1", "L_F1", "M_Precision", "M_Recall",
           "M_to_E", "M_to_L", "Rev", "Jump", "Smooth"]


def d1_point_estimate(method: str) -> dict | None:
    cfg_path = BOOT_DIR / method / "bootstrap_config.json"
    summ_path = BOOT_DIR / method / "bootstrap_summary.csv"
    if not (cfg_path.exists() and summ_path.exists()):
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    summ = pd.read_csv(summ_path).set_index("metric")["point_estimate"].to_dict()
    row = {m: summ.get(m) for m in METRICS if m in summ}
    row.update(cfg["point_estimate_sequence_metrics"])
    return row


def transfer_metrics(task: str, method: str) -> dict | None:
    p = TRANSFER_ROOT / task / method / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_rows = []
    missing = []
    for method in METHODS:
        for task in ["D1", "D2", "D3"]:
            if task == "D1":
                m = d1_point_estimate(method)
                source = "reused fixed-model point estimate (results/D1_MAIN_BOOTSTRAP_CI.csv)"
            else:
                m = transfer_metrics(task, method)
                source = f"freshly trained, TRAIN_SEED=42 (transfer_tasks/{task}/{method}/metrics.json)"
            if m is None:
                missing.append((task, method))
                continue
            row = {"Method": DISPLAY_NAME[method], "method_id": method, "Task": task, "source": source}
            row.update({k: m.get(k) for k in METRICS})
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_path = RESULTS_DIR / "TRANSFER_TASKS_D1_D2_D3.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

    if missing:
        print(f"\nWARNING: {len(missing)} (task, method) cells missing, excluded from mean+-std:")
        for task, method in missing:
            print(f"  {task}/{method}")

    summary_rows = []
    for method in METHODS:
        sub = long_df[long_df["method_id"] == method]
        if len(sub) == 0:
            continue
        row = {"Method": DISPLAY_NAME[method], "n_tasks": len(sub)}
        for metric in METRICS:
            vals = sub[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "TRANSFER_TASKS_MEAN_STD.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df)} rows)")
    print("\nNote: std is sample std (ddof=1) ACROSS THE 3 TRANSFER TASKS per method -- "
          "not a random-seed std. Rows with n_tasks < 3 indicate incomplete D2/D3 coverage; "
          "see printed warnings above and STATUS.json.")
    print(summary_df[["Method", "n_tasks", "Acc_mean", "Acc_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
