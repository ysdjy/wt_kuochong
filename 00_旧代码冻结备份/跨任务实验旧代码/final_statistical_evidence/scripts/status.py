# -*- coding: utf-8 -*-
"""python final_statistical_evidence/scripts/status.py"""
from __future__ import annotations
import json
from pathlib import Path

FSE_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = FSE_ROOT / "STATUS.json"
DISPLAY = {
    "rf": "RF", "tcn_gru": "TCN-GRU", "multitask_tcn_gru": "Multi-task TCN-GRU",
    "dc_psr": "DC-PSR", "htt_net": "HTT-Net (adapted)",
    "multi_source_attention": "Multi-source Attention", "mtf_avitk": "MTF-AViTK",
    "dynamic_gin_tgp": "Dynamic GIN + TGP", "dp2net_adapted": "DP2Net-adapted",
}
ORDER = ["rf", "tcn_gru", "multitask_tcn_gru", "dc_psr", "htt_net",
         "multi_source_attention", "dp2net_adapted", "dynamic_gin_tgp", "mtf_avitk"]


def main():
    d1_bootstrap = (FSE_ROOT / "results" / "D1_MAIN_BOOTSTRAP_CI.csv").exists()
    print(f"D1 Bootstrap: {'DONE' if d1_bootstrap else 'PENDING'}\n")

    if not STATUS_PATH.exists():
        print("STATUS.json not found -- run_transfer_tasks.py has not been launched yet.")
        return
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    for task in ["D2", "D3"]:
        print(f"{task}:")
        for m in ORDER:
            state = status.get("tasks", {}).get(task, {}).get(m, "PENDING")
            print(f"  {DISPLAY[m]:<28} {state}")
        print()

    agg_done = (FSE_ROOT / "results" / "TRANSFER_TASKS_MEAN_STD.csv").exists()
    print(f"Cross-task aggregate: {'DONE' if agg_done else 'PENDING'}")


if __name__ == "__main__":
    main()
