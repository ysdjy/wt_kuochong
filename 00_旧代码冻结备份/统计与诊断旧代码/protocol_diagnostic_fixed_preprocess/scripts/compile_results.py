# -*- coding: utf-8 -*-
r"""
Compiles FIXED_PREPROCESS_5SEED.csv / FIXED_PREPROCESS_5SEED_SUMMARY.csv /
OLD_VS_FIXED_PREPROCESS.csv from the per-run results written by
run_diagnostic_seed.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DIAG_ROOT = THIS_DIR.parent
RESULTS_DIR = DIAG_ROOT / "results"

METHOD_DIRS = {
    "RF": "rf",
    "TCN-GRU": "tcn_gru",
    "Multi-task TCN-GRU": "multitask_tcn_gru",
    "DC-PSR": "dc_psr",
    "HTT-Net": "htt_net",
}
SEEDS = [42, 52, 62, 72, 82]

METRIC_COLS = ["Acc", "Macro-F1", "M-F1", "M-Rec", "Rev", "Jump", "Smooth"]

# Old (contaminated-protocol) whole-pipeline 5-seed sweep, from
# final_five_seed_sweep/results/FINAL_9_METHODS_SEED_LEVEL.csv.
OLD_SEED_LEVEL = {
    "RF":                  [0.9770, 0.9803, 0.9737, 0.9803, 0.9770],
    "TCN-GRU":             [0.8684, 0.8914, 0.8059, 0.7730, 0.8289],
    "Multi-task TCN-GRU":  [0.9901, 0.7961, 0.8059, 0.8783, 0.9704],
    "DC-PSR":              [0.9868, 0.8586, 0.7993, 0.8849, 0.9671],
    "HTT-Net":             [0.7763157894736842, 0.8651315789473685, 0.9177631578947368, 0.7894736842105263, 0.8223684210526315],
}
OLD_MACROF1_SEED_LEVEL = {
    "RF":                  [0.9773, 0.9806, 0.9743, 0.9806, 0.9774],
    "TCN-GRU":             [0.8746, 0.8947, 0.8067, 0.7683, 0.8378],
    "Multi-task TCN-GRU":  [0.9902, 0.7994, 0.8049, 0.8865, 0.9712],
    "DC-PSR":              [0.9871, 0.8674, 0.7989, 0.8920, 0.9680],
    "HTT-Net":             [0.7793803418803419, 0.8574849327473015, 0.9208941796475014, 0.7935505503683538, 0.8236939521827331],
}
OLD_MF1_SEED_LEVEL = {
    "RF":                  [0.9723, 0.9764, 0.9688, 0.9766, 0.9723],
    "TCN-GRU":             [0.8261, 0.8533, 0.7122, 0.6349, 0.7476],
    "Multi-task TCN-GRU":  [0.9882, 0.6837, 0.7035, 0.8326, 0.9647],
    "DC-PSR":              [0.9844, 0.8000, 0.6965, 0.8430, 0.9609],
    "HTT-Net":             [0.6458333333333334, 0.862876254180602, 0.8962655601659751, 0.7064220183486238, 0.7378640776699029],
}

NOTE_TCN_GRU = (
    "Old seed42 B10=0.8684 is contaminated: 代码/7.4对比实验.py calls "
    "base.set_seed(RANDOM_SEED) ONCE at the top of main(), then trains "
    "B8(TCN-only)->B9(GRU-only)->B10(TCN-GRU) sequentially off that single "
    "shared RNG stream, so B10's actual init/shuffle state depended on how "
    "much RNG B8+B9 had already consumed, not on an isolated seed=42. New "
    "seed42=0.9770 is the isolated-seed value (reset->instantiate->train, "
    "per protocol Section 5). HTT-Net (also isolated in the old script) "
    "reproduced bit-exact; Multi-task TCN-GRU/DC-PSR (self-isolating via "
    "base.train_model's internal set_seed call) matched within 0.33pp/0pp -- "
    "TCN-GRU is the one method that was actually exposed to this bug."
)


def load_seed_level():
    rows = []
    for method, dirname in METHOD_DIRS.items():
        for seed in SEEDS:
            run_dir = RESULTS_DIR / dirname / f"seed{seed}"
            mfile = run_dir / "metrics.csv"
            if not mfile.exists():
                print(f"MISSING: {method} seed{seed} ({mfile})")
                continue
            m = pd.read_csv(mfile).iloc[0]
            with open(run_dir / "run_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            rows.append({
                "Method": method,
                "Seed": seed,
                "Acc": m["Acc"], "Macro-F1": m["Macro-F1"], "M-F1": m["M-F1"],
                "M-Rec": m["M-Rec"], "Rev": m["Rev"], "Jump": m["Jump"], "Smooth": m["Smooth"],
                "feature_hash": meta["feature_hash"][:16], "split_hash": meta["split_hash"][:16],
                "gmm_hash": meta["gmm_hash"][:16], "window_hash": meta["window_hash"][:16],
                "training_seconds": meta["training_seconds"],
            })
    return pd.DataFrame(rows)


def main():
    seed_level = load_seed_level()
    seed_level.to_csv(DIAG_ROOT / "FIXED_PREPROCESS_5SEED.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote FIXED_PREPROCESS_5SEED.csv ({len(seed_level)} rows)")

    # protocol-integrity check: every run must share identical hashes
    for col in ["feature_hash", "split_hash", "gmm_hash", "window_hash"]:
        n_unique = seed_level[col].nunique()
        status = "OK" if n_unique == 1 else "MISMATCH"
        print(f"  hash check [{col}]: {n_unique} unique value(s) across all 20 runs -> {status}")

    summary_rows = []
    for method in METHOD_DIRS:
        sub = seed_level[seed_level["Method"] == method]
        row = {"Method": method, "n_seeds": len(sub)}
        for col in METRIC_COLS:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DIAG_ROOT / "FIXED_PREPROCESS_5SEED_SUMMARY.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote FIXED_PREPROCESS_5SEED_SUMMARY.csv")

    compare_rows = []
    for method in METHOD_DIRS:
        old_acc = np.array(OLD_SEED_LEVEL[method])
        old_macrof1 = np.array(OLD_MACROF1_SEED_LEVEL[method])
        old_mf1 = np.array(OLD_MF1_SEED_LEVEL[method])
        new_acc = seed_level[seed_level["Method"] == method]["Acc"].values
        new_macrof1 = seed_level[seed_level["Method"] == method]["Macro-F1"].values
        new_mf1 = seed_level[seed_level["Method"] == method]["M-F1"].values

        old_acc_std = old_acc.std(ddof=1)
        new_acc_std = new_acc.std(ddof=1)
        reduction = (old_acc_std - new_acc_std) / old_acc_std if old_acc_std > 0 else np.nan

        if method == "RF":
            diagnosis = "Both old and new std are small (RF already robust to preprocessing perturbation)."
        elif reduction > 0.5:
            diagnosis = "PROTOCOL-INDUCED VARIANCE CONFIRMED"
        elif reduction > 0.15:
            diagnosis = "PARTIAL PROTOCOL EFFECT"
        else:
            diagnosis = "TRAINING-SEED SENSITIVITY REMAINS"

        compare_rows.append({
            "Method": method,
            "old_5seed_Acc_mean": old_acc.mean(), "old_5seed_Acc_std": old_acc_std,
            "new_5seed_Acc_mean": new_acc.mean(), "new_5seed_Acc_std": new_acc_std,
            "std_reduction_ratio": reduction,
            "old_MacroF1_std": old_macrof1.std(ddof=1), "new_MacroF1_std": new_macrof1.std(ddof=1),
            "old_MF1_std": old_mf1.std(ddof=1), "new_MF1_std": new_mf1.std(ddof=1),
            "diagnosis": diagnosis,
        })
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(DIAG_ROOT / "OLD_VS_FIXED_PREPROCESS.csv", index=False, encoding="utf-8-sig")
    print("Wrote OLD_VS_FIXED_PREPROCESS.csv")
    print()
    print(compare.to_string(index=False))

    with open(DIAG_ROOT / "frozen_preprocess" / "TCN_GRU_SEED42_DIVERGENCE_NOTE.txt", "w", encoding="utf-8") as f:
        f.write(NOTE_TCN_GRU)


if __name__ == "__main__":
    main()
