# -*- coding: utf-8 -*-
"""
Shared D2/D3 condition-split monkeypatch + metric-key normalizer for the
4 published raw-signal baselines (multi_source_attention, mtf_avitk,
dynamic_gin_tgp, dp2net), which all share byte-for-byte the same
`baselines/<name>/data/label_utils.py::get_unified_split()` ->
`dcpsr_base.split_grouped_lifecycle(labeled)` pattern (confirmed by
direct grep across all 4 -- identical docstring/implementation).

`split_grouped_lifecycle` is hardcoded train={C1,C4}, test={C6}. Rather
than edit any file under baselines/ (must stay read-only per the freeze),
each per-method launcher monkeypatches the ONE shared `dcpsr_base` module
object's `.split_grouped_lifecycle` attribute to a parameterized
reimplementation before calling that baseline's `run_protocol_b()`. Since
`dcpsr_base` is a single module object imported by reference everywhere
(`from label_utils import dcpsr_base`), patching the attribute on the
module object itself (not a local name binding) affects every caller.

This performs the *exact same split logic* already validated in
`final_statistical_evidence/scripts/methods/common_pipeline.py::split_by_conditions`
(itself copied from `代码/7.7跨工况实验.py`), just against the raw-signal
baselines' `label_df` schema (identical columns: condition, run_id, stage,
stage_id, ...).
"""
from __future__ import annotations


def make_split_fn(base, train_conditions, test_condition):
    """Returns a function with the same signature as
    dcpsr_base.split_grouped_lifecycle(df) -> (train_df, val_df, test_df)."""
    def split_fn(df):
        if set(train_conditions) & {test_condition}:
            raise ValueError(f"Leakage: train {train_conditions} overlaps test {test_condition}.")
        train_parts, val_parts = [], []
        for cond in train_conditions:
            sub = df[df["condition"] == cond].sort_values("run_id").reset_index(drop=True).copy()
            val_idx = []
            for st in base.STAGE_NAMES:
                gs = sub[sub["stage"] == st].sort_values("run_id")
                if len(gs) == 0:
                    continue
                n = max(base.MIN_STAGE_VAL_LEN, int(round(len(gs) * base.VAL_RATIO_STAGE)))
                n = min(n, max(len(gs) - 2, 1))
                start = max(0, (len(gs) - n) // 2)
                val_idx.extend(gs.iloc[start:start + n].index.tolist())
            val_idx = sorted(set(val_idx))
            val_parts.append(sub.loc[val_idx].copy())
            train_parts.append(sub.drop(index=val_idx).copy())
        import pandas as pd
        final_train = pd.concat(train_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
        final_val = pd.concat(val_parts).sort_values(["condition", "run_id"]).reset_index(drop=True)
        test_df = df[df["condition"] == test_condition].sort_values("run_id").reset_index(drop=True).copy()
        return final_train, final_val, test_df
    return split_fn


def apply_patch(dcpsr_base_module, train_conditions, test_condition):
    dcpsr_base_module.split_grouped_lifecycle = make_split_fn(dcpsr_base_module, train_conditions, test_condition)


# manuscript_metric_row() key -> this project's standardized metrics.json key
METRIC_KEY_MAP = {
    "Acc": "Acc",
    "Macro-F1": "MacroF1",
    "E-F1": "E_F1",
    "M-F1": "M_F1",
    "L-F1": "L_F1",
    "M-Pre": "M_Precision",
    "M-Rec": "M_Recall",
    "M→E": "M_to_E",
    "M→L": "M_to_L",
    "Rev": "Rev",
    "Jump": "Jump",
    "Smooth": "Smooth",
}


def normalize_metric_row(row: dict) -> dict:
    return {std_key: row[raw_key] for raw_key, std_key in METRIC_KEY_MAP.items() if raw_key in row}
