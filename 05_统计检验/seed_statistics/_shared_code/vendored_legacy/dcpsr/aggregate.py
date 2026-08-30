"""mean +/- std aggregation.

TWO DIFFERENT STANDARD DEVIATIONS -- never merged.

    std_seed : within one task, across the 5 fixed seeds.  This is the
               run-to-run uncertainty that belongs after a number in a
               per-task table ("0.9868 +/- 0.0031").
    std_task : across tasks, computed from the PER-TASK MEANS.  This is
               cross-domain stability.

Pooling all task x seed numbers into one std is explicitly NOT done: it
conflates two different sources of variation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import DETERMINISTIC_METHODS
from .metrics import METRIC_COLS, Q_METRIC_COLS

ALL_METRICS = METRIC_COLS + Q_METRIC_COLS
KEY = ["dataset", "channel_set", "group", "task", "Method"]


def _present(df):
    return [m for m in ALL_METRICS if m in df.columns]


def by_seed(units: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(units, ignore_index=True)
    front = [c for c in KEY + ["seed", "kind", "variant"] if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def mean_std_by_task(bs: pd.DataFrame) -> pd.DataFrame:
    """Per (task, method): mean over seeds and std_seed."""
    mets = _present(bs)
    g = bs.groupby([c for c in KEY if c in bs.columns], dropna=False)
    out = g[mets].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{'mean' if s == 'mean' else ('std_seed' if s == 'std' else 'n_seeds')}"
                   for m, s in out.columns]
    out = out.reset_index()
    for m in mets:
        c = f"{m}_std_seed"
        out[c] = out[c].fillna(0.0)
    out["deterministic"] = out["Method"].isin(DETERMINISTIC_METHODS)
    # a deterministic method has zero run-to-run variance by construction
    for m in mets:
        out.loc[out["deterministic"], f"{m}_std_seed"] = 0.0
    return out


def cross_task_summary(ms: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """Per method: mean of the per-task means, and std_task across tasks."""
    d = ms if group is None else ms[ms["group"] == group]
    mets = [c[:-5] for c in d.columns if c.endswith("_mean") and c[:-5] in ALL_METRICS]
    keys = [c for c in ["dataset", "channel_set", "group", "Method"] if c in d.columns]
    rows = []
    for k, sub in d.groupby(keys, dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n_tasks"] = int(sub["task"].nunique())
        for m in mets:
            v = sub[f"{m}_mean"].astype(float)
            r[f"{m}_mean_tasks"] = float(v.mean())
            r[f"{m}_std_task"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(mean, std, nd=4) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{nd}f} ± {std:.{nd}f}" if np.isfinite(std) else f"{mean:.{nd}f}"


def publication_table(ms: pd.DataFrame, metrics=("Acc", "Macro_F1", "M_F1", "M_Rec", "Smooth"),
                      std_kind: str = "seed") -> pd.DataFrame:
    """Human-readable table. The numeric source columns are always kept too."""
    suf = "_std_seed" if std_kind == "seed" else "_std_task"
    base = "_mean" if std_kind == "seed" else "_mean_tasks"
    keys = [c for c in ["dataset", "channel_set", "group", "task", "Method", "n_seeds", "n_tasks"]
            if c in ms.columns]
    out = ms[keys].copy()
    for m in metrics:
        mc, sc = f"{m}{base}", f"{m}{suf}"
        if mc in ms.columns:
            out[m] = [fmt(a, b) for a, b in zip(ms[mc], ms.get(sc, pd.Series(np.nan, index=ms.index)))]
    out["std_kind"] = std_kind
    return out


def ablation_complete_table(ms: pd.DataFrame) -> pd.DataFrame:
    """Merged replacement for the old 'ablation settings' + 'ablation results'
    tables. Keeps raw numbers alongside the formatted strings."""
    conf = {
        "A1": dict(configuration="Raw stage head only", Raw="Y", Fine="", Prior="", Ordered="", Final_fusion=""),
        "A2": dict(configuration="Raw + fine-state head", Raw="Y", Fine="Y", Prior="", Ordered="", Final_fusion=""),
        "A3": dict(configuration="Raw + degradation-position prior", Raw="Y", Fine="", Prior="Y", Ordered="", Final_fusion=""),
        "A4": dict(configuration="Mix (raw + fine + prior)", Raw="Y", Fine="Y", Prior="Y", Ordered="", Final_fusion=""),
        "A5": dict(configuration="Causal ordered filtering of the mix", Raw="Y", Fine="Y", Prior="Y", Ordered="Y", Final_fusion=""),
        "A6": dict(configuration="Final fusion (DC-PSR)", Raw="Y", Fine="Y", Prior="Y", Ordered="Y", Final_fusion="Y"),
    }
    d = ms[ms["Method"].isin(conf)].copy()
    for c, v in pd.DataFrame(conf).T.items():
        pass
    meta = pd.DataFrame(conf).T.rename_axis("Method").reset_index()
    out = meta.merge(d, on="Method", how="right")
    for m in ("Acc", "Macro_F1", "E_F1", "M_F1", "L_F1", "M_Pre", "M_Rec", "Smooth"):
        if f"{m}_mean" in out.columns:
            out[f"{m}_display"] = [fmt(a, b) for a, b in zip(out[f"{m}_mean"], out[f"{m}_std_seed"])]
    return out.sort_values(["task", "Method"]).reset_index(drop=True)
