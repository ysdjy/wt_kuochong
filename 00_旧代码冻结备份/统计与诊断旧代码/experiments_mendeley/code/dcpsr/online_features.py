"""Causal, condition-relative online features.

For every raw feature x^(j) and every step t of a sequence, only observations
1..t of THAT sequence are used:

    x_rel   = (x_t - mean_{1..t}) / (std_{1..t} + eps)     historical z-score
    x_slope = x_t - x_{t-1}                                local variation
    x_rank  = (1/t) * #{k<=t : x_k <= x_t}                 online relative rank

These are computed over the FULL sequence in run order. Because a sequence is
assigned entirely to one split (tool-level holdout), this is both causal and
free of the train/serve skew that arises when the history is built on a
sequence with holes punched in it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets.base import META_COLS, OPTIONAL_META_COLS, DERIVED_COLS

EPS = 1e-12
SUFFIXES = ("rel", "slope", "rank")


def build_online_features(df: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    keep = [c for c in META_COLS + OPTIONAL_META_COLS + DERIVED_COLS if c in df.columns]
    parts = []
    for _, sub in df.groupby("sequence_id", sort=False):
        sub = sub.sort_values("order_key").reset_index(drop=True)
        out = sub[keep].copy()
        for col in raw_cols:
            x = pd.to_numeric(sub[col], errors="coerce").astype(float)
            x = x.replace([np.inf, -np.inf], np.nan)
            # causal fill: forward-fill, then the running median of the past
            x = x.ffill()
            x = x.fillna(x.expanding().median()).fillna(0.0)
            v = x.values
            mu = x.expanding().mean().values
            sd = x.expanding().std(ddof=0).fillna(0.0).values
            out[f"{col}__rel"] = (v - mu) / (sd + EPS)
            out[f"{col}__slope"] = np.concatenate([[0.0], np.diff(v)])
            out[f"{col}__rank"] = _online_rank(v)
        parts.append(out)
    res = pd.concat(parts).sort_values(["sequence_id", "order_key"]).reset_index(drop=True)
    return res.replace([np.inf, -np.inf], np.nan)


def _online_rank(v: np.ndarray) -> np.ndarray:
    """rank_t = (1/t) * #{k<=t: v_k <= v_t}, O(n log n) via sorted insertion."""
    import bisect
    seen: list[float] = []
    out = np.empty(len(v), dtype=float)
    for i, x in enumerate(v):
        bisect.insort(seen, x)
        out[i] = bisect.bisect_right(seen, x) / (i + 1)
    return out


def online_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(tuple("__" + s for s in SUFFIXES))]
