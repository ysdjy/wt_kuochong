"""Train / internal-validation / test construction.

HARD RULES
----------
1. Test sequences never touch feature selection, scaling, GMM fitting, early
   stopping or fusion-parameter search.
2. The internal validation split is TOOL-LEVEL (whole sequences held out from
   the training pool), never a random run-level split, so no future sample of a
   training sequence can leak into validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets.base import Task


def make_splits(labelled: pd.DataFrame, task: Task, seed: int = 42
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tr_seqs = list(task.train_sequences)
    te_seqs = list(task.test_sequences)
    pool = labelled[labelled.sequence_id.isin(tr_seqs)].copy()
    test = labelled[labelled.sequence_id.isin(te_seqs)].copy()
    if pool.empty or test.empty:
        raise ValueError(f"{task.name}: empty train pool or test set")

    val_seqs = _choose_validation_sequences(pool, seed)
    if val_seqs:
        val = pool[pool.sequence_id.isin(val_seqs)].copy()
        train = pool[~pool.sequence_id.isin(val_seqs)].copy()
        mode = "tool_holdout"
    else:
        # only one training sequence available -> fall back to a CONTIGUOUS
        # tail block so no future sample of the training part leaks backwards.
        train, val, mode = _contiguous_tail_holdout(pool)
        val_seqs = sorted(val.sequence_id.unique().tolist())

    info = dict(task=task.name, group=task.group,
                train_domains=task.train_domains, test_domains=task.test_domains,
                train_sequences=sorted(train.sequence_id.unique().tolist()),
                validation_sequences=sorted(val_seqs),
                test_sequences=sorted(test.sequence_id.unique().tolist()),
                validation_mode=mode,
                n_train=int(len(train)), n_val=int(len(val)), n_test=int(len(test)))
    _assert_no_leak(train, val, test, task)
    return train, val, test, info


def _choose_validation_sequences(pool: pd.DataFrame, seed: int) -> list[str]:
    """Hold out one training sequence per training domain (deterministic in the
    seed).  With a single training sequence, return [] and let the caller use
    the contiguous-tail fallback."""
    seqs = sorted(pool.sequence_id.unique().tolist())
    if len(seqs) < 2:
        return []
    rng = np.random.default_rng(seed)
    chosen = []
    for dom, sub in pool.groupby("domain_id", sort=True):
        s = sorted(sub.sequence_id.unique().tolist())
        if len(s) < 2:                 # do not empty a domain
            continue
        chosen.append(s[int(rng.integers(len(s)))])
    if not chosen:                     # all domains have one sequence
        chosen = [seqs[int(rng.integers(len(seqs)))]]
    if len(chosen) >= len(seqs):       # never hold out everything
        chosen = chosen[:len(seqs) - 1]
    return sorted(chosen)


def _contiguous_tail_holdout(pool: pd.DataFrame, ratio: float = 0.20):
    parts_tr, parts_va = [], []
    for _, sub in pool.groupby("sequence_id", sort=True):
        sub = sub.sort_values("order_key")
        k = max(int(round(len(sub) * ratio)), 1)
        parts_tr.append(sub.iloc[:-k])
        parts_va.append(sub.iloc[-k:])
    return (pd.concat(parts_tr).reset_index(drop=True),
            pd.concat(parts_va).reset_index(drop=True),
            "contiguous_tail")


def _assert_no_leak(train, val, test, task: Task) -> None:
    for a, b, na, nb in ((train, test, "train", "test"), (val, test, "val", "test"),
                         (train, val, "train", "val")):
        ov = set(map(tuple, a[["sequence_id", "order_key"]].values)) & \
             set(map(tuple, b[["sequence_id", "order_key"]].values))
        if ov:
            raise AssertionError(f"{task.name}: {na}/{nb} share {len(ov)} rows")
    if set(train.sequence_id) & set(test.sequence_id):
        raise AssertionError(f"{task.name}: a sequence appears in both train and test")
    if set(val.sequence_id) & set(test.sequence_id):
        raise AssertionError(f"{task.name}: a sequence appears in both val and test")
