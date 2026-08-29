"""Feature selection.  Fitted on TRAINING data only.

    Score_j = a1*MI(x_j, s) + a2*MI(x_j, q) + a3*|rho_spearman(x_j, q)| - a4*DI_j

DI_j is a distribution-instability term. In the paper it was the standardised
mean gap between the two PHM training conditions. Here it generalises to the
mean over all pairs of training GROUPS, where a group is a training domain when
more than one is available (cross-machine dual-source) and a training sequence
otherwise (single-source / LOTO). This keeps the term meaningful for every task
type instead of silently degenerating to zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from . import config as C


def distribution_instability(df: pd.DataFrame, cols: list[str], group_col: str) -> np.ndarray:
    groups = [g for _, g in df.groupby(group_col, sort=True)]
    if len(groups) < 2:
        return np.zeros(len(cols))
    acc = np.zeros(len(cols))
    n = 0
    for a, b in combinations(groups, 2):
        mu_a, sd_a = a[cols].mean().values, a[cols].std(ddof=0).values
        mu_b, sd_b = b[cols].mean().values, b[cols].std(ddof=0).values
        acc += np.abs(mu_a - mu_b) / (0.5 * (sd_a + sd_b) + C.EPS)
        n += 1
    return acc / max(n, 1)


def select_features(train_df: pd.DataFrame, candidate_cols: list[str],
                    seed: int = 42,
                    n_selected: int = C.N_SELECTED_FEATURES,
                    max_pool: int = C.MAX_FEATURE_POOL,
                    redundancy_threshold: float = C.REDUNDANCY_THRESHOLD
                    ) -> tuple[list[str], pd.DataFrame]:
    X = train_df[candidate_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    var = X.var(ddof=0)
    cols = [c for c in candidate_cols if var[c] > C.VAR_THRESHOLD]
    if not cols:
        raise ValueError("all candidate features are constant on the training set")
    Xv = X[cols].values
    s = train_df["stage_id"].values.astype(int)
    q = train_df["q_true"].values.astype(float)

    mi_s = mutual_info_classif(Xv, s, random_state=seed)
    mi_q = mutual_info_regression(Xv, q, random_state=seed)
    rho = np.array([abs(spearmanr(Xv[:, i], q).statistic) if np.std(Xv[:, i]) > 0 else 0.0
                    for i in range(Xv.shape[1])])
    rho = np.nan_to_num(rho)
    gcol = "domain_id" if train_df["domain_id"].nunique() > 1 else "sequence_id"
    di = distribution_instability(train_df.assign(**{c: X[c] for c in cols}), cols, gcol)

    w = C.SCORE_WEIGHTS
    score = (w["mi_stage"] * _n(mi_s) + w["mi_q"] * _n(mi_q)
             + w["spearman_q"] * rho - w["instability"] * _n(di))

    rank = (pd.DataFrame(dict(feature=cols, MI_stage=mi_s, MI_q=mi_q,
                              Spearman_q=rho, distribution_instability=di,
                              final_score=score, instability_group=gcol))
            .sort_values("final_score", ascending=False).reset_index(drop=True))
    pool = rank.head(max_pool)

    # greedy redundancy filter
    corr = pd.DataFrame(Xv, columns=cols)[pool["feature"].tolist()].corr().abs().fillna(0.0)
    selected: list[str] = []
    max_corr_of = {}
    for f in pool["feature"]:
        mc = max((corr.loc[f, g] for g in selected), default=0.0)
        if mc < redundancy_threshold:
            selected.append(f)
            max_corr_of[f] = float(mc)
        if len(selected) >= n_selected:
            break
    rank["selected"] = rank["feature"].isin(selected)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank["max_abs_corr_with_selected"] = rank["feature"].map(max_corr_of)
    if not selected:
        raise ValueError("redundancy filter removed every feature")
    return selected, rank


def _n(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.asarray(v, dtype=float))
    rng = v.max() - v.min()
    return (v - v.min()) / (rng + C.EPS) if rng > 0 else np.zeros_like(v)
