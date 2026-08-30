"""Condition-relative stage semantics and ordered fine-grained states.

q and nu are defined WITHIN one degradation sequence (`sequence_id`).
For the Mendeley dataset that unit is the Tool, not the Machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from . import config as C


def build_stage_labels(df: pd.DataFrame,
                       q_early: float = C.Q_EARLY,
                       q_late: float = C.Q_LATE,
                       rate_late_q: float = C.RATE_LATE_Q) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add VB_smooth, q_true, rate_norm, stage, stage_id per sequence.

    NOTE ON EVALUATION SEMANTICS
    ---------------------------
    For TEST sequences these columns are derived from the complete VB record
    and are used ONLY as offline evaluation reference. They are never fed to
    the model and never used to fit anything.
    """
    parts, rows = [], []
    for sid, sub in df.groupby("sequence_id", sort=False):
        sub = sub.sort_values("order_key").reset_index(drop=True).copy()
        vb = sub["VB"].astype(float)
        vbs = vb.rolling(C.VB_SMOOTH_WINDOW, min_periods=1, center=True).mean()
        q = (vbs - vbs.min()) / (vbs.max() - vbs.min() + C.EPS)
        rate = q.diff().fillna(0.0).rolling(C.RATE_SMOOTH_WINDOW, min_periods=1, center=True).mean()
        rn = (rate - rate.min()) / (rate.max() - rate.min() + C.EPS)
        tE, tL, tV = float(q.quantile(q_early)), float(q.quantile(q_late)), float(rn.quantile(rate_late_q))
        stage = np.where(q <= tE, "early", np.where((q >= tL) | (rn >= tV), "late", "middle"))
        sub["VB_smooth"], sub["q_true"], sub["rate_norm"] = vbs.values, q.values, rn.values
        sub["stage"] = stage
        sub["stage_id"] = pd.Series(stage).map(C.STAGE_TO_ID).astype(int).values
        parts.append(sub)
        cnt = pd.Series(stage).value_counts().reindex(C.STAGE_NAMES, fill_value=0)
        rows.append(dict(sequence_id=sid, domain_id=sub["domain_id"].iloc[0],
                         n=len(sub), theta_E=tE, theta_L=tL, theta_v=tV,
                         VB_min=float(vb.min()), VB_max=float(vb.max()),
                         VB_first=float(vb.iloc[0]),
                         early=int(cnt["early"]), middle=int(cnt["middle"]), late=int(cnt["late"])))
    out = pd.concat(parts).sort_values(["sequence_id", "order_key"]).reset_index(drop=True)
    return out, pd.DataFrame(rows)


class FineStateModel:
    """5-component GMM on u=[q, nu_norm], components ordered by mean q.

    Fitted on TRAINING sequences only.
    """

    def __init__(self, n_states: int = C.N_FINE_STATES, seed: int = 42):
        self.n_states, self.seed = n_states, seed
        self.gmm: GaussianMixture | None = None
        self.order: dict[int, int] = {}
        self.mapping_table: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> "FineStateModel":
        X = _u(train_df)
        self.gmm = GaussianMixture(n_components=self.n_states, covariance_type="full",
                                   random_state=self.seed, reg_covar=1e-5, n_init=10)
        self.gmm.fit(X)
        comp = self.gmm.predict(X)
        mean_q = (pd.DataFrame({"c": comp, "q": train_df["q_true"].values})
                  .groupby("c")["q"].mean().sort_values())
        self.order = {int(raw): i for i, raw in enumerate(mean_q.index.tolist())}
        self.mapping_table = pd.DataFrame(
            [dict(raw_component=int(r), fine_state=int(o), mean_q_train=float(mean_q.loc[r]),
                  macro_stage=C.ID_TO_STAGE[C.FINE_TO_STAGE[o]])
             for r, o in self.order.items()]).sort_values("fine_state").reset_index(drop=True)
        return self

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gmm is None:
            raise RuntimeError("FineStateModel not fitted")
        out = df.copy()
        raw = self.gmm.predict(_u(df))
        out["fine_state_true"] = np.array([self.order[int(r)] for r in raw], dtype=int)
        return out


def _u(df: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(df[["q_true", "rate_norm"]].values.astype(float),
                         nan=0.0, posinf=1.0, neginf=0.0)


def stage_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sequence stage availability -- required so macro metrics can declare
    which stages actually exist (T8 has a truncated early phase)."""
    rows = []
    for sid, sub in df.groupby("sequence_id", sort=False):
        c = sub["stage"].value_counts().reindex(C.STAGE_NAMES, fill_value=0)
        r = dict(sequence_id=sid, domain_id=sub["domain_id"].iloc[0], total=len(sub))
        for s in C.STAGE_NAMES:
            r[f"{s}_n"] = int(c[s])
            r[f"{s}_present"] = bool(c[s] > 0)
            g = sub[sub["stage"] == s]["VB"]
            r[f"{s}_VB_range"] = f"{g.min():.0f}-{g.max():.0f}" if len(g) else ""
        r["VB_first"] = float(sub.sort_values("order_key")["VB"].iloc[0])
        rows.append(r)
    out = pd.DataFrame(rows)
    # A sequence is flagged as run-in-truncated when its first recorded wear is
    # far above what the other sequences start from (Mendeley T8 starts at
    # VB=34um while the other eight tools start at VB=3um).
    ref = out["VB_first"].median()
    out["early_truncated"] = out["VB_first"] > (ref + 10.0)
    out["VB_first_median_all_sequences"] = ref
    return out
