"""DC-PSR probabilistic inference.

Produces every intermediate probability, so the ablation A1..A6 is derived from
ONE trained backbone rather than six separate trainings:

    A1 raw        p_raw
    A2 raw+fine   eta*p_raw + (1-eta)*p_fine
    A3 raw+prior  eta*p_raw + (1-eta)*p_prior
    A4 mix        eta*p_raw + (1-eta)*(w_f*p_fine + (1-w_f)*p_prior)
    A5 ordered    alpha   (causal ordered filter applied to p_mix)
    A6 final      (1-beta)*p_mix + beta*alpha
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from . import config as C

VARIANTS = ["raw", "fine", "prior", "raw_fine", "raw_prior", "mix", "ordered", "final"]
ABLATION_OF = {"raw": "A1", "raw_fine": "A2", "raw_prior": "A3",
               "mix": "A4", "ordered": "A5", "final": "A6"}


def _norm(P: np.ndarray) -> np.ndarray:
    return P / (P.sum(axis=1, keepdims=True) + C.EPS)


def qhat_prior(q_hat, early_tau, late_tau, mid_floor,
               centers=None, sigma=C.PRIOR_SIGMA) -> np.ndarray:
    """Gaussian position prior with dual-side inhibition and a middle floor."""
    c = centers or C.PRIOR_CENTERS
    q = np.clip(np.asarray(q_hat, dtype=float).ravel(), 0.0, 1.0)
    pe = np.exp(-0.5 * ((q - c["early"]) / sigma) ** 2) * expit(C.EARLY_SUPPRESS_K * (early_tau - q))
    pm = np.maximum(np.exp(-0.5 * ((q - c["middle"]) / sigma) ** 2), mid_floor)
    pl = np.exp(-0.5 * ((q - c["late"]) / sigma) ** 2) * expit(C.LATE_SUPPRESS_K * (q - late_tau))
    return _norm(np.vstack([pe, pm, pl]).T)


def fine_to_stage_prob(Fp: np.ndarray) -> np.ndarray:
    Fp = _norm(np.asarray(Fp, dtype=float))
    P = np.zeros((len(Fp), 3))
    for k in range(Fp.shape[1]):
        P[:, C.FINE_TO_STAGE[k]] += Fp[:, k]
    return _norm(P)


def temperature_scale(P: np.ndarray, temp: float) -> np.ndarray:
    lp = np.log(np.clip(P, 1e-12, 1.0)) / max(temp, 1e-6)
    lp -= lp.max(axis=1, keepdims=True)
    return _norm(np.exp(lp))


def causal_ordered_filter(P: np.ndarray, A=None, alpha0=None) -> np.ndarray:
    """Forward-only (causal) recursion. Never looks at future observations."""
    A = np.asarray(A if A is not None else C.TRANSITION_MATRIX, dtype=float)
    a = np.asarray(alpha0 if alpha0 is not None else C.ALPHA_INIT, dtype=float)
    a = a / a.sum()
    out = np.zeros_like(P)
    for i in range(len(P)):
        a = (a @ A if i > 0 else a) * P[i]
        a = a / (a.sum() + C.EPS)
        out[i] = a
    return out


def apply_inference(pred: pd.DataFrame, params: dict) -> pd.DataFrame:
    """pred must carry raw_prob_*, fine_prob_0..K-1, q_hat, sequence_id, order_key."""
    df = pred.sort_values(["sequence_id", "order_key"]).reset_index(drop=True).copy()
    eta, wf = float(params["eta"]), float(params["fine_weight"])
    beta = float(params["order_blend"])
    if beta <= 0:
        raise ValueError("order_blend must be > 0: A6/B12 requires the ordered term")

    raw = temperature_scale(df[[f"raw_prob_{s}" for s in C.STAGE_NAMES]].values,
                            params["temperature"])
    fine = fine_to_stage_prob(df[[f"fine_prob_{i}" for i in range(C.N_FINE_STATES)]].values)
    prior = qhat_prior(df["q_hat"].values, params["early_tau"], params["late_tau"],
                       params["mid_floor"])

    raw_fine = _norm(eta * raw + (1 - eta) * fine)
    raw_prior = _norm(eta * raw + (1 - eta) * prior)
    aux = wf * fine + (1 - wf) * prior
    mix = _norm(eta * raw + (1 - eta) * aux)

    ordered = np.zeros_like(mix)
    for _, idx in df.groupby("sequence_id", sort=False).groups.items():
        idx = list(idx)                       # already in order_key order
        ordered[idx] = causal_ordered_filter(mix[idx])
    final = _norm((1 - beta) * mix + beta * ordered)

    for name, arr in [("raw", raw), ("fine", fine), ("prior", prior),
                      ("raw_fine", raw_fine), ("raw_prior", raw_prior),
                      ("mix", mix), ("ordered", ordered), ("final", final)]:
        for i, s in enumerate(C.STAGE_NAMES):
            df[f"{name}_prob_{s}"] = arr[:, i]
        df[f"stage_pred_{name}"] = np.argmax(arr, axis=1)
        df[f"stage_pred_{name}_name"] = df[f"stage_pred_{name}"].map(C.ID_TO_STAGE)
    return df


def fusion_grid(grid: dict | None = None):
    from itertools import product
    g = grid or C.FUSION_GRID
    keys = list(g)
    for vals in product(*[g[k] for k in keys]):
        yield dict(zip(keys, vals))
