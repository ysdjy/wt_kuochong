"""Non-deep baselines B1..B6.

B1 IS AN ORACLE BASELINE.
-------------------------
It classifies using the ground-truth wear of the TEST sequence. It is kept
because the paper reports it, but every row it produces is tagged
`oracle_wear_reference=True` so it can never be presented as a sensor-based
method. B2..B12 use sensor features only.

B2 FIX.
-------
The original code compared a min-max rescaled test degradation rate against a
threshold taken from the UNSCALED training rate, so the rate condition fired
almost everywhere and B2 could never predict `middle`. Here the training rate
is rescaled with the same transform before the quantile is taken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from . import config as C

BASELINE_INFO = {
    "B1": ("Fixed-stage Rule (ORACLE)", "Fixed VB threshold", "rule on ground-truth wear"),
    "B2": ("Relative-stage q-proxy Rule", "Condition-relative", "RF q-regressor + relative rule"),
    "B3": ("Fixed-stage RF", "Fixed VB threshold", "RandomForest"),
    "B4": ("Relative-stage SVM", "Condition-relative", "SVM (RBF)"),
    "B5": ("Relative-stage RF", "Condition-relative", "RandomForest"),
    "B6": ("Relative-stage GBDT", "Condition-relative", "HistGradientBoosting / XGBoost"),
    "B7": ("Relative-stage MLP", "Condition-relative", "MLP"),
    "B8": ("Relative-stage TCN", "Condition-relative", "TCN"),
    "B9": ("Relative-stage GRU", "Condition-relative", "GRU"),
    "B10": ("Relative-stage TCN-GRU", "Condition-relative", "TCN-GRU"),
    "B11": ("Multi-task TCN-GRU", "Condition-relative", "TCN-GRU + fine + q heads"),
    "B12": ("DC-PSR", "Condition-relative", "B11 + degradation-consistent inference"),
}
ORACLE_METHODS = {"B1", "B3"}     # both threshold on ground-truth wear
DETERMINISTIC_METHODS = {"B1"}    # produces identical output for every seed


def _onehot(y, k=3):
    P = np.zeros((len(y), k)); P[np.arange(len(y)), np.asarray(y, int)] = 1.0
    return P


def _causal_rate(q: np.ndarray, w: int = C.RATE_SMOOTH_WINDOW) -> np.ndarray:
    d = np.diff(np.asarray(q, float), prepend=q[0])
    return pd.Series(d).rolling(w, min_periods=1).mean().values     # causal, no centering


def run_b1_b6(train, test, features, seed=42):
    """Returns {method: (y_pred, prob, info)} on the TEST windows."""
    Xtr = train[features].values
    Xte = test[features].values
    ytr = train["stage_id"].values.astype(int)
    out = {}

    # ---- B1 ORACLE: threshold on ground-truth wear ------------------------
    th_e = float(train["VB_smooth"].quantile(C.Q_EARLY))
    th_l = float(train["VB_smooth"].quantile(C.Q_LATE))
    vb_te = test["VB_smooth"].values.astype(float)
    y = np.where(vb_te <= th_e, 0, np.where(vb_te >= th_l, 2, 1)).astype(int)
    out["B1"] = (y, _onehot(y), dict(oracle_wear_reference=True,
                                     note="uses ground-truth VB of the test sequence"))

    # ---- B2 sensor-only q proxy + relative rule (units bug fixed) ---------
    reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, min_samples_leaf=2)
    reg.fit(Xtr, train["q_true"].values)
    q_tr = np.clip(reg.predict(Xtr), 0, 1)
    q_te = np.clip(reg.predict(Xte), 0, 1)
    r_tr, r_te = _causal_rate(q_tr), _causal_rate(q_te)
    lo, hi = float(r_tr.min()), float(r_tr.max())            # ONE transform...
    r_tr_s = (r_tr - lo) / (hi - lo + C.EPS)                 # ...applied to BOTH
    r_te_s = np.clip((r_te - lo) / (hi - lo + C.EPS), 0, 1)
    tE, tL = float(np.quantile(q_tr, C.Q_EARLY)), float(np.quantile(q_tr, C.Q_LATE))
    tV = float(np.quantile(r_tr_s, C.RATE_LATE_Q))           # same scale as r_te_s
    y = np.where(q_te <= tE, 0, np.where((q_te >= tL) | (r_te_s >= tV), 2, 1)).astype(int)
    out["B2"] = (y, _onehot(y), dict(oracle_wear_reference=False,
                                     note="RF q-proxy from sensor features + relative rule; "
                                          "rate threshold and rate are on the same scale"))

    # ---- B3 ORACLE labels from fixed wear thresholds ----------------------
    ytr_fixed = np.where(train["VB_smooth"].values <= th_e, 0,
                         np.where(train["VB_smooth"].values >= th_l, 2, 1)).astype(int)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                class_weight="balanced_subsample", n_jobs=-1).fit(Xtr, ytr_fixed)
    out["B3"] = (rf.predict(Xte), rf.predict_proba(Xte),
                 dict(oracle_wear_reference=False,
                      note="fixed-threshold LABELS from training wear; sensor features only"))

    svm = SVC(C=5.0, gamma="scale", kernel="rbf", class_weight="balanced",
              probability=True, random_state=seed).fit(Xtr, ytr)
    out["B4"] = (svm.predict(Xte), svm.predict_proba(Xte), dict(oracle_wear_reference=False))

    rf5 = RandomForestClassifier(n_estimators=400, random_state=seed,
                                 class_weight="balanced_subsample", n_jobs=-1).fit(Xtr, ytr)
    out["B5"] = (rf5.predict(Xte), rf5.predict_proba(Xte), dict(oracle_wear_reference=False))

    gb, note = _fit_gbdt(Xtr, ytr, seed)
    out["B6"] = (gb.predict(Xte), gb.predict_proba(Xte),
                 dict(oracle_wear_reference=False, note=note))
    return out


def _fit_gbdt(X, y, seed):
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.9, random_state=seed,
                          tree_method="hist", num_class=3, objective="multi:softprob")
        m.fit(X, y)
        return m, "xgboost"
    except Exception:                                             # noqa: BLE001
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=400, random_state=seed).fit(X, y)
        return m, "sklearn HistGradientBoosting (xgboost unavailable)"
