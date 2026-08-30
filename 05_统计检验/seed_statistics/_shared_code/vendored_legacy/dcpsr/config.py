"""Global configuration. Fixed BEFORE any experiment is run.

Nothing in this file may be changed on the basis of a test-set result.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Seeds. Fixed in advance, never re-drawn after seeing results.
# ----------------------------------------------------------------------------
FINAL_SEEDS = [42, 52, 62, 72, 82]
DEBUG_SEEDS = [42]

# ----------------------------------------------------------------------------
# Stage semantics (paper protocol, applied PER DEGRADATION SEQUENCE)
# ----------------------------------------------------------------------------
Q_EARLY = 0.30       # Q_E
Q_LATE = 0.72        # Q_L
RATE_LATE_Q = 0.78   # Q_nu
VB_SMOOTH_WINDOW = 7
RATE_SMOOTH_WINDOW = 5

STAGE_NAMES = ["early", "middle", "late"]
STAGE_TO_ID = {"early": 0, "middle": 1, "late": 2}
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}

# ----------------------------------------------------------------------------
# Fine-grained degradation states
# ----------------------------------------------------------------------------
N_FINE_STATES = 5
FINE_TO_STAGE = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}   # S0->E, S1..S3->M, S4->L

# ----------------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------------
VAR_THRESHOLD = 1e-10
MAX_FEATURE_POOL = 260
N_SELECTED_FEATURES = 45
REDUNDANCY_THRESHOLD = 0.92
SCORE_WEIGHTS = dict(mi_stage=1.0, mi_q=1.0, spearman_q=1.0, instability=1.0)  # a1..a4

# ----------------------------------------------------------------------------
# Network (paper Table 6 -- NOT re-searched on the new dataset)
# ----------------------------------------------------------------------------
ARCH = dict(L=12, dropout=0.20, lr=5e-4, tcn_channels=(32, 64, 64), gru_hidden=64)
BATCH_SIZE = 32
EPOCHS = 120
PATIENCE = 18
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

LAMBDA_STAGE = 1.00
LAMBDA_FINE = 0.25
LAMBDA_Q = 0.30
LAMBDA_MONO = 0.03

# ----------------------------------------------------------------------------
# DC-PSR fusion. Defaults = paper Table 6. The search grid is FIXED here in
# advance and is only ever evaluated on the internal validation split.
# ----------------------------------------------------------------------------
FUSION_DEFAULT = dict(eta=0.75, fine_weight=0.30, temperature=1.20,
                      mid_floor=0.12, late_tau=0.66, early_tau=0.38,
                      order_blend=0.25)

FUSION_GRID = dict(
    eta=[0.55, 0.65, 0.75],
    fine_weight=[0.10, 0.20, 0.30],
    temperature=[1.0, 1.2],
    mid_floor=[0.04, 0.08, 0.12],
    late_tau=[0.60, 0.66, 0.72],
    early_tau=[0.34, 0.38, 0.42],
    order_blend=[0.25, 0.50],
)

PRIOR_CENTERS = dict(early=0.18, middle=0.50, late=0.84)
PRIOR_SIGMA = 0.17
EARLY_SUPPRESS_K = 18.0
LATE_SUPPRESS_K = 18.0

TRANSITION_MATRIX = [[0.9400, 0.0550, 0.0050],
                     [0.0050, 0.9550, 0.0400],
                     [0.0005, 0.0100, 0.9895]]
ALPHA_INIT = [0.90, 0.10, 1e-6]

EPS = 1e-12
