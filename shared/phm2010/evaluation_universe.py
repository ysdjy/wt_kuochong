"""Common evaluation universe for PHM2010 formal comparison tables.

Different method families cover different physical run ranges on the same test
cutter:

- Window-based methods (RF / TCN-GRU / Multi-task TCN-GRU / DC-PHSR / HTT-Net,
  all built on L=12-sample sliding windows) can only emit a prediction once 12
  consecutive samples of the run's history exist, so their first predicted
  run_id is 12 (1-indexed run numbering).
- Raw-signal methods (Multi-source Attention / MTF-AViTK / Dynamic GIN+TGP /
  DP2Net-adapted) predict every physical run from 1.

Empirically verified (2026-08-30, against final_statistical_evidence's already
frozen, paper-authoritative per-method D1/D2/D3 predictions — see
final_statistical_evidence/transfer_tasks/{D2,D3}/*/predictions.csv and
final_statistical_evidence/predictions_common_universe/D1_*_304runs.csv):
window-based methods cover run_id 12-315 (304 runs) and raw-signal methods cover
run_id 1-315 (315 runs) on EVERY test cutter (C1, C4, C6) — i.e. the offset is a
property of the windowing scheme, not of which physical cutter is under test.
This holds identically for D1 (test=C6), D2 (test=C4), D3 (test=C1). Per project
policy (`EXPERIMENT_REGISTRY.md` / task spec section 21), this must be verified
empirically rather than assumed for D2/D3 — the above is that verification.

The common evaluation universe used for every FORMAL comparison-table number,
regardless of task, is therefore fixed: run_id 12 to 315 inclusive, n=304.
"""
from __future__ import annotations

COMMON_UNIVERSE_RUN_ID_START = 12
COMMON_UNIVERSE_RUN_ID_END = 315  # inclusive
COMMON_UNIVERSE_N = 304

# Methods whose native prediction coverage already equals the common universe
# (no reduction needed before computing formal metrics).
WINDOW_BASED_METHODS = {
    "B1", "B2", "B3", "B9",  # RF, TCN-GRU, Multi-task TCN-GRU, DC-PHSR
    "B4",                     # HTT-Net (also window/run-level-feature based)
}

# Methods whose native prediction coverage is run_id 1-315 (315 runs) and must be
# reduced to the common universe before computing formal metrics.
RAW_SIGNAL_METHODS = {
    "B5",  # Multi-source Channel-Spatial Attention
    "B6",  # MTF-AViTK
    "B7",  # Dynamic GIN + TGP
    "B8",  # DP2Net-adapted
}


def is_in_common_universe(run_id: int) -> bool:
    return COMMON_UNIVERSE_RUN_ID_START <= run_id <= COMMON_UNIVERSE_RUN_ID_END


def restrict_to_common_universe(predictions_df, run_id_col: str = "run_id"):
    """Return only the rows of `predictions_df` whose run_id falls inside the
    common evaluation universe (run_id 12-315). Safe to call on a method whose
    native coverage is already exactly the common universe (no-op in that case).

    `predictions_df` is a pandas DataFrame; imported lazily so this module has
    no hard pandas dependency for callers that only need the constants above.
    """
    mask = (predictions_df[run_id_col] >= COMMON_UNIVERSE_RUN_ID_START) & (
        predictions_df[run_id_col] <= COMMON_UNIVERSE_RUN_ID_END
    )
    return predictions_df.loc[mask].reset_index(drop=True)


def assert_common_universe(predictions_df, run_id_col: str = "run_id", context: str = "") -> None:
    """Raise AssertionError if `predictions_df` (after restriction) does not have
    exactly COMMON_UNIVERSE_N rows with run_id covering exactly
    [COMMON_UNIVERSE_RUN_ID_START, COMMON_UNIVERSE_RUN_ID_END]. Call this on every
    method's per-run-id predictions right before computing formal metrics, so a
    silently-truncated or duplicated prediction set fails loudly instead of
    producing a quietly-wrong comparison-table number.
    """
    restricted = restrict_to_common_universe(predictions_df, run_id_col)
    n = len(restricted)
    if n != COMMON_UNIVERSE_N:
        raise AssertionError(
            f"{context}: expected {COMMON_UNIVERSE_N} rows in common evaluation "
            f"universe (run_id {COMMON_UNIVERSE_RUN_ID_START}-{COMMON_UNIVERSE_RUN_ID_END}), got {n}"
        )
    run_ids = sorted(restricted[run_id_col].tolist())
    expected = list(range(COMMON_UNIVERSE_RUN_ID_START, COMMON_UNIVERSE_RUN_ID_END + 1))
    if run_ids != expected:
        raise AssertionError(
            f"{context}: run_id set does not exactly match the common evaluation "
            f"universe range {COMMON_UNIVERSE_RUN_ID_START}-{COMMON_UNIVERSE_RUN_ID_END} "
            f"(missing or duplicated run_ids present)"
        )
