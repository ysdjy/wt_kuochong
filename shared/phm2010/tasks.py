"""Authoritative PHM2010 task (train/test cutter) registry.

Single source of truth for the three formal PHM2010 tasks. No method may define
its own D1/D2/D3 split independently — every adapter must import TASKS (or the
CUTTERS / TASK_NAMES helpers below) from here.

Task definitions (full lifecycle sequences, no random 80/20 split; test cutter's
data must never leak into feature selection, scaler fitting, GMM fitting,
hyperparameter tuning, early stopping, or model selection for that task — any
validation set must be constructed only from the train cutters):

    D1: train = C1 + C4, test = C6
    D2: train = C1 + C6, test = C4
    D3: train = C4 + C6, test = C1
"""
from __future__ import annotations

CUTTERS = ["C1", "C4", "C6"]

TASKS: dict[str, dict[str, list[str] | str]] = {
    "D1": {"train": ["C1", "C4"], "test": "C6"},
    "D2": {"train": ["C1", "C6"], "test": "C4"},
    "D3": {"train": ["C4", "C6"], "test": "C1"},
}

TASK_NAMES = list(TASKS.keys())


def train_cutters(task: str) -> list[str]:
    return list(TASKS[task]["train"])


def test_cutter(task: str) -> str:
    return TASKS[task]["test"]


def resolve_tasks(tasks_arg: str) -> list[str]:
    """Parse a `--tasks` CLI value into a validated list of task names.

    Accepts "all" or a comma-separated subset, e.g. "D1,D3". Raises ValueError on
    an unknown task name so a typo fails fast instead of silently running nothing.
    """
    if tasks_arg.strip().lower() == "all":
        return list(TASK_NAMES)
    requested = [t.strip().upper() for t in tasks_arg.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task name(s): {unknown}. Valid tasks: {TASK_NAMES} or 'all'.")
    return requested


def assert_no_test_leakage(train_cutter_list: list[str], test_cutter_name: str) -> None:
    """Sanity check: the test cutter must never appear among the train cutters."""
    if test_cutter_name in train_cutter_list:
        raise AssertionError(
            f"Test-leakage guard triggered: test cutter {test_cutter_name!r} also "
            f"appears in train cutters {train_cutter_list!r}"
        )
    if set(train_cutter_list + [test_cutter_name]) != set(CUTTERS):
        raise AssertionError(
            f"Train+test cutters {train_cutter_list + [test_cutter_name]} do not "
            f"exactly cover the full cutter set {CUTTERS}"
        )
