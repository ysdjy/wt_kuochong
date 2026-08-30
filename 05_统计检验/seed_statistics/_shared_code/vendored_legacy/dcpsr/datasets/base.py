"""Dataset adapter contract.

A dataset adapter is the ONLY dataset-specific component. It must produce a
run-level table with these META columns:

    sequence_id : str   independent degradation sequence.  q is min-max
                        normalised WITHIN this unit.  (Mendeley: Tool T1..T9;
                        PHM2010: condition C1/C4/C6; NASA: case 1..16)
    domain_id   : str   transfer domain / operating scenario. (Mendeley:
                        Machine M1..M3; PHM2010: same as sequence_id)
    order_key   : int   monotone temporal index within a sequence (run/cut id)
    VB          : float wear label.  EVALUATION TRUTH ONLY -- never an input.

plus any number of numeric signal-feature columns, and optionally:

    ctime       : float cumulated tool contact time
    source_file : str

Everything downstream (stages, online features, selection, model, inference,
metrics, aggregation) is dataset-agnostic and consumes only this contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence
import abc
import pandas as pd

META_COLS = ["sequence_id", "domain_id", "order_key", "VB"]
OPTIONAL_META_COLS = ["ctime", "source_file", "split_tag"]
# columns produced downstream that must never be treated as input features
DERIVED_COLS = ["VB_smooth", "q_true", "rate_norm", "stage", "stage_id",
                "fine_state_true"]


@dataclass
class Task:
    """One transfer experiment."""
    name: str                      # "MD1", "MS3", "LOTO_T7", ...
    group: str                     # "cross_machine_dual" | "cross_machine_single" | "leave_one_tool_out"
    train_sequences: list[str]
    test_sequences: list[str]
    train_domains: list[str] = field(default_factory=list)
    test_domains: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        ov = set(self.train_sequences) & set(self.test_sequences)
        if ov:
            raise ValueError(f"{self.name}: train/test sequence overlap {sorted(ov)}")
        if not self.train_sequences or not self.test_sequences:
            raise ValueError(f"{self.name}: empty train or test sequence list")


class DatasetAdapter(abc.ABC):
    """Base class. Subclass per dataset."""

    name: str = "abstract"

    @abc.abstractmethod
    def load_run_level_table(self) -> pd.DataFrame:
        """Return the run-level feature table (see module docstring)."""

    @abc.abstractmethod
    def task_definitions(self) -> list[Task]:
        """Return every task this dataset defines."""

    # -- shared helpers -----------------------------------------------------
    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        skip = set(META_COLS) | set(OPTIONAL_META_COLS) | set(DERIVED_COLS)
        out = []
        for c in df.columns:
            if c in skip:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
        return out

    def validate_table(self, df: pd.DataFrame) -> None:
        missing = [c for c in META_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"adapter {self.name}: missing meta columns {missing}")
        if df["order_key"].isna().any():
            raise ValueError("order_key contains NaN")
        for sid, sub in df.groupby("sequence_id"):
            if sub["order_key"].duplicated().any():
                raise ValueError(f"sequence {sid}: duplicated order_key")
            if sub["domain_id"].nunique() != 1:
                raise ValueError(f"sequence {sid} spans multiple domains -- "
                                 "a degradation sequence must belong to one domain")
        if not self.feature_columns(df):
            raise ValueError("no numeric feature columns found")
