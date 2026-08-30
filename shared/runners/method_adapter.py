"""Unified adapter interface every method (B1-B9) implements.

`run_phm2010.py` never knows any method's internals — it only constructs a
`MethodAdapter` subclass instance and calls `.run(...)`. The base class's `run()`
provides ALL the boilerplate (resume/DONE.flag checking, seeding, timing, writing
the unified result schema per RESULTS_POLICY.md, atomic DONE.flag) — subclasses
only implement the four domain hooks: prepare / train / predict / evaluate.

A subclass MAY override `run()` itself for a method whose lifecycle genuinely does
not fit the template (e.g. B9 reusing B3's checkpoint under strict hash-matching
conditions per EXPERIMENT_REGISTRY.md section 34) — but should prefer the hooks
whenever possible so the result-writing/resume logic stays in one place.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "shared"))
from metrics.metrics import confusion_matrix, stage_id  # noqa: E402
from phm2010.evaluation_universe import assert_common_universe  # noqa: E402
from phm2010.tasks import assert_no_test_leakage  # noqa: E402
from utils.run_meta import build_run_meta, dict_hash  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402


class MethodAdapter(ABC):
    """Base class for one method (e.g. B9_DC_PHSR) on one (task, seed) cell."""

    #: Set by subclasses: "B1".."B9"
    method_id: str = ""
    #: Set by subclasses: e.g. "RF", "DC-PHSR"
    method_name: str = ""

    def __init__(
        self,
        *,
        task: str,
        train_cutters: list[str],
        test_cutter: str,
        seed: int,
        preprocess_seed: int,
        output_dir: Path,
        device: str = "auto",
        config: dict | None = None,
    ):
        assert_no_test_leakage(train_cutters, test_cutter)
        self.task = task
        self.train_cutters = train_cutters
        self.test_cutter = test_cutter
        self.seed = seed
        self.preprocess_seed = preprocess_seed
        self.output_dir = Path(output_dir)
        self.device = device
        self.config = config or {}
        self._training_log_rows: list[dict] = []
        # Default; overwritten by run(resume=..., smoke_test=...) — safe to read
        # from prepare()/build_model()/train()/predict() even if an adapter is
        # ever driven by something other than run() (e.g. a bespoke test script).
        self.smoke_test = False

    # ---- domain hooks every subclass implements -----------------------------

    @abstractmethod
    def prepare(self) -> None:
        """Load/prepare data for self.task (train_cutters -> test_cutter).
        Must use self.preprocess_seed for anything preprocessing-related
        (feature selection, scaler, GMM, train/val split) and must construct any
        validation set only from self.train_cutters — never touch test-cutter
        data here."""

    @abstractmethod
    def build_model(self) -> Any:
        """Instantiate a fresh model. Caller (`run()`) calls
        `seed_everything(self.seed)` immediately before this — do not seed again
        inside build_model unless you need a second, explicitly-documented RNG
        stream."""

    @abstractmethod
    def train(self, model: Any) -> None:
        """Train `model` in place. Append one dict per epoch/iteration to
        `self._training_log_rows` (via `self.log_epoch(**kwargs)`) as you go."""

    @abstractmethod
    def predict(self, model: Any) -> pd.DataFrame:
        """Return predictions on the test cutter as a DataFrame with at least
        columns: run_id, true_stage, pred_stage, p_early, p_middle, p_late.
        Optionally q_true, q_pred if this method produces a continuous
        degradation index. Rows need not be pre-restricted to the common
        evaluation universe — `run()` restricts before scoring."""

    # ---- optional hooks ------------------------------------------------------

    def feature_hash(self) -> str | None:
        return None

    def split_hash(self) -> str | None:
        return None

    def label_hash(self) -> str | None:
        return None

    def extra_run_meta(self) -> dict:
        return {}

    # ---- helpers for subclasses ----------------------------------------------

    def log_epoch(self, **kwargs) -> None:
        self._training_log_rows.append(kwargs)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ---- templated orchestration ----------------------------------------------

    def is_done(self) -> bool:
        return (self.output_dir / "DONE.flag").exists()

    def is_failed(self) -> bool:
        return (self.output_dir / "FAILED.flag").exists()

    def run(self, resume: bool = True, smoke_test: bool = False) -> dict:
        """Full lifecycle: resume-check -> seed -> prepare -> build -> train ->
        predict -> evaluate -> write unified result schema -> DONE.flag.

        Returns a small status dict: {"status": "done"|"skipped"|"failed", ...}.
        Never raises on a training-time failure — writes FAILED.flag + error.log
        and returns status="failed" instead, so a driver loop over many
        (method, task, seed) cells can keep going.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if resume and self.is_done():
            return {"status": "skipped", "reason": "DONE.flag present"}

        # Exposed to prepare()/build_model()/train()/predict() so a method can
        # shrink its dataset / use a tiny dummy batch during a smoke test
        # instead of always operating on the full real data.
        self.smoke_test = smoke_test

        start_dt = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        try:
            seed_everything(self.seed)
            self.prepare()
            seed_everything(self.seed)  # re-assert immediately before model build
            model = self.build_model()
            if not smoke_test:
                self.train(model)
            predictions = self.predict(model)
            self._write_outputs(predictions, start_dt, start_perf, smoke_test=smoke_test)
            return {"status": "done"}
        except Exception as exc:  # noqa: BLE001 - deliberately broad: isolate one cell's failure
            self._write_failure(exc)
            return {"status": "failed", "error": str(exc)}

    def _write_failure(self, exc: Exception) -> None:
        (self.output_dir / "error.log").write_text(
            "".join(traceback.format_exception(exc)), encoding="utf-8"
        )
        (self.output_dir / "FAILED.flag").write_text("", encoding="utf-8")

    def _write_outputs(
        self,
        predictions: pd.DataFrame,
        start_dt: datetime,
        start_perf: float,
        smoke_test: bool,
    ) -> None:
        out = self.output_dir
        end_dt = datetime.now(timezone.utc)
        runtime_sec = time.perf_counter() - start_perf

        predictions.to_csv(out / "predictions.csv", index=False)

        pd.DataFrame(self._training_log_rows).to_csv(out / "training_log.csv", index=False)

        if smoke_test:
            # Smoke tests skip formal metrics (predictions may be a single tiny
            # batch, not a full run) — record that explicitly rather than writing
            # a misleading metrics.json.
            metrics = {"smoke_test": True}
        else:
            eval_df = predictions
            if "run_id" in eval_df.columns:
                assert_common_universe(
                    eval_df, context=f"{self.method_id}/{self.task}/seed{self.seed}"
                )
            metrics = self.evaluate(eval_df)

        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)

        if not smoke_test and {"true_stage", "pred_stage"}.issubset(predictions.columns):
            truth_ids = [stage_id(v) for v in predictions["true_stage"]]
            pred_ids = [stage_id(v) for v in predictions["pred_stage"]]
            cm = confusion_matrix(truth_ids, pred_ids)
            pd.DataFrame(cm, index=["true_early", "true_middle", "true_late"],
                         columns=["pred_early", "pred_middle", "pred_late"]).to_csv(
                out / "confusion_matrix.csv"
            )

        config_resolved = dict(self.config)
        config_resolved.update({
            "method": self.method_id,
            "task": self.task,
            "train_cutters": self.train_cutters,
            "test_cutter": self.test_cutter,
            "seed": self.seed,
            "preprocess_seed": self.preprocess_seed,
        })
        (out / "config_resolved.yaml").write_text(
            yaml.safe_dump(config_resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        run_meta = build_run_meta(
            method=self.method_id,
            dataset="PHM2010",
            task=self.task,
            train_cutters=self.train_cutters,
            test_cutter=self.test_cutter,
            seed=self.seed,
            preprocess_seed=self.preprocess_seed,
            repo_root=REPO_ROOT,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            runtime_sec=runtime_sec,
            feature_hash=self.feature_hash(),
            split_hash=self.split_hash(),
            label_hash=self.label_hash(),
            evaluation_universe_hash=None,
            config_hash=dict_hash(config_resolved),
            extra=self.extra_run_meta(),
        )
        (out / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        if not smoke_test:
            # Clear any stale failure markers from an earlier failed attempt at
            # this same cell -- otherwise a successful retry leaves a
            # confusing FAILED.flag/error.log sitting next to its own DONE.flag.
            (out / "FAILED.flag").unlink(missing_ok=True)
            (out / "error.log").unlink(missing_ok=True)
            # Atomic DONE.flag: write-then-rename so a crash mid-write never
            # leaves a DONE.flag behind for an incomplete run.
            tmp = out / "DONE.flag.tmp"
            tmp.write_text("", encoding="utf-8")
            os.replace(tmp, out / "DONE.flag")

    # abstract, but only required for non-smoke-test runs
    def evaluate(self, predictions: pd.DataFrame) -> dict:
        from metrics.metrics import compute_all_metrics

        ordered = predictions.sort_values("run_id").reset_index(drop=True)
        probs = ordered[["p_early", "p_middle", "p_late"]].to_numpy(dtype=float)
        q_true = ordered["q_true"] if "q_true" in ordered.columns else None
        q_pred = ordered["q_pred"] if "q_pred" in ordered.columns else None
        return compute_all_metrics(
            ordered["true_stage"], ordered["pred_stage"], probs, q_true=q_true, q_pred=q_pred
        )
