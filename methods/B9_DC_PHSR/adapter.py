"""B9 (DC-PHSR -- paper name; legacy code identifiers DC-PSR / B12 / FGDS-PSI)
adapter.

DC-PHSR = B3's Multi-task TCN-GRU backbone + a deterministic post-hoc
probability-inference step (methods/_internal_shared/code/pipeline.py::
apply_probability_inference, frozen B12_PARAMS). See
methods/_internal_shared/code/pipeline.py's module docstring for what was
changed vs. the original code.

**Known scope limitation, by deliberate choice (2026-08-30):** B9 trains its
OWN copy of the identical B3 backbone inside this (method=B9, task, seed) cell,
rather than reusing an already-trained B3 checkpoint from a sibling
methods/B3_Multitask_TCN_GRU run. EXPERIMENT_REGISTRY.md section 34 requires
checkpoint reuse only when seed/task/config_hash/preprocess_hash all match
exactly -- building that cross-adapter lookup+verification plumbing safely is
real work, and doing it hastily risks exactly the bug that section warns
against ("B9 seed17 错用 B3 seed42 checkpoint"). For this round, correctness
and safety win over the (real, but modest -- B3 trains in a few minutes) compute
duplication. A future optimization: have run_phm2010.py's orchestrator train
B3 once per (task, seed) and hand both B3's and B9's adapters the same
checkpoint path with a verified hash match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "_internal_shared" / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402
from utils.run_meta import dict_hash  # noqa: E402
import pipeline as pl  # noqa: E402


class B9DCPHSRAdapter(MethodAdapter):
    method_id = "B9"
    method_name = "DC-PHSR"

    def prepare(self) -> None:
        (self.tr_pack, self.va_pack, self.te_pack, self.selected, self.selected_df,
         self.feat_train, self.feat_test) = pl.prepare_task_data(
            self.train_cutters, self.test_cutter, preprocess_seed=self.preprocess_seed
        )
        self.input_dim = len(self.selected)

    def build_model(self):
        return pl.build_multitask_model(self.input_dim, self.resolve_device())

    def train(self, model) -> None:
        model, hist, best_score, best_epoch = pl.train_multitask_model(
            model, self.tr_pack, self.va_pack, self.resolve_device(), log_fn=self.log_epoch
        )
        self._best_epoch = best_epoch
        self._best_score = best_score

    def predict(self, model) -> pd.DataFrame:
        raw = pl.predict_multitask_model(model, self.te_pack, self.resolve_device())
        inferred = pl.apply_probability_inference(raw, pl.B12_PARAMS)
        out = pd.DataFrame({
            "run_id": inferred["cut_index"].astype(int),
            "true_stage": inferred["stage_true"],
            "pred_stage": [pl.ID_TO_STAGE[int(v)] for v in inferred["stage_pred_final"]],
            "p_early": inferred["final_prob_early"],
            "p_middle": inferred["final_prob_middle"],
            "p_late": inferred["final_prob_late"],
            "q_true": inferred["q_true_model"],
            "q_pred": inferred["q_hat"],
        })
        return out

    def feature_hash(self) -> str | None:
        return dict_hash({"selected_features": self.selected})

    def extra_run_meta(self) -> dict:
        return {
            "best_epoch": getattr(self, "_best_epoch", None),
            "best_score": getattr(self, "_best_score", None),
            "n_selected_features": len(self.selected),
            "arch": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pl.BEST_ARCH.items()},
            "B12_PARAMS_legacy_name": pl.B12_PARAMS,
            "checkpoint_sharing_with_B3": "not enabled this round -- see adapter.py module docstring",
        }


ADAPTER_CLASS = B9DCPHSRAdapter
