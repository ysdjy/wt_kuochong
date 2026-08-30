"""B3 (Multi-task TCN-GRU) adapter -- DC-PHSR's backbone.

Wraps methods/_internal_shared/code/pipeline.py (vendored from
代码/main_experiment_3_fgds_psi_optimized.py + the validated D2/D3
generalization in final_statistical_evidence/scripts/methods/common_pipeline.py).
See methods/_internal_shared/code/pipeline.py's module docstring for the exact,
itemized list of what was changed vs. the original code (only task/seed/output
routing and paths -- architecture/hyperparameters untouched).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "_internal_shared" / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402
from utils.run_meta import dict_hash  # noqa: E402
import pipeline as pl  # noqa: E402


class B3MultitaskTCNGRUAdapter(MethodAdapter):
    method_id = "B3"
    method_name = "Multi-task TCN-GRU"

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
        out = pd.DataFrame({
            "run_id": raw["cut_index"].astype(int),
            "true_stage": raw["stage_true"],
            "pred_stage": [pl.ID_TO_STAGE[int(v)] for v in raw["stage_pred_raw"]],
            "p_early": raw["raw_prob_early"],
            "p_middle": raw["raw_prob_middle"],
            "p_late": raw["raw_prob_late"],
            "q_true": raw["q_true_model"],
            "q_pred": raw["q_hat"],
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
        }


ADAPTER_CLASS = B3MultitaskTCNGRUAdapter
