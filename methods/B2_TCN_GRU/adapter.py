"""B2 (TCN-GRU, stage-only) adapter. Wraps methods/B2_TCN_GRU/code/model.py
(vendored stage-only model) on top of methods/_internal_shared/code/pipeline.py
(vendored shared data pipeline)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "_internal_shared" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402
from utils.run_meta import dict_hash  # noqa: E402
import pipeline as pl  # noqa: E402
import model as m  # noqa: E402


class B2TCNGRUAdapter(MethodAdapter):
    method_id = "B2"
    method_name = "TCN-GRU"

    def prepare(self) -> None:
        (self.tr_pack, self.va_pack, self.te_pack, self.selected, self.selected_df,
         self.feat_train, self.feat_test) = pl.prepare_task_data(
            self.train_cutters, self.test_cutter, preprocess_seed=self.preprocess_seed
        )
        self.input_dim = len(self.selected)

    def build_model(self):
        return m.build_stage_model(self.input_dim, self.resolve_device())

    def train(self, model) -> None:
        _, best_info = m.train_stage_model(
            model, self.tr_pack, self.va_pack, self.resolve_device(),
            epochs=pl.EPOCHS, patience=pl.PATIENCE, log_fn=self.log_epoch,
        )
        self._best_info = best_info

    def predict(self, model) -> pd.DataFrame:
        y_pred, prob = m.predict_stage_model(model, self.te_pack, self.resolve_device())
        meta = self.te_pack["meta"].reset_index(drop=True)
        out = pd.DataFrame({
            "run_id": meta["cut_index"].astype(int),
            "true_stage": meta["stage_true"],
            "pred_stage": [pl.ID_TO_STAGE[int(v)] for v in y_pred],
            "p_early": prob[:, 0],
            "p_middle": prob[:, 1],
            "p_late": prob[:, 2],
        })
        return out

    def feature_hash(self) -> str | None:
        return dict_hash({"selected_features": self.selected})

    def extra_run_meta(self) -> dict:
        return {
            **getattr(self, "_best_info", {}),
            "n_selected_features": len(self.selected),
            "arch": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pl.BEST_ARCH.items()},
        }


ADAPTER_CLASS = B2TCNGRUAdapter
