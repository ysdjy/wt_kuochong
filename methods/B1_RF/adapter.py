"""B1: Random Forest baseline adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(_THIS_DIR / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402
from utils.run_meta import dict_hash  # noqa: E402

import preprocessing as prep  # noqa: E402


class RFAdapter(MethodAdapter):
    method_id = "B1"
    method_name = "RF"

    N_ESTIMATORS = 400
    CLASS_WEIGHT = "balanced_subsample"

    def prepare(self) -> None:
        (self.feat_train, self.feat_val, self.feat_test,
         self.selected, self.meta_test) = prep.prepare_task_data(
            self.train_cutters, self.test_cutter, self.preprocess_seed
        )

    def build_model(self):
        from sklearn.ensemble import RandomForestClassifier

        # A smoke test only needs to prove the plumbing works, not produce a
        # real result -- shrink n_estimators drastically for speed. Real runs
        # (smoke_test=False) always use the full, paper-faithful N_ESTIMATORS.
        n_estimators = 5 if self.smoke_test else self.N_ESTIMATORS
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.seed,
            class_weight=self.CLASS_WEIGHT,
            n_jobs=-1,
        )

    def train(self, model) -> None:
        import time

        Xtr = self.feat_train[self.selected].values
        ytr = self.feat_train["stage_id"].values.astype(int)
        t0 = time.time()
        model.fit(Xtr, ytr)
        self.log_epoch(step=0, train_seconds=time.time() - t0, n_train_rows=len(Xtr))

    def predict(self, model) -> pd.DataFrame:
        # Unlike a neural net's forward pass, sklearn's RandomForestClassifier
        # cannot predict at all without being fit first. The base class's
        # run() skips train() when smoke_test=True (correct for the other 8
        # NN-based methods, which produce valid-shaped garbage from a random
        # init); RF needs a real (just drastically abbreviated) fit instead.
        if self.smoke_test:
            from sklearn.exceptions import NotFittedError

            try:
                model.predict(self.feat_test[self.selected].values[:1])
            except NotFittedError:
                self.train(model)

        test_by_run = self.feat_test.set_index("run_id")
        # Evaluate only on the windowed-feasible test run_ids (meta_test["cut_index"])
        # so RF's test universe matches the other window-based methods' (L=12
        # common evaluation universe) -- matches historical run_rf() exactly.
        run_ids = self.meta_test["cut_index"].values
        Xte = test_by_run.loc[run_ids, self.selected].values
        y_pred = model.predict(Xte)
        prob = model.predict_proba(Xte)

        df = pd.DataFrame({
            "run_id": run_ids,
            "true_stage": self.meta_test["stage_true"].values,
            "pred_stage": [prep.ID_TO_STAGE[int(v)] for v in y_pred],
            "p_early": prob[:, 0],
            "p_middle": prob[:, 1],
            "p_late": prob[:, 2],
        })
        return df

    def feature_hash(self) -> str | None:
        return dict_hash({"selected_features": sorted(self.selected)}) if hasattr(self, "selected") else None


ADAPTER_CLASS = RFAdapter
