# -*- coding: utf-8 -*-
"""B4: HTT-Net (adapted) — MethodAdapter implementation.

Wraps the vendored `code/pipeline.py` (shared window-based preprocessing,
trimmed from `代码/main_experiment_3_fgds_psi_optimized.py`) and `code/model.py`
(HTTNet architecture, verbatim copy of `baselines/htt_net/model.py`) behind the
unified `shared.runners.method_adapter.MethodAdapter` interface.

Only task/seed/output routing changed vs. the original `baselines/htt_net/train.py`
— architecture, optimizer, lr, loss, and training protocol are unchanged. See
README.md and source_manifest.json for the full provenance/adaptation record.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

sys.path.insert(0, str(THIS_DIR / "code"))
sys.path.insert(0, str(REPO_ROOT / "shared"))

import pipeline as P  # noqa: E402
from model import HTTNet  # noqa: E402
from runners.method_adapter import MethodAdapter  # noqa: E402

HTT_ARCH = {
    "embed_dim": 32,
    "depths": (2, 2, 2, 2),
    "num_heads": 4,
    "window_size": 3,
    "dropout": P.BEST_ARCH["dropout"],  # 0.20
}
TRAIN_CFG = {
    "lr": P.BEST_ARCH["lr"],            # 5e-4
    "weight_decay": P.WEIGHT_DECAY,     # 1e-5
    "epochs": P.EPOCHS,                 # 120
    "patience": P.PATIENCE,             # 18
    "grad_clip": P.GRAD_CLIP,           # 1.0
}


def _class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    cnt = np.bincount(y, minlength=3).astype(float)
    w = cnt.sum() / (3 * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def _predict_stage_model(model, pack, device: str):
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for X, _ in pack["loader"]:
            p = F.softmax(model(X.to(device)), dim=1).detach().cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
    return np.concatenate(preds), np.concatenate(probs)


class HTTNetAdapter(MethodAdapter):
    method_id = "B4"
    method_name = "HTT-Net"

    def prepare(self) -> None:
        raw_df = P.load_feature_table()
        label_df = P.define_condition_relative_stages(raw_df)
        final_train_raw, final_val_raw, test_raw = P.split_grouped_lifecycle(
            label_df, self.train_cutters, self.test_cutter
        )

        raw_cols = P.get_raw_numeric_sensor_cols(final_train_raw)
        split_feat = P.build_online_features_by_split(
            {"final_train": final_train_raw, "final_internal_val": final_val_raw, "test": test_raw},
            raw_cols,
        )
        feat_train = split_feat[split_feat["split_name_for_feature_build"] == "final_train"].copy()
        feat_val = split_feat[split_feat["split_name_for_feature_build"] == "final_internal_val"].copy()
        feat_test = split_feat[split_feat["split_name_for_feature_build"] == "test"].copy()

        all_cols = P.feature_cols_from(feat_train)
        feat_train, feat_val = P.fill_by_train_median(feat_train, feat_val, all_cols)
        _, feat_test = P.fill_by_train_median(feat_train, feat_test, all_cols)
        selected = P.select_features_train_only(feat_train, preprocess_seed=self.preprocess_seed)
        feat_train, feat_val = P.fill_by_train_median(feat_train, feat_val, selected)
        _, feat_test = P.fill_by_train_median(feat_train, feat_test, selected)

        gmm, raw_to_order = P.fit_train_gmm(feat_train, preprocess_seed=self.preprocess_seed)
        feat_train = P.assign_fine_states(feat_train, gmm, raw_to_order)
        feat_val = P.assign_fine_states(feat_val, gmm, raw_to_order)
        feat_test = P.assign_fine_states(feat_test, gmm, raw_to_order)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(feat_train[selected].values)
        for df in [feat_train, feat_val, feat_test]:
            df[selected] = np.nan_to_num(scaler.transform(df[selected].values), nan=0.0, posinf=0.0, neginf=0.0)

        L = P.BEST_ARCH["L"]
        self._selected = selected
        self._tr_pack = P.make_pack(feat_train, selected, L, "final_train")
        self._va_pack = P.make_pack(feat_val, selected, L, "final_internal_val")
        self._te_pack = P.make_pack(feat_test, selected, L, "test")
        self._device = self.resolve_device()

    def build_model(self):
        model = HTTNet(input_dim=len(self._selected), num_classes=3, **HTT_ARCH)
        return model.to(self._device)

    def train(self, model) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
        w = _class_weights(self._tr_pack["ys"], self._device)
        best_state, best_score, wait = None, np.inf, 0

        for epoch in range(1, TRAIN_CFG["epochs"] + 1):
            model.train()
            epoch_losses = []
            for X, ys in self._tr_pack["loader"]:
                X, ys = X.to(self._device), ys.to(self._device)
                logits = model(X)
                loss = F.cross_entropy(logits, ys, weight=w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip"])
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))

            yv, pv = _predict_stage_model(model, self._va_pack, self._device)
            m = P.clf_metrics(self._va_pack["ys"], yv)
            val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), self._va_pack["ys"]], 1e-12, 1.0)))
            score = 0.7 * (1 - m["f1"]) + 1.0 * (1 - m["middle_recall"]) + 0.15 * val_loss
            scheduler.step(score)

            self.log_epoch(
                epoch=epoch,
                train_loss=float(np.mean(epoch_losses)),
                val_loss=float(val_loss),
                val_acc=m["acc"],
                val_macro_f1=m["f1"],
                val_middle_recall=m["middle_recall"],
                score=score,
            )
            if score < best_score:
                best_score, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
            if wait >= TRAIN_CFG["patience"]:
                break
        model.load_state_dict(best_state)

    def predict(self, model) -> pd.DataFrame:
        y_pred, prob = _predict_stage_model(model, self._te_pack, self._device)
        meta = self._te_pack["meta"].copy().reset_index(drop=True)
        out = pd.DataFrame({
            "run_id": meta["run_id"],
            "true_stage": meta["stage_true"],
            "pred_stage": [P.ID_TO_STAGE[int(v)] for v in y_pred],
            "p_early": prob[:, 0],
            "p_middle": prob[:, 1],
            "p_late": prob[:, 2],
        })
        return out

    def feature_hash(self) -> str | None:
        from utils.run_meta import dict_hash

        return dict_hash({"selected_features": sorted(getattr(self, "_selected", []))})

    def extra_run_meta(self) -> dict:
        return {"n_selected_features": len(getattr(self, "_selected", [])), "arch": HTT_ARCH}


ADAPTER_CLASS = HTTNetAdapter
