# -*- coding: utf-8 -*-
"""B5: Multi-source Channel-Spatial Attention (Multi-Attention-CNN), Unified
Protocol B adapter. See README.md for full method details.

Wraps the vendored code/ (model.py, signal_preprocessing.py, dcpsr_labels.py,
dataset.py) — all copied+adapted (task/seed/path routing only, per task spec
section 35) from the old project's `baselines/multi_source_attention/` at
legacy_git_commit 811da096ee47bea4f65db193aa49e793dba6f47d. See
source_manifest.json for exact file provenance/hashes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402

import dcpsr_labels as L  # noqa: E402
from dataset import ImagePairDataset  # noqa: E402
from model import MultiAttentionCNN  # noqa: E402

# Original PROTO_B_CFG from baselines/multi_source_attention/train.py, UNCHANGED
# except `seed` is no longer a fixed constant -- it comes from self.seed (task
# spec section 22-23: TRAIN_SEED must vary independently per run).
DEFAULT_CFG = dict(max_epochs=100, patience=15, batch_size=64, lr=0.001, weight_decay=1e-4)


class MultiSourceAttentionAdapter(MethodAdapter):
    method_id = "B5"
    method_name = "Multi-source Channel-Spatial Attention"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_smoke = False
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update({k: v for k, v in self.config.items() if k in DEFAULT_CFG})
        self._train_meta = self._val_meta = self._test_meta = None

    # Base class doesn't currently thread `smoke_test` into prepare()/predict()
    # (flagged in fork report) -- this tiny override captures it locally.
    def run(self, resume: bool = True, smoke_test: bool = False) -> dict:
        self._is_smoke = smoke_test
        return super().run(resume=resume, smoke_test=smoke_test)

    def prepare(self) -> None:
        train_df, val_df, test_df = L.get_task_split(self.train_cutters, self.test_cutter)
        train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
        for d in (train_df, val_df, test_df):
            d["label"] = d["stage_id"]
        if self._is_smoke:
            # Smoke test: a handful of rows per split is enough to verify
            # end-to-end wiring/shapes without building ~300 CWT images.
            train_df = train_df.head(4)
            val_df = val_df.head(2)
            test_df = test_df.head(3)
        self._train_meta, self._val_meta, self._test_meta = train_df, val_df, test_df

    def build_model(self) -> Any:
        model = MultiAttentionCNN()
        return model.to(self.resolve_device())

    def train(self, model: Any) -> None:
        device = self.resolve_device()
        train_loader = DataLoader(ImagePairDataset(self._train_meta, "label"),
                                   batch_size=self.cfg["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(ImagePairDataset(self._val_meta, "label"),
                                 batch_size=self.cfg["batch_size"], shuffle=False, num_workers=0)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

        best_val_acc, best_epoch, patience_ctr = -1.0, -1, 0
        best_state = None
        for epoch in range(self.cfg["max_epochs"]):
            tr_loss, tr_acc = _run_epoch(model, train_loader, device, opt)
            va_loss, va_acc = _run_epoch(model, val_loader, device, None)
            self.log_epoch(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc)
            improved = va_acc > best_val_acc
            if improved:
                best_val_acc, best_epoch = va_acc, epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= self.cfg["patience"]:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._best_epoch = best_epoch
        self._best_val_acc = best_val_acc

    def predict(self, model: Any) -> pd.DataFrame:
        device = self.resolve_device()
        test_loader = DataLoader(ImagePairDataset(self._test_meta, "label"),
                                  batch_size=self.cfg["batch_size"], shuffle=False, num_workers=0)
        model.eval()
        all_probs, all_true = [], []
        with torch.no_grad():
            for force, vib, label in test_loader:
                force, vib = force.to(device), vib.to(device)
                logits = model(force, vib)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_true.append(label.numpy())
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 3))
        true_ids = np.concatenate(all_true) if all_true else np.zeros((0,), dtype=int)
        pred_ids = probs.argmax(axis=1) if len(probs) else np.zeros((0,), dtype=int)

        out = pd.DataFrame({
            "run_id": self._test_meta["run_id"].values.astype(int),
            "true_stage": [L.ID_TO_STAGE[i] for i in true_ids],
            "pred_stage": [L.ID_TO_STAGE[i] for i in pred_ids],
            "p_early": probs[:, 0] if len(probs) else [],
            "p_middle": probs[:, 1] if len(probs) else [],
            "p_late": probs[:, 2] if len(probs) else [],
        })
        return out

    def extra_run_meta(self) -> dict:
        return {"best_epoch": getattr(self, "_best_epoch", None),
                "best_val_acc_source_only": getattr(self, "_best_val_acc", None)}


def _run_epoch(model, loader, device, optimizer):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for force, vib, label in loader:
            force, vib, label = force.to(device), vib.to(device), label.to(device)
            logits = model(force, vib)
            loss = F.cross_entropy(logits, label)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(label)
            correct += (logits.argmax(dim=1) == label).sum().item()
            n += len(label)
    return total_loss / max(n, 1), correct / max(n, 1)


ADAPTER_CLASS = MultiSourceAttentionAdapter
