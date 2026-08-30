# -*- coding: utf-8 -*-
"""B6 MTF-AViTK method adapter.

309M-param ViT-L/32 + KAN classifier head, MTF (Markov Transition Field)
image input from raw PHM2010 force signal. See README.md for full method
details, hyperparameters, and reproduction notes.

WARNING (see README.md "Compute cost"): a real training run of this method
is historically ~30-40 minutes on an 8GB laptop GPU, `best.pt` checkpoint
~1.2GB. This adapter's `train()` implements the REAL training loop
(byte-for-byte the original SGD/AMP/early-stopping logic) for future formal
runs, but this round only ran a CPU-only plumbing smoke test -- see
`tests/smoke_test.py` and its recorded output in README.md. Never invoke
`run(smoke_test=False)` for this method without explicit user authorization
per this project's training-execution policy.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402

import data_prep as D  # noqa: E402
import label_utils as L  # noqa: E402
import train_core as TC  # noqa: E402
from train_core import PROTO_B_CFG  # noqa: E402


class MTFAViTKAdapter(MethodAdapter):
    method_id = "B6"
    method_name = "MTF-AViTK"

    def run(self, resume: bool = True, smoke_test: bool = False) -> dict:
        # Stash the flag so predict() can stay cheap during a smoke test --
        # the base class template does not pass smoke_test into predict().
        self._smoke_test = smoke_test
        return super().run(resume=resume, smoke_test=smoke_test)

    # ---- hooks -----------------------------------------------------------

    def prepare(self) -> None:
        self._smoke_test = getattr(self, "_smoke_test", False)
        cfg = dict(PROTO_B_CFG)
        cfg.update(self.config or {})
        self.cfg = cfg
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        self.batch_size = int(cfg.get("batch_size", PROTO_B_CFG["batch_size"]))
        self.device_resolved = self.resolve_device()

        # Labeling/split is cheap (just wear-CSV parsing + arithmetic) --
        # always build it, even during a smoke test.
        self.meta = D.build_metadata(self.train_cutters, self.test_cutter)

    def build_model(self):
        return TC.build_model(self.device_resolved, self.grad_checkpoint)

    def train(self, model) -> None:
        """Real Protocol-B training loop (never called during smoke_test --
        the base class only calls train() when smoke_test=False)."""
        train_ds = D.MTFImageDataset(self.meta["train"])
        val_ds = D.MTFImageDataset(self.meta["val"])
        train_loader = TC.make_loader(train_ds, self.batch_size, shuffle=True)
        val_loader = TC.make_loader(val_ds, self.batch_size, shuffle=False)

        opt = torch.optim.SGD(model.parameters(), lr=self.cfg["lr"], momentum=self.cfg["momentum"],
                               weight_decay=self.cfg["weight_decay"])
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device_resolved == "cuda"))

        best_val_acc, best_epoch, patience_ctr = -1.0, -1, 0
        self._best_state = None
        for epoch in range(self.cfg["max_epochs"]):
            t0 = time.time()
            tr_loss, tr_acc = TC.run_epoch(model, train_loader, self.device_resolved, opt, scaler)
            probs, true_ids = TC.predict(model, val_loader, self.device_resolved)
            sw = TC.build_pred_df_subwindow(self.meta["val"], probs, true_ids)
            run_lvl = TC.aggregate_to_run_level(sw)
            va_acc = float((run_lvl["pred_id"] == run_lvl["stage_true_id"]).mean())
            improved = va_acc > best_val_acc
            if improved:
                best_val_acc, best_epoch = va_acc, epoch
                self._best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            self.log_epoch(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_run_level_acc=va_acc, best_val_acc=best_val_acc,
                            best_epoch=best_epoch, time_s=time.time() - t0)
            if self.device_resolved == "cuda":
                torch.cuda.empty_cache()
            if patience_ctr >= self.cfg["patience"]:
                break
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            model.to(self.device_resolved)
        self._trained_epochs = len(self._training_log_rows)

        if self.config.get("save_checkpoint") == "final" and self._best_state is not None:
            torch.save(self._best_state, self.output_dir / "checkpoint_best.pth")

    def predict(self, model) -> pd.DataFrame:
        test_rows = self.meta["test"]
        if getattr(self, "_smoke_test", False):
            # Absolute minimum plumbing check: ONE real sub-window image,
            # single forward pass, no aggregation loop over the full test set.
            test_rows = test_rows.iloc[[0]]
        ds = D.MTFImageDataset(test_rows)
        loader = TC.make_loader(ds, batch_size=min(self.batch_size, len(test_rows)) or 1, shuffle=False)
        probs, true_ids = TC.predict(model, loader, self.device_resolved)
        sw = TC.build_pred_df_subwindow(test_rows, probs, true_ids)
        run_lvl = TC.aggregate_to_run_level(sw)
        return pd.DataFrame({
            "run_id": run_lvl["run_id"].astype(int),
            "true_stage": run_lvl["stage_true_id"].map(L.ID_TO_STAGE),
            "pred_stage": run_lvl["pred_id"].map(L.ID_TO_STAGE),
            "p_early": run_lvl["prob_early"],
            "p_middle": run_lvl["prob_middle"],
            "p_late": run_lvl["prob_late"],
        })

    # ---- provenance --------------------------------------------------------

    def extra_run_meta(self) -> dict:
        return {
            "n_params": None,  # filled lazily below if a model was built during this run
            "grad_checkpoint": self.grad_checkpoint,
            "batch_size": self.batch_size,
            "trained_epochs": getattr(self, "_trained_epochs", None),
        }


ADAPTER_CLASS = MTFAViTKAdapter
