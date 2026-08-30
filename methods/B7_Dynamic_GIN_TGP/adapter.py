# -*- coding: utf-8 -*-
"""B7: Dynamic GIN + TGP adapter.

Wraps the vendored `code/model.py` (DynamicGIN_TGP, unchanged from the
original published-baseline reproduction) and `code/preprocessing.py` /
`code/stage_labels.py` (path-routing/task-generalization only, per task spec
section 35) behind the unified MethodAdapter interface.

Only the "Protocol B" (unified DC-PSR-comparison) training regime is
implemented here — this repo's formal comparison table never uses the
paper-native "Protocol A" sanity-check split (see old
`baselines/dynamic_gin_tgp/train.py` docstring for that distinction).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from runners.method_adapter import MethodAdapter  # noqa: E402

import preprocessing as P  # noqa: E402
import stage_labels as SL  # noqa: E402
from model import DynamicGIN_TGP  # noqa: E402

DEFAULT_CFG = dict(max_epochs=50, patience=15, batch_size=4, lr=1e-4, weight_decay=0.1,
                    plateau_patience=10, topk=144)


class DynamicGinTgpDataset:
    """One row per (condition, run_id) x 10 portions. Rows are shuffled once
    (fixed shuffle_seed, NOT self.seed) at construction — required because the
    model's static graph (Eq.7-9) is built by concatenating every sample in a
    batch before computing cosine similarity, so an eval loader built with
    shuffle=False from a manifest grouped by run would put only one run's
    (hence one label's) samples in a batch, letting the true label leak into
    each sample's own prediction via its batch-mates. See old
    baselines/dynamic_gin_tgp/train.py::ProtocolBSegmentDataset docstring for
    the full incident this was caught from (val_run_acc hit 100% by epoch 1).
    shuffle_seed is fixed (not TRAIN_SEED) so evaluation order/results stay
    reproducible across repeated calls regardless of the training seed."""

    def __init__(self, label_df: pd.DataFrame, shuffle_seed: int = 12345):
        rows = []
        for _, r in label_df.iterrows():
            for portion_idx in range(10):
                rows.append({"condition": r["condition"], "run_id": int(r["run_id"]),
                             "stage_id": int(r["stage_id"]), "portion_idx": portion_idx})
        self.rows = pd.DataFrame(rows).sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import torch
        r = self.rows.iloc[idx]
        windows = P.load_windows(r.condition, int(r.run_id))
        x = windows[int(r.portion_idx)]
        return torch.from_numpy(x), int(r.stage_id)


def _make_loader(ds, batch_size, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class DynamicGinTgpAdapter(MethodAdapter):
    method_id = "B7"
    method_name = "Dynamic GIN + TGP"

    def prepare(self) -> None:
        cfg = {**DEFAULT_CFG, **self.config}
        self.cfg = cfg
        debug_max_runs = self.config.get("debug_max_runs")

        all_conditions = tuple(sorted(set(self.train_cutters) | {self.test_cutter}))
        P.build_windows_cache(conditions=all_conditions, max_runs=debug_max_runs, verbose=False)

        self.tr_df, self.va_df, self.te_df = SL.get_task_split(
            self.train_cutters, self.test_cutter, conditions=all_conditions
        )
        # If debug_max_runs truncated the cache, also truncate the label
        # frames to runs actually cached (smoke-test convenience only).
        if debug_max_runs is not None:
            self.tr_df = self.tr_df[self.tr_df["run_id"] <= debug_max_runs].reset_index(drop=True)
            self.va_df = self.va_df[self.va_df["run_id"] <= debug_max_runs].reset_index(drop=True)
            self.te_df = self.te_df[self.te_df["run_id"] <= debug_max_runs].reset_index(drop=True)

        self.train_ds = DynamicGinTgpDataset(self.tr_df)
        self.val_ds = DynamicGinTgpDataset(self.va_df)
        self.test_ds = DynamicGinTgpDataset(self.te_df)

    def build_model(self):
        import torch
        device = self.resolve_device()
        self._device = device
        model = DynamicGIN_TGP(topk=self.cfg["topk"]).to(device)
        return model

    def train(self, model) -> None:
        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        device = self._device
        train_loader = _make_loader(self.train_ds, cfg["batch_size"], True)
        val_loader = _make_loader(self.val_ds, cfg["batch_size"], False)

        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=cfg["plateau_patience"]
        )

        def run_epoch(loader, optimizer=None):
            train_mode = optimizer is not None
            model.train() if train_mode else model.eval()
            total_loss, n, correct = 0.0, 0, 0
            with torch.set_grad_enabled(train_mode):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if train_mode:
                        optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    if train_mode:
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item() * len(y)
                    correct += (logits.argmax(1) == y).sum().item()
                    n += len(y)
            return total_loss / max(n, 1), correct / max(n, 1)

        def run_level_val_acc():
            model.eval()
            probs, rows = [], []
            with torch.no_grad():
                for x, y in _make_loader(self.val_ds, cfg["batch_size"], False):
                    x = x.to(device)
                    probs.append(F.softmax(model(x), dim=1).cpu().numpy())
            probs = np.concatenate(probs)
            df = self.val_ds.rows.copy()
            df["p_early"], df["p_middle"], df["p_late"] = probs[:, 0], probs[:, 1], probs[:, 2]
            agg = df.groupby(["condition", "run_id"]).agg(
                stage_id=("stage_id", "first"),
                p_early=("p_early", "mean"), p_middle=("p_middle", "mean"), p_late=("p_late", "mean"),
            ).reset_index()
            pred = agg[["p_early", "p_middle", "p_late"]].values.argmax(axis=1)
            return float((pred == agg["stage_id"]).mean())

        best_val_acc, best_epoch, patience_ctr, best_state = -1.0, -1, 0, None
        for epoch in range(cfg["max_epochs"]):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(train_loader, opt)
            va_acc = run_level_val_acc()
            scheduler.step(va_acc)
            improved = va_acc > best_val_acc
            if improved:
                best_val_acc, best_epoch, patience_ctr = va_acc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
            self.log_epoch(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_run_level_acc=va_acc, time_s=time.time() - t0,
                            lr=opt.param_groups[0]["lr"])
            if patience_ctr >= cfg["patience"]:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
        self._best_epoch = best_epoch
        self._best_val_acc = best_val_acc

    def predict(self, model) -> pd.DataFrame:
        import torch
        import torch.nn.functional as F

        device = self._device
        loader = _make_loader(self.test_ds, self.cfg["batch_size"], False)
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device)
                probs.append(F.softmax(model(x), dim=1).cpu().numpy())
        probs = np.concatenate(probs) if probs else np.zeros((0, 3))

        df = self.test_ds.rows.copy()
        if len(df) != len(probs):
            df = df.iloc[:len(probs)]
        df["p_early"], df["p_middle"], df["p_late"] = probs[:, 0], probs[:, 1], probs[:, 2]
        agg = df.groupby(["condition", "run_id"]).agg(
            stage_id=("stage_id", "first"),
            p_early=("p_early", "mean"), p_middle=("p_middle", "mean"), p_late=("p_late", "mean"),
        ).reset_index()
        agg["pred_id"] = agg[["p_early", "p_middle", "p_late"]].values.argmax(axis=1)
        agg["true_stage"] = agg["stage_id"].map(SL.ID_TO_STAGE)
        agg["pred_stage"] = agg["pred_id"].map(SL.ID_TO_STAGE)
        return agg[["run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]].sort_values(
            "run_id"
        ).reset_index(drop=True)

    def extra_run_meta(self) -> dict:
        return {"best_epoch": getattr(self, "_best_epoch", None),
                "best_val_run_level_acc": getattr(self, "_best_val_acc", None)}


ADAPTER_CLASS = DynamicGinTgpAdapter
