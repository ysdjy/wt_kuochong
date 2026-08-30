# -*- coding: utf-8 -*-
"""B8: DP2Net-adapted (pooled source) adapter.

Pins the old code's "Protocol B-D1" variant — pooled-source (train_cutters ->
test_cutter, generalized from the paper's own C1+C4->C6) — as the ONE official
B8 entry in this repo's formal comparison table. This is the variant the old
project's own docstring names "DP2Net-adapted (pooled source)" and states
explicitly enters the D1 main comparison table (NOT "original DP2Net", which
is single-source-only and paper-native — see old
`baselines/dp2net/train.py::run_protocol_b` docstring, `variant="B-D1"`).
Protocol A (paper-native sanity check) and Protocol B-S (native single-source)
are intentionally not ported here.

Wraps the vendored `code/model.py` (S/G/F, unchanged) and `code/preprocessing.py`
(path-routing only) behind the unified MethodAdapter interface, following
Algorithm 1's two-stage protocol: pretrain S+F (classification), then train
G+F with S frozen (MMD-guided generation). Inference uses trained S+F only
(G is training-time-only, per the paper's own Algorithm 1 "Inference: Model =
Trained S and F.").
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
from model import SpatialAttention, Generator, WDCNN, mmd_loss  # noqa: E402

DEFAULT_CFG = dict(pretrain_epochs=100, gen_epochs=100, batch_size=64, lr_s=1e-3, lr_f=1e-3,
                    lr_g=1e-5, cosine_period=20, alpha=20.0, windows_per_run=8)


class UnifiedWindowDataset:
    def __init__(self, manifest: pd.DataFrame, label_df: pd.DataFrame):
        merged = manifest.merge(
            label_df[["condition", "run_id", "stage_id"]], on=["condition", "run_id"], how="inner"
        )
        self.rows = merged.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import torch
        r = self.rows.iloc[idx]
        x = P.load_window(r.npy_path)
        return torch.from_numpy(x).unsqueeze(0), int(r.stage_id)


def _make_loader(ds, batch_size, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class Dp2NetAdapter(MethodAdapter):
    method_id = "B8"
    method_name = "DP2Net-adapted (pooled source)"

    def prepare(self) -> None:
        cfg = {**DEFAULT_CFG, **self.config}
        self.cfg = cfg
        debug_max_runs = self.config.get("debug_max_runs")

        all_conditions = tuple(sorted(set(self.train_cutters) | {self.test_cutter}))
        manifest = P.build_unified_windows_cache(
            conditions=all_conditions, windows_per_run=cfg["windows_per_run"],
            max_runs=debug_max_runs, verbose=False,
        )
        self.tr_df, self.va_df, self.te_df = SL.get_task_split(
            self.train_cutters, self.test_cutter, conditions=all_conditions
        )
        if debug_max_runs is not None:
            self.tr_df = self.tr_df[self.tr_df["run_id"] <= debug_max_runs].reset_index(drop=True)
            self.va_df = self.va_df[self.va_df["run_id"] <= debug_max_runs].reset_index(drop=True)
            self.te_df = self.te_df[self.te_df["run_id"] <= debug_max_runs].reset_index(drop=True)

        self.train_ds = UnifiedWindowDataset(manifest, self.tr_df)
        self.val_ds = UnifiedWindowDataset(manifest, self.va_df)
        self.test_ds = UnifiedWindowDataset(manifest, self.te_df)

    def build_model(self):
        device = self.resolve_device()
        self._device = device
        s, g, f = SpatialAttention().to(device), Generator().to(device), WDCNN(num_classes=3).to(device)
        return {"s": s, "g": g, "f": f}

    def train(self, model) -> None:
        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        device = self._device
        s, g, f = model["s"], model["g"], model["f"]

        train_loader = _make_loader(self.train_ds, cfg["batch_size"], True)
        val_loader = _make_loader(self.val_ds, cfg["batch_size"], False) if len(self.val_ds) else None

        @torch.no_grad()
        def eval_acc():
            if val_loader is None:
                return float("nan")
            s.eval(); f.eval()
            correct, n = 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                fa, _ = s(x)
                pred = f(fa).argmax(1)
                correct += (pred == y).sum().item()
                n += len(y)
            return correct / max(n, 1)

        # --- Stage 1: pretrain S+F (classification) ---
        opt1 = torch.optim.Adam(list(s.parameters()) + list(f.parameters()), lr=cfg["lr_s"])
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg["cosine_period"])
        for epoch in range(cfg["pretrain_epochs"]):
            t0 = time.time()
            s.train(); f.train()
            total_loss, n, correct = 0.0, 0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt1.zero_grad()
                fa, _ = s(x)
                logits = f(fa)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt1.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
            sched1.step()
            va_acc = eval_acc()
            self.log_epoch(stage="pretrain", epoch=epoch, train_loss=total_loss / max(n, 1),
                            train_acc=correct / max(n, 1), val_acc=va_acc, time_s=time.time() - t0)

        # --- Stage 2: train G+F, S frozen (MMD-guided generation) ---
        for p in s.parameters():
            p.requires_grad_(False)
        vst = torch.from_numpy(P.build_vst()).float().unsqueeze(0).to(device)
        opt_g = torch.optim.Adam(g.parameters(), lr=cfg["lr_g"])
        opt_f = torch.optim.Adam(f.parameters(), lr=cfg["lr_f"])
        for epoch in range(cfg["gen_epochs"]):
            t0 = time.time()
            g.train(); f.train(); s.eval()
            total_lg, total_ltask, n = 0.0, 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    fa, _ = s(x)
                wg = g(fa)
                l_mse = F.mse_loss(wg, vst.expand_as(wg))
                fg = wg * x
                with torch.no_grad():
                    _, emb_s = f(fa, return_features=True)
                _, emb_g = f(fg, return_features=True)
                l_mmd = mmd_loss(emb_s, emb_g)
                l_g = l_mse - cfg["alpha"] * l_mmd
                opt_g.zero_grad()
                l_g.backward()
                opt_g.step()

                with torch.no_grad():
                    fa2, _ = s(x)
                    wg2 = g(fa2)
                    fg2 = wg2 * x
                logits_s = f(fa2)
                logits_g = f(fg2)
                l_task = F.cross_entropy(logits_s, y) + F.cross_entropy(logits_g, y)
                opt_f.zero_grad()
                l_task.backward()
                opt_f.step()

                total_lg += l_g.item() * len(y)
                total_ltask += l_task.item() * len(y)
                n += len(y)
            va_acc = eval_acc()
            self.log_epoch(stage="generalize", epoch=epoch, l_g=total_lg / max(n, 1),
                            l_task=total_ltask / max(n, 1), val_acc=va_acc, time_s=time.time() - t0)

    def predict(self, model) -> pd.DataFrame:
        import torch
        import torch.nn.functional as F

        device = self._device
        s, f = model["s"], model["f"]
        loader = _make_loader(self.test_ds, self.cfg["batch_size"], False)
        s.eval(); f.eval()
        probs = []
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device)
                fa, _ = s(x)
                probs.append(F.softmax(f(fa), dim=1).cpu().numpy())
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


ADAPTER_CLASS = Dp2NetAdapter
