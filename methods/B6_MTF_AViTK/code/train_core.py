# -*- coding: utf-8 -*-
"""Model-building / epoch / inference / run-level-aggregation helpers, ported
from the old project's `baselines/mtf_avitk/train.py::run_protocol_b` (its
Protocol B / "Unified" path only -- Protocol A, the original-paper sanity
check, and the old smoke-test helper are not needed by this adapter).

ADAPTATION vs. the original (task spec section 35, allowed: seed routing):
the original hardcoded `PROTO_B_CFG["seed"] = 42` and called its own
`set_seed(cfg["seed"])` inside `run_protocol_b`. Here, seeding is the CALLER's
responsibility (`shared/utils/seeding.py::seed_everything`, called by
`shared/runners/method_adapter.py::MethodAdapter.run()` immediately before
`build_model()`) -- this module performs NO seeding itself. Architecture
(ViT-L/32 + KAN), optimizer (SGD, lr=0.0006, wd=2e-4, momentum=0.9), loss
(cross-entropy), and AMP/grad-checkpoint/grad-accum mechanics are otherwise
byte-for-byte identical to the original.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MTF_AViTK

# Explicit: PAPER_SPEC.md Table 5 (unchanged from the original)
PROTO_B_CFG = dict(max_epochs=50, patience=10, batch_size=8, lr=0.0006, weight_decay=2e-4, momentum=0.9)


def build_model(device: str, grad_checkpoint: bool) -> MTF_AViTK:
    model = MTF_AViTK().to(device)
    if grad_checkpoint:
        model.backbone.grad_checkpointing = True
    return model


def run_epoch(model, loader, device, optimizer=None, scaler=None, grad_accum_steps=1):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n, correct = 0.0, 0, 0
    use_amp = device == "cuda"
    if train_mode:
        optimizer.zero_grad()
    with torch.set_grad_enabled(train_mode):
        for i, (img, label) in enumerate(loader):
            img, label = img.to(device), label.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(img)
                loss = F.cross_entropy(logits, label)
            if train_mode:
                loss_scaled = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                if (i + 1) % grad_accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            total_loss += loss.item() * len(label)
            correct += (logits.argmax(dim=1) == label).sum().item()
            n += len(label)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    use_amp = device == "cuda"
    for img, label in loader:
        img = img.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(img)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_true.append(label.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)


def build_pred_df_subwindow(rows: pd.DataFrame, probs: np.ndarray, true_ids: np.ndarray) -> pd.DataFrame:
    pred_ids = probs.argmax(axis=1)
    return pd.DataFrame({
        "condition": rows["condition"].values,
        "run_id": rows["run_id"].values.astype(int),
        "subwindow": rows["subwindow"].values.astype(int),
        "stage_true_id": true_ids.astype(int),
        "prob_early": probs[:, 0], "prob_middle": probs[:, 1], "prob_late": probs[:, 2],
    })


def aggregate_to_run_level(sw_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-probability aggregation of the 5 sub-window predictions per run
    -> one run-level prediction (the old project's documented
    reimplementation choice: the original MTF-AViTK paper evaluates at
    image/sub-window level, not run level; this project's unified comparison
    requires one prediction per physical run, matching every other method)."""
    agg = sw_df.groupby(["condition", "run_id"]).agg(
        stage_true_id=("stage_true_id", "first"),
        prob_early=("prob_early", "mean"),
        prob_middle=("prob_middle", "mean"),
        prob_late=("prob_late", "mean"),
    ).reset_index()
    probs = agg[["prob_early", "prob_middle", "prob_late"]].values
    agg["pred_id"] = probs.argmax(axis=1)
    return agg


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
