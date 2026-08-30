"""B2 (TCN-GRU, stage-only baseline) model + train/predict.

Vendored verbatim (architecture/hyperparameters unchanged) from
final_statistical_evidence/scripts/methods/run_internal_methods_transfer_task.py
(TCNGRUStageOnly, class_weights, predict_stage_model, train_stage_model), which
itself is a copy of 代码/7.7跨工况实验.py section 2. Only change: model
CONSTRUCTION is factored out into build_stage_model() (called separately from
training) so it fits this repo's MethodAdapter.build_model()/train() hook split
-- see methods/_internal_shared/code/pipeline.py's docstring point 4 for why
this is behavior-preserving.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pipeline as pl  # methods/_internal_shared/code/pipeline.py (added to sys.path by adapter.py)


class TCNGRUStageOnly(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), hidden=64, dropout=0.2):
        super().__init__()
        layers, ch = [], input_dim
        for i, out_ch in enumerate(channels):
            layers.append(pl.TemporalBlock(ch, out_ch, 3, 2 ** i, dropout))
            ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.stage_head(self.shared(h[:, -1, :]))


def build_stage_model(input_dim: int, device: str) -> TCNGRUStageOnly:
    return TCNGRUStageOnly(
        input_dim, pl.BEST_ARCH["channels"], pl.BEST_ARCH["gru_hidden"], pl.BEST_ARCH["dropout"]
    ).to(device)


def predict_stage_model(model, pack, device: str):
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for X, _, _, _ in pack["loader"]:
            p = F.softmax(model(X.to(device)), dim=1).detach().cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
    return np.concatenate(preds), np.concatenate(probs)


def train_stage_model(model, tr_pack, va_pack, device: str, epochs: int, patience: int, log_fn=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=pl.BEST_ARCH["lr"], weight_decay=pl.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    w = pl.class_weights(tr_pack["ys"], 3, device)
    best_state, best_score, wait = None, np.inf, 0
    best_info = {"best_epoch": 0, "best_val_acc": np.nan, "best_val_macro_f1": np.nan, "best_val_MRec": np.nan}
    for epoch in range(1, epochs + 1):
        model.train()
        for X, ys, _, _ in tr_pack["loader"]:
            X, ys = X.to(device), ys.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, ys, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), pl.GRAD_CLIP)
            opt.step()
        yv, pv = predict_stage_model(model, va_pack, device)
        m = pl.clf_metrics(va_pack["ys"], yv)
        val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), va_pack["ys"]], 1e-12, 1.0)))
        score = 0.7 * (1 - m["f1"]) + 1.0 * (1 - m["middle_recall"]) + 0.15 * val_loss
        scheduler.step(score)
        if log_fn:
            log_fn(epoch=epoch, val_acc=m["acc"], val_macro_f1=m["f1"], val_middle_recall=m["middle_recall"],
                   val_loss=val_loss, score=score)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            best_info = {"best_epoch": epoch, "best_val_acc": m["acc"], "best_val_macro_f1": m["f1"], "best_val_MRec": m["middle_recall"]}
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_info
