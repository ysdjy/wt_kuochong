"""Multi-task TCN-GRU backbone (B11) and a sequence-aware monotonic loss.

WHY THE MONOTONIC LOSS IS DIFFERENT FROM THE ORIGINAL CODE
----------------------------------------------------------
The original implementation was

    relu(-(q_hat[1:] - q_hat[:-1])).mean()

evaluated on whatever rows happened to land next to each other in the batch,
with a shuffled DataLoader. Those rows are not temporally adjacent, so the term
did not express monotonic degradation at all. See monotonic_loss_fix.md.

Here every window carries (seq_index, order_key). Batches are drawn as shuffled
CONTIGUOUS BLOCKS, and the monotonic penalty is applied only to pairs that are
genuinely consecutive windows of the same sequence, via an explicit mask. The
block order is still shuffled, so the stochasticity of training is preserved.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from . import config as C


# --------------------------------------------------------------------- data
def build_windows(df, features: list[str], L: int, split_name: str):
    """Sliding windows. A window is emitted only when the L runs it covers are
    consecutive in order_key within one sequence (no bridging over gaps)."""
    import pandas as pd
    Xs, ys, yf, yq, si, ok, meta = [], [], [], [], [], [], []
    for k, (sid, sub) in enumerate(df.sort_values(["sequence_id", "order_key"])
                                   .groupby("sequence_id", sort=True)):
        sub = sub.sort_values("order_key").reset_index(drop=True)
        Xv = sub[features].values.astype(np.float32)
        keys = sub["order_key"].values.astype(int)
        for end in range(L - 1, len(sub)):
            st = end - L + 1
            if not np.all(np.diff(keys[st:end + 1]) == 1):
                continue
            Xs.append(Xv[st:end + 1])
            ys.append(int(sub["stage_id"].iloc[end]))
            yf.append(int(sub["fine_state_true"].iloc[end]))
            yq.append(float(sub["q_true"].iloc[end]))
            si.append(k)
            ok.append(int(keys[end]))
            meta.append(dict(sequence_id=sid, domain_id=sub["domain_id"].iloc[0],
                             order_key=int(keys[end]), VB=float(sub["VB"].iloc[end]),
                             ctime=float(sub["ctime"].iloc[end]) if "ctime" in sub else np.nan,
                             q_true=float(sub["q_true"].iloc[end]),
                             stage_true_id=int(sub["stage_id"].iloc[end]),
                             stage_true=sub["stage"].iloc[end],
                             fine_state_true=int(sub["fine_state_true"].iloc[end]),
                             split=split_name))
    if not Xs:
        raise ValueError(f"{split_name}: no valid windows of length {L}")
    return (np.asarray(Xs, np.float32), np.asarray(ys), np.asarray(yf),
            np.asarray(yq, np.float32), np.asarray(si), np.asarray(ok),
            pd.DataFrame(meta))


class WindowDataset(Dataset):
    def __init__(self, X, ys, yf, yq, si, ok):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.long)
        self.yf = torch.tensor(yf, dtype=torch.long)
        self.yq = torch.tensor(yq, dtype=torch.float32).view(-1, 1)
        self.si = torch.tensor(si, dtype=torch.long)
        self.ok = torch.tensor(ok, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.ys[i], self.yf[i], self.yq[i], self.si[i], self.ok[i]


class TemporalBlockSampler(Sampler[list[int]]):
    """Shuffled contiguous blocks, so adjacent rows in a batch are adjacent in
    time. Assumes the dataset is ordered by (sequence, order_key)."""

    def __init__(self, seq_index: np.ndarray, batch_size: int, seed: int = 42,
                 shuffle: bool = True):
        self.batches: list[list[int]] = []
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        idx = np.arange(len(seq_index))
        for s in np.unique(seq_index):
            ids = idx[seq_index == s]
            for i in range(0, len(ids), batch_size):
                self.batches.append(ids[i:i + batch_size].tolist())

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


def make_loader(pack, batch_size=C.BATCH_SIZE, seed=42, shuffle=True):
    ds = WindowDataset(pack["X"], pack["ys"], pack["yf"], pack["yq"],
                       pack["seq_index"], pack["order_key"])
    sampler = TemporalBlockSampler(pack["seq_index"], batch_size, seed, shuffle)
    return DataLoader(ds, batch_sampler=sampler), sampler


# -------------------------------------------------------------------- model
class Chomp1d(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return x if self.n == 0 else x[:, :, :-self.n]


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout))
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        y = self.net(x)
        r = x if self.down is None else self.down(x)
        m = min(y.size(-1), r.size(-1))
        return F.relu(y[:, :, :m] + r[:, :, :m])


class TCNGRUMultiTask(nn.Module):
    """Three heads: macro stage, ordered fine state, relative degradation position."""

    def __init__(self, input_dim, channels=C.ARCH["tcn_channels"],
                 gru_hidden=C.ARCH["gru_hidden"], dropout=C.ARCH["dropout"],
                 n_fine=C.N_FINE_STATES):
        super().__init__()
        layers, ch = [], input_dim
        for i, co in enumerate(channels):
            layers.append(TemporalBlock(ch, co, 3, 2 ** i, dropout)); ch = co
        self.tcn = nn.Sequential(*layers)
        self.gru = nn.GRU(channels[-1], gru_hidden, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(gru_hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(64, 3)
        self.fine_head = nn.Linear(64, n_fine)
        self.q_head = nn.Linear(64, 1)

    def forward(self, x, return_hidden=False):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        z = self.shared(h[:, -1, :])
        out = dict(stage_logits=self.stage_head(z), fine_logits=self.fine_head(z),
                   q_hat=torch.sigmoid(self.q_head(z)))
        out["stage_prob"] = F.softmax(out["stage_logits"], 1)
        out["fine_prob"] = F.softmax(out["fine_logits"], 1)
        if return_hidden:
            out["h"] = z
        return out


class StageOnlyNet(nn.Module):
    """Single-task deep baselines: B8 TCN, B9 GRU, B10 TCN-GRU."""

    def __init__(self, input_dim, kind="tcngru", channels=C.ARCH["tcn_channels"],
                 gru_hidden=C.ARCH["gru_hidden"], dropout=C.ARCH["dropout"], L=C.ARCH["L"]):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(input_dim * L, 128), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(64, 3))
        if kind in ("tcn", "tcngru"):
            layers, ch = [], input_dim
            for i, co in enumerate(channels):
                layers.append(TemporalBlock(ch, co, 3, 2 ** i, dropout)); ch = co
            self.tcn = nn.Sequential(*layers)
        if kind in ("gru", "tcngru"):
            self.gru = nn.GRU(channels[-1] if kind == "tcngru" else input_dim,
                              gru_hidden, batch_first=True)
        feat = gru_hidden if kind in ("gru", "tcngru") else channels[-1]
        self.head = nn.Sequential(nn.Linear(feat, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 3))

    def forward(self, x):
        if self.kind == "mlp":
            return dict(stage_logits=self.mlp(x))
        h = x
        if self.kind in ("tcn", "tcngru"):
            h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        if self.kind in ("gru", "tcngru"):
            h, _ = self.gru(h)
        return dict(stage_logits=self.head(h[:, -1, :]))


# --------------------------------------------------------------------- loss
def sequence_aware_monotonic_loss(q_hat, seq_index, order_key, return_stats=False):
    """Hinge on q_hat[t] > q_hat[t+1] for GENUINELY consecutive windows only.

    Returns (loss, stats) when return_stats=True, where stats carries
    n_valid_pairs and violation_rate for the diagnostics required by the
    experiment protocol.
    """
    q = q_hat.view(-1)
    zero = q.sum() * 0.0
    if q.numel() < 2:
        return (zero, dict(n_valid_pairs=0, n_violations=0, violation_rate=0.0)) \
            if return_stats else zero
    same = seq_index[1:] == seq_index[:-1]
    adjacent = (order_key[1:] - order_key[:-1]) == 1
    mask = (same & adjacent)
    n_valid = int(mask.sum())
    if n_valid < 1:
        return (zero, dict(n_valid_pairs=0, n_violations=0, violation_rate=0.0)) \
            if return_stats else zero
    mf = mask.float()
    diff = q[1:] - q[:-1]
    viol = torch.relu(-diff) * mf
    loss = viol.sum() / mf.sum()
    if not return_stats:
        return loss
    n_v = int(((diff < 0) & mask).sum())
    return loss, dict(n_valid_pairs=n_valid, n_violations=n_v,
                      violation_rate=n_v / max(n_valid, 1))


def class_weights(y: np.ndarray, n: int, device) -> torch.Tensor:
    cnt = np.bincount(y, minlength=n).astype(float)
    w = cnt.sum() / (n * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
