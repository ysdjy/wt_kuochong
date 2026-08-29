# -*- coding: utf-8 -*-
r"""
DP2Net architecture: S (spatial attention), G (physically-constrained
generator), F (WDCNN classifier), and MMD. Reimplemented from
`baselines/dp2net/PAPER_SPEC.md` sec 3-5 (Lai et al., MSSP 2024, DOI:
10.1016/j.ymssp.2024.111421).

Shapes throughout: raw input is [B,1,4608] (Fx only, low-pass filtered,
see preprocessing.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import K_RECEPTIVE, KPOOL, SAMPLE_LEN


# ---------------------------------------------------------------------------
# 3.1 S: discontinuous-physical-property-guided spatial attention
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    """[B,1,4608] -> Wa [B,1,4608] (attention weights in [0,1]), Fa = Wa*input.

    Structure (Fig.4: pooling -> conv -> BN -> ReLU, PAPER_SPEC.md sec 3):
    AvgPool1d(kpool) -> Conv1d(kernel=k, SAME padding) -> BN -> ReLU
    -> Sigmoid gate -> nearest-upsample back to input length.
    Sigmoid is a documented implementation choice (Missing in paper
    whether attention is gated in [0,1]; standard for a "spatial
    attention" gate).
    """

    def __init__(self, kpool: int = KPOOL, k: int = K_RECEPTIVE):
        super().__init__()
        self.kpool = kpool
        self.pool = nn.AvgPool1d(kernel_size=kpool, stride=kpool)
        pad = k // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=pad)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor):
        pooled = self.pool(x)                       # [B,1,L/kpool]
        wa_pooled = torch.sigmoid(self.bn(self.conv(pooled)))
        wa = F.interpolate(wa_pooled, size=x.shape[-1], mode="nearest")
        fa = wa * x
        return fa, wa


# ---------------------------------------------------------------------------
# 3.2 G: generation module (pooling + 3 conv[1,4,4] + AdaIN + transposed conv)
# ---------------------------------------------------------------------------
class AdaIN1d(nn.Module):
    """Adaptive Instance Norm, style vector from random Gaussian noise
    through a small learnable affine (StyleGAN-style noise injection --
    paper cites AdaIN's origin [37]=Karras et al. but gives no
    PHM2010-specific style source, PAPER_SPEC.md sec 4)."""

    def __init__(self, channels: int, style_dim: int = 16):
        super().__init__()
        self.style_dim = style_dim
        self.affine = nn.Linear(style_dim, channels * 2)
        self.inorm = nn.InstanceNorm1d(channels, affine=False)

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None):
        B, C, _ = x.shape
        if noise is None:
            noise = torch.randn(B, self.style_dim, device=x.device, dtype=x.dtype)
        style = self.affine(noise)                    # [B, 2C]
        gamma, beta = style[:, :C], style[:, C:]
        x = self.inorm(x)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class Generator(nn.Module):
    """Fa [B,1,4608] -> Wg [B,1,4608] (in [-1,1], matches Vst's range)."""

    def __init__(self, kpool: int = KPOOL, k: int = K_RECEPTIVE, sample_len: int = SAMPLE_LEN):
        super().__init__()
        pad = k // 2
        self.pool = nn.AvgPool1d(kernel_size=kpool, stride=kpool)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=pad)
        self.bn1 = nn.BatchNorm1d(1)
        self.conv2 = nn.Conv1d(1, 4, kernel_size=k, padding=pad)
        self.bn2 = nn.BatchNorm1d(4)
        self.adain2 = AdaIN1d(4)
        self.conv3 = nn.Conv1d(4, 4, kernel_size=k, padding=pad)
        self.bn3 = nn.BatchNorm1d(4)
        self.adain3 = AdaIN1d(4)
        # ConvTranspose1d sized to exactly invert the AvgPool1d(kpool):
        # out_len = (in_len-1)*kpool - 2*pad + k + output_padding
        pooled_len = sample_len // kpool
        target_len = sample_len
        out_pad = target_len - ((pooled_len - 1) * kpool - 2 * pad + k)
        assert 0 <= out_pad < kpool, f"output_padding {out_pad} out of range for kpool={kpool}"
        self.deconv = nn.ConvTranspose1d(4, 1, kernel_size=k, stride=kpool, padding=pad,
                                          output_padding=out_pad)

    def forward(self, fa: torch.Tensor) -> torch.Tensor:
        x = self.pool(fa)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adain2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adain3(x)
        wg = torch.tanh(self.deconv(x))     # bounded to [-1,1], matches Vst's range
        return wg


# ---------------------------------------------------------------------------
# 3.3 F: WDCNN (Zhang et al. 2018, canonical public architecture --
# "External dependency / inferred component", PAPER_SPEC.md sec 5)
# ---------------------------------------------------------------------------
class WDCNN(nn.Module):
    """Wide-first-layer-kernel deep CNN. Input length adapted from [38]'s
    original 2048/6000-pt vibration window to this task's 4608-pt window
    (first-layer kernel/stride kept at the paper's own 64/16, which scales
    naturally to any reasonable input length; PAPER_SPEC.md sec 5)."""

    def __init__(self, num_classes: int = 3, in_len: int = SAMPLE_LEN):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4, 100), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        emb = self.classifier(feat)          # pre-logit embedding, used for MMD
        logits = self.head(emb)
        if return_features:
            return logits, emb
        return logits


# ---------------------------------------------------------------------------
# MMD (Eq.9), median-heuristic gamma (Missing in paper -> standard default)
# ---------------------------------------------------------------------------
def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    x2 = (x * x).sum(1, keepdim=True)
    y2 = (y * y).sum(1, keepdim=True)
    xy = x @ y.t()
    dist2 = x2 + y2.t() - 2 * xy
    dist2 = dist2.clamp_min(0.0)
    return torch.exp(-dist2 / (2 * gamma ** 2 + 1e-12))


def median_heuristic_gamma(x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        n = z.shape[0]
        if n < 2:
            return 1.0
        d2 = torch.cdist(z, z).pow(2)
        iu = torch.triu_indices(n, n, offset=1)
        vals = d2[iu[0], iu[1]]
        med = vals.median().clamp_min(1e-6)
        return med.sqrt().item()


def mmd_loss(xs: torch.Tensor, xg: torch.Tensor, gamma: float | None = None) -> torch.Tensor:
    """Eq.(9): sqrt(E[k(xS,xS)] - 2E[k(xS,xG)] + E[k(xG,xG)])."""
    if gamma is None:
        gamma = median_heuristic_gamma(xs, xg)
    kss = gaussian_kernel(xs, xs, gamma).mean()
    kgg = gaussian_kernel(xg, xg, gamma).mean()
    ksg = gaussian_kernel(xs, xg, gamma).mean()
    val = (kss - 2 * ksg + kgg).clamp_min(0.0)
    return val.sqrt()


# ---------------------------------------------------------------------------
# Convenience: parameter count helper (S+G+F combined, matches how the
# other baselines in this project report model size)
# ---------------------------------------------------------------------------
class DP2Net(nn.Module):
    """Bundles S+G+F for parameter counting / inference-time convenience.
    Training (the two-stage Algorithm 1 protocol) is implemented in
    train.py, which calls S/G/F directly (not this wrapper) since the
    two stages optimize different module subsets with different losses."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.s = SpatialAttention()
        self.g = Generator()
        self.f = WDCNN(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        """Inference path: S then F only (G is a training-time-only
        diversity generator, not used at test time -- Algorithm 1's
        Inference stage: "Model: Trained S and F.")."""
        fa, _ = self.s(x)
        return self.f(fa)

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
