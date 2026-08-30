# -*- coding: utf-8 -*-
r"""
Dynamic GIN + TGP architecture, reimplemented from Table 1 / Eq.(1)-(17) of
`PAPER_SPEC.md` (Cao et al., Measurement 2026, DOI:
10.1016/j.measurement.2025.119007). Every shape below is checked against
Table 1's own stated I/O sizes (batch=4 reference) in
`tests/test_shapes.py`.

Additional (minor) implementation choices made here, beyond what
PAPER_SPEC.md already tables, are documented inline where they occur:

  - Cross-attention score normalization (Eq.5): the paper says attention
    scores are "computed... which are then used to weight and filter",
    without stating whether a softmax normalizes them. We apply softmax
    over the flattened spatial axis -- standard attention practice, and
    numerically necessary given the very large flattened spatial
    dimension (~80k) that an un-normalized dot product would otherwise
    produce. [Missing in paper -> softmax, documented here]
  - TGP feature pooling (Eq.14, "DimTran"/Conv2d): implemented as a
    Conv2d whose in/out-channels are the pre/post node counts (this *is*
    the pooling / node-count reduction), operating with a temporal kernel
    (width = paper's Ks) SAME-padded so the 288-length time axis is
    preserved exactly, consistent with (a) Table 1 never changing the
    "288" column across TGP layers, and (b) the paper's own description
    "the method utilizes the temporal convolutions to cluster nodes".
    [Missing exact axis bookkeeping in paper -> this reading, documented
    in PAPER_SPEC.md sec 6]
  - TGP adjacency pooling S_p (Eq.15-16): learned directly as a single
    [N_pos, N_pre] parameter (a strict superset of the paper's W_p . Z_p
    factorization, whose W_p is Missing in the paper).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 6     # Fx,Fy,Fz,Vx,Vy,Vz
N_TIME = 288        # Table 1
N_NODES = 24         # Nsf, temporal feature dimension after Conv2d_1/2
EPS_GIN = 0.5         # Table 1, all 3 GIN layers


# ---------------------------------------------------------------------------
# 2.2.1 Temporal feature extraction (Eq.1, Table 1 rows Conv2d_1/Conv2d_2)
# ---------------------------------------------------------------------------
class TemporalFeatureExtraction(nn.Module):
    """[B,1,6,288] -> Xsf3 [B,1,24,288].

    Conv2d_1 uses a full-channel-height kernel (6,9) so a single conv
    collapses the 6-channel axis to 1 while doing a SAME conv on the time
    axis -- the only kernel choice consistent with Table 1's own stated
    [4,1,6,288]->[4,14,1,288] I/O shapes (PAPER_SPEC.md Conflict #1).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 14, kernel_size=(N_CHANNELS, 9), padding=(0, 4))
        self.bn1 = nn.BatchNorm2d(14)
        self.conv2 = nn.Conv2d(14, N_NODES, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(N_NODES)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))          # [B,14,1,288]
        x = F.relu(self.bn2(self.conv2(x)))           # [B,24,1,288]
        x = self.dropout(x)
        x = x.transpose(1, 2)                          # [B,1,24,288]  (Table 1's stated Transpose(1,2))
        return x


# ---------------------------------------------------------------------------
# 2.2.2 Spatial feature extraction: GASF (Eq.2-4) + CNN (Table 1 Conv2d_3/4)
# ---------------------------------------------------------------------------
class GASFEncoder(nn.Module):
    """Xsf3 [B,1,24,288] -> GASF images [B,24,288,288].

    GASF(i,j) = cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
    with cos(phi) = x~ (the [-1,1]-normalized signal itself, since
    phi=arccos(x~)) and sin(phi) = sqrt(1-x~^2) (phi in [0,pi] so sin>=0).
    Vectorized outer-product form, numerically identical to the
    arccos/cos formula in Eq.(2)-(4) but avoids redundant
    arccos-then-cos round trips.
    """

    def forward(self, xsf3):
        x = xsf3.squeeze(1)  # [B,24,288]
        xmin = x.min(dim=-1, keepdim=True).values
        xmax = x.max(dim=-1, keepdim=True).values
        denom = (xmax - xmin).clamp_min(1e-8)
        x_norm = (2 * x - xmax - xmin) / denom          # Eq.(2), in [-1,1]
        x_norm = x_norm.clamp(-1.0, 1.0)
        cos_phi = x_norm
        # clamp_min(1e-8), NOT 0.0: by construction x_norm hits exactly
        # +-1 at each feature's own min/max point (every row, every
        # sample), so 1-x_norm^2 is exactly 0 there; sqrt's gradient is
        # infinite at 0, which produced NaN gradients on the very first
        # backward pass (caught by tests/test_pipeline.py's single-batch
        # overfit test). The epsilon floor keeps sqrt in its
        # well-behaved region without perceptibly changing GASF's value
        # (sqrt(1e-8)=1e-4, negligible vs the [-1,1] value range).
        sin_phi = (1.0 - x_norm.pow(2)).clamp_min(1e-8).sqrt()
        gasf = (cos_phi.unsqueeze(-1) * cos_phi.unsqueeze(-2)
                - sin_phi.unsqueeze(-1) * sin_phi.unsqueeze(-2))  # [B,24,288,288]
        return gasf


class SpatialFeatureExtraction(nn.Module):
    """GASF [B,24,288,288] -> Xsp1 [B,288,H',W'] (valid convs, see
    PAPER_SPEC.md Conflict #2 for the ~2px discrepancy vs Table 1's
    printed 285/284 -- does not affect trainability)."""

    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(N_NODES, 64, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, N_TIME, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(N_TIME)
        self.dropout = nn.Dropout(0.5)

    def forward(self, gasf):
        x = F.relu(self.bn3(self.conv3(gasf)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        return x  # [B,288,H',W']


class CrossAttentionFusion(nn.Module):
    """Eq.(5)-(6): Xsf3, Xsp1 -> XF [B,1,24,288]."""

    def forward(self, xsf3, xsp1):
        B = xsf3.shape[0]
        xsf = xsf3.squeeze(1)                                   # [B,24,288]
        C, H, W = xsp1.shape[1:]
        xsp_flat = xsp1.reshape(B, C, H * W)                     # [B,288,H'W']
        alpha = torch.bmm(xsf, xsp_flat)                          # [B,24,H'W']
        alpha = F.softmax(alpha, dim=-1)                           # [Missing in paper -> softmax, see module docstring]
        xsp2 = torch.bmm(alpha, xsp_flat.transpose(1, 2))          # [B,24,288]
        xf = xsf + xsp2                                             # Eq.(6)
        return xf.unsqueeze(1)                                       # [B,1,24,288]


# ---------------------------------------------------------------------------
# 2.3 Static / dynamic graph generation and fusion (Eq.7-12)
# ---------------------------------------------------------------------------
class GraphEmbeddingMLP(nn.Module):
    """Eq.(7): Xsf3 [B,1,24,288] -> Xg [B,24,64].

    NOTE (verified via tests/test_gradients.py): this module's 4 params
    receive NO gradient from the classification loss. This is an inherent
    consequence of a literal Eq.(12) implementation, not a bug here: the
    adjacency built from Xg is hard-thresholded (top-k values replaced by
    the constant 1.0, `A_g(~index)=0`), which is a non-differentiable
    step-function w.r.t. Xg. The paper gives no straight-through/Gumbel
    relaxation, so a literal reproduction of Eq.(12) has this property --
    Xg's embedding MLP is effectively frozen at its random init throughout
    training. Documented in FINAL_REPORT.md as a paper-inherent property,
    not "fixed" here (doing so would deviate from the paper's own stated
    hard top-k mechanism, task instruction #80)."""

    def __init__(self, in_dim=N_TIME, hid=256, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, xsf3):
        x = xsf3.squeeze(1)   # [B,24,288]
        return self.net(x)    # [B,24,64]


def build_static_graph(xg: torch.Tensor) -> torch.Tensor:
    """Eq.(8)-(9): concatenate all samples in the batch along the
    embedding dim before cosine similarity -> one shared [24,24] adjacency."""
    B, N, D = xg.shape
    flat = xg.permute(1, 0, 2).reshape(N, B * D)   # [24, B*64]
    flat = F.normalize(flat, dim=1, eps=1e-8)
    return flat @ flat.t()                          # [24,24]


def build_dynamic_graph(xg: torch.Tensor) -> torch.Tensor:
    """Eq.(10)-(11): per-sample cosine similarity, no batch concatenation."""
    xg_n = F.normalize(xg, dim=2, eps=1e-8)
    return torch.bmm(xg_n, xg_n.transpose(1, 2))      # [B,24,24]


def fuse_and_sparsify(a_static: torch.Tensor, a_dynamic: torch.Tensor, topk: int) -> torch.Tensor:
    """Eq.(12): A = A_static + A_dynamic, keep global top-k entries per
    sample (flattened NxN), symmetrize. See PAPER_SPEC.md Conflict #3 for
    the topk=144 (chosen) vs 288 (Table 1) resolution."""
    B = a_dynamic.shape[0]
    a = a_static.unsqueeze(0) + a_dynamic             # [B,24,24]
    n2 = a.shape[1] * a.shape[2]
    k = min(topk, n2)
    flat = a.reshape(B, -1)
    _, idx = torch.topk(flat, k, dim=1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    mask = mask.reshape(B, a.shape[1], a.shape[2])
    mask = torch.maximum(mask, mask.transpose(1, 2))   # keep undirected
    return mask


# ---------------------------------------------------------------------------
# 2.4 GIN + TGP (Eq.13-16, Table 1)
# ---------------------------------------------------------------------------
class GINLayer(nn.Module):
    """Eq.(13): H^l = MLP((1+eps)*H^(l-1) + D^-1/2 A D^-1/2 H^(l-1)).
    MLP realized as a 1x1 Conv2d over the (channel) axis, per Table 1's
    `Ks=1` framing for all 3 GIN layers."""

    def __init__(self, in_ch: int, out_ch: int, eps: float = EPS_GIN):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # h: [B,C,N,T], a: [B,N,N]
        B, C, N, T = h.shape
        deg = a.sum(dim=-1).clamp_min(1e-8)              # [B,N]
        dinv = deg.pow(-0.5)
        a_norm = a * dinv.unsqueeze(-1) * dinv.unsqueeze(-2)  # symmetric norm
        h_flat = h.permute(0, 2, 1, 3).reshape(B, N, C * T)     # [B,N,C*T]
        agg = torch.bmm(a_norm, h_flat).reshape(B, N, C, T).permute(0, 2, 1, 3)  # [B,C,N,T]
        h_new = (1.0 + self.eps) * h + agg
        h_new = self.mlp(h_new)
        h_new = F.relu(self.bn(h_new))
        return h_new


class TGPLayer(nn.Module):
    """Eq.(14)-(16): jointly pools the feature matrix (node axis, via a
    temporal-kernel conv realizing the node-count reduction through its
    in/out channel dims) and the adjacency matrix (learned assignment
    S_p, A_next = S_p A S_p^T)."""

    def __init__(self, n_pre: int, n_pos: int, kernel_w: int, channels: int, dropout: float = 0.3):
        super().__init__()
        pad_total = kernel_w - 1
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left
        self.conv = nn.Conv2d(n_pre, n_pos, kernel_size=(1, kernel_w))
        self.bn = nn.BatchNorm2d(n_pos)
        self.dropout = nn.Dropout(dropout)
        self.s_p = nn.Parameter(torch.randn(n_pos, n_pre) * (1.0 / n_pre ** 0.5))

    def forward(self, h: torch.Tensor, a: torch.Tensor):
        # h: [B,C,N_pre,T]
        hp = h.permute(0, 2, 1, 3)                              # [B,N_pre,C,T]
        hp = F.pad(hp, (self.pad_left, self.pad_right))
        hp = self.conv(hp)                                       # [B,N_pos,C,T]
        hp = F.relu(self.bn(hp))
        hp = self.dropout(hp)
        h_new = hp.permute(0, 2, 1, 3)                            # [B,C,N_pos,T]

        a_next = torch.einsum('pn,bnm,qm->bpq', self.s_p, a, self.s_p)  # Eq.(16)
        return h_new, a_next


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class DynamicGIN_TGP(nn.Module):
    def __init__(self, topk: int = 144):
        super().__init__()
        self.topk = topk
        self.temporal = TemporalFeatureExtraction()
        self.gasf = GASFEncoder()
        self.spatial = SpatialFeatureExtraction()
        self.cross_attn = CrossAttentionFusion()
        self.graph_mlp = GraphEmbeddingMLP()

        self.gin1 = GINLayer(1, 32)
        self.tgp1 = TGPLayer(n_pre=N_NODES, n_pos=19, kernel_w=18, channels=32)
        self.gin2 = GINLayer(32, 64)
        self.tgp2 = TGPLayer(n_pre=19, n_pos=14, kernel_w=9, channels=64)
        self.gin3 = GINLayer(64, 128)
        self.tgp3 = TGPLayer(n_pre=14, n_pos=10, kernel_w=5, channels=128)

        self.out_pool = nn.AdaptiveAvgPool2d(1)
        self.out_fc = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """x: [B,6,288] raw force+vibration window."""
        x = x.unsqueeze(1)                        # [B,1,6,288]
        xsf3 = self.temporal(x)                     # [B,1,24,288]
        gasf = self.gasf(xsf3)                        # [B,24,288,288]
        xsp1 = self.spatial(gasf)                       # [B,288,H',W']
        xf = self.cross_attn(xsf3, xsp1)                  # [B,1,24,288]

        xg = self.graph_mlp(xsf3)                           # [B,24,64], per Eq.7 uses Xsf3
        a_static = build_static_graph(xg)                     # [24,24]
        a_dynamic = build_dynamic_graph(xg)                     # [B,24,24]
        a = fuse_and_sparsify(a_static, a_dynamic, self.topk)    # [B,24,24]

        h = xf                                                    # [B,1,24,288]
        h = self.gin1(h, a)
        h, a = self.tgp1(h, a)
        h = self.gin2(h, a)
        h, a = self.tgp2(h, a)
        h = self.gin3(h, a)
        h, a = self.tgp3(h, a)                                       # [B,128,10,288]

        pooled = self.out_pool(h).flatten(1)                            # [B,128]
        logits = self.out_fc(pooled)                                      # [B,3]

        if return_intermediates:
            return logits, {"xsf3": xsf3, "gasf": gasf, "xsp1": xsp1, "xf": xf,
                             "xg": xg, "a_static": a_static, "a_dynamic": a_dynamic}
        return logits

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
