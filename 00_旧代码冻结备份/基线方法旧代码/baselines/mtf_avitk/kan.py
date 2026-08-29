# -*- coding: utf-8 -*-
r"""
Minimal from-scratch Kolmogorov-Arnold Network (KAN) linear layer.

Implemented from the original KAN formulation (Liu et al., arXiv:2404.19756,
cited by the MTF-AViTK paper as ref. [57] with no implementation/library
name given -- see PAPER_SPEC.md, "KAN implementation source"), following
the widely-used "efficient-kan" B-spline reformulation (each edge activation
= scale_base * base_activation(x) [linear-like residual branch] +
scale_spline * (learnable B-spline curve over a fixed grid)), which is
mathematically equivalent to the original KAN paper's spline-based edge
functions but avoids the original's slower recursive spline evaluation.

This file has no dependency on any third-party KAN package (pykan,
efficient-kan, etc.) -- everything below is implemented directly against
PyTorch tensor ops, per the task instruction to avoid installing unvetted
third-party libraries for this component.

Hyperparameters used here (grid_size=5, spline_order=3, base_activation=
SiLU, scale_base=1.0, scale_spline=1.0) come directly from the paper's own
Table 1 (Explicit) -- see PAPER_SPEC.md "KAN hyperparameters" row. The
per-layer hidden width (Missing in paper) is a documented implementation
choice, set by the caller (see model.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,       # Explicit: paper Table 1, "grid intervals G = 5"
        spline_order: int = 3,    # Explicit: paper Table 1, "polynomial order k = 3"
        scale_base: float = 1.0,  # Explicit: paper Table 1, "scale of ... base spline(x) = 1.0"
        scale_spline: float = 1.0,  # Explicit: paper Table 1, "scale of residual function b(x) = 1.0"
        base_activation: type[nn.Module] = nn.SiLU,  # Explicit: paper Table 1, "residual function b(x) = SiLU(x)"
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        grid = grid.expand(in_features, -1).contiguous()  # [in_features, grid_size + 2*spline_order + 1]
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * 0.1
                / self.grid_size
            )
            coeff = self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            self.spline_weight.data.copy_(coeff * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, in_features] -> [batch, in_features, grid_size + spline_order] B-spline bases."""
        grid = self.grid  # [in_features, grid_pts]
        x = x.unsqueeze(-1)  # [batch, in_features, 1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # order-0
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-12)
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k] + 1e-12)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases  # [batch, in_features, grid_size + spline_order]

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares fit of spline coefficients to sampled points, used only at init.
        x: [n_pts, in_features], y: [n_pts, in_features, out_features]."""
        A = self.b_splines(x).transpose(0, 1)  # [in_features, n_pts, coeff]
        B = y.transpose(0, 1)  # [in_features, n_pts, out_features]
        sol = torch.linalg.lstsq(A, B).solution  # [in_features, coeff, out_features]
        return sol.permute(2, 0, 1)  # [out_features, in_features, coeff]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        bases = self.b_splines(x).view(x.size(0), -1)  # [batch, in_features*coeff]
        spline_output = F.linear(bases, self.spline_weight.view(self.out_features, -1))
        out = self.scale_base * base_output + spline_output
        return out.reshape(*orig_shape[:-1], self.out_features)


class KANClassifier(nn.Module):
    """2-layer KAN, matching PAPER_SPEC.md "KAN classifier: structure":
    [CLS] token (1024) -> KANLinear -> hidden -> KANLinear -> 3 classes."""

    def __init__(self, in_features: int = 1024, hidden: int = 64, num_classes: int = 3, **kan_kwargs):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden, **kan_kwargs)
        self.layer2 = KANLinear(hidden, num_classes, **kan_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x
