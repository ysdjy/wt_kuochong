# -*- coding: utf-8 -*-
r"""
Vendored verbatim from methods/B4_HTT_Net/code/model.py (HTT-Net architecture,
itself a verbatim copy of baselines/htt_net/model.py). Self-contained, torch-only.
No modifications -- copied here so run_b4_seed_task.py is git-clone portable
like run_b1/b2_seed_task.py (see vendored_legacy/README.md).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    L = x.shape[1]
    pad = (multiple - L % multiple) % multiple
    if pad == 0:
        return x, L
    last = x[:, -1:, :].expand(-1, pad, -1)
    return torch.cat([x, last], dim=1), L


class RelativePositionBias1D(nn.Module):
    def __init__(self, max_window_size: int, num_heads: int):
        super().__init__()
        self.max_window_size = max_window_size
        self.num_heads = num_heads
        self.bias_table = nn.Parameter(torch.zeros(2 * max_window_size - 1, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, T: int) -> torch.Tensor:
        coords = torch.arange(T, device=self.bias_table.device)
        rel = coords[None, :] - coords[:, None]
        rel_index = rel + (self.max_window_size - 1)
        bias = self.bias_table[rel_index.reshape(-1)]
        bias = bias.reshape(T, T, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()


class WindowAttention1D(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, "embed dim must be divisible by num_heads at every stage"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rel_pos_bias = RelativePositionBias1D(max_window_size=window_size, num_heads=num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Bw, T, C = x.shape
        qkv = self.qkv(x).reshape(Bw, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias(T)[None, :, :, :]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.num_heads, T, T) + mask[None, :, None, :, :]
            attn = attn.view(Bw, self.num_heads, T, T)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(Bw, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    return x.reshape(-1, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, B: int, L: int) -> torch.Tensor:
    x = windows.view(B, L // window_size, window_size, -1)
    return x.reshape(B, L, -1)


def build_shift_mask(L: int, window_size: int, shift_size: int, device) -> torch.Tensor:
    img_mask = torch.zeros(1, L, 1, device=device)
    segments = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    count = 0
    for s in segments:
        img_mask[:, s, :] = count
        count += 1
    mask_windows = window_partition(img_mask, window_size).squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(1, dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def _get_mask(self, L: int, device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        key = (L, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = build_shift_mask(L, self.window_size, self.shift_size, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=-self.shift_size, dims=1)
            mask = self._get_mask(L, x.device)
        else:
            shifted = x
            mask = None

        windows = window_partition(shifted, self.window_size)
        attn_out = self.attn(windows, mask=mask)
        shifted = window_reverse(attn_out, self.window_size, B, L)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=self.shift_size, dims=1)
        else:
            x = shifted

        x = shortcut + self.drop(x)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TokenMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = pad_to_multiple(x, 2)
        B, L, C = x.shape
        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class HTTNetStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        eff_window = min(self.window_size, L)
        for blk in self.blocks:
            blk.window_size = eff_window
            blk.shift_size = 0 if eff_window >= L else blk.shift_size % eff_window
            x = blk(x)
        return x


class HTTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        embed_dim: int = 32,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: int = 4,
        window_size: int = 3,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        dims = [embed_dim * (2 ** i) for i in range(4)]

        self.stages = nn.ModuleList(
            [
                HTTNetStage(dims[i], depths[i], num_heads, window_size, dropout)
                for i in range(4)
            ]
        )
        self.merges = nn.ModuleList([TokenMerging(dims[i]) for i in range(3)])

        self.final_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

        self.apply(_trunc_normal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L0, _ = x.shape
        x = self.embed(x)

        valid_len = L0
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.merges):
                x, valid_len_pre = pad_to_multiple(x, 2)
                valid_len = math.ceil(valid_len / 2)
                x = self.merges[i](x)

        x = self.final_norm(x)
        L_final = x.shape[1]
        valid_len = min(valid_len, L_final)
        idx = torch.arange(L_final, device=x.device).view(1, -1)
        pool_mask = (idx < valid_len).float().unsqueeze(-1)
        pool_mask = pool_mask.expand(B, -1, -1)
        pooled = (x * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
