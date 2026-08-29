# -*- coding: utf-8 -*-
r"""
Adapt-ViT_L/32 + AdaptMLP + KAN ("MTF-AViTK"), reimplemented from
`baselines/mtf_avitk/PAPER_SPEC.md` (Dong et al., MSSP 2025).

Architecture (PAPER_SPEC.md §2):

    [384,384,3] image
      -> Conv2D patch embed (32x32, stride 32) -> [12,12,1024] -> flatten [144,1024]   Eq. 2  Explicit
      -> + class token, + position embedding -> [145,1024]                             Eq. 3  Explicit
      -> 24x { LayerNorm -> MHSA(16 heads) -> +residual
               -> LayerNorm -> AdaptMLP(MLP + adapter, d_hat=64, s=0.1) -> +residual }  Eq. 4-6 Explicit topology,
                                                                                                Missing d_hat/s
      -> LayerNorm -> [class] token [1,1024]
      -> 2-layer KAN (G=5, k=3, SiLU, scale=1.0, hidden=64) -> 3 classes                Table 1

See PAPER_SPEC.md for every Explicit/Inferable/Missing determination and
the reasoning behind each Missing-value choice (bottleneck dim, adapter
scale, KAN hidden width, GELU exact-vs-tanh form, pretrained-weight
status). No pretrained-weight loading is performed anywhere in this file
(paper never states a pretraining source -- trained from scratch, see
PAPER_SPEC.md "ViT pretrained weights").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint

from kan import KANClassifier

# Explicit (PAPER_SPEC.md "Adapt-ViT_L/32" rows)
PATCH_SIZE = 32
IMG_SIZE = 384
EMBED_DIM = 1024
DEPTH = 24
NUM_HEADS = 16
MLP_HIDDEN = 4096
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 144

# Missing in paper -> AdaptFormer's own defaults (PAPER_SPEC.md "AdaptMLP structure")
ADAPTER_BOTTLENECK = 64
ADAPTER_SCALE = 0.1

# Missing in paper -> implementation choice (PAPER_SPEC.md "KAN classifier: hidden width")
KAN_HIDDEN = 64


class PatchEmbed(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        return x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class AdaptMLP(nn.Module):
    """Eq. 6: Sl = MLP(LayerNorm(S'')) + s * [ReLU(LayerNorm(S'') . Wdown) . Wup].

    Both the original MLP branch and the adapter branch consume the SAME
    LayerNorm(S'') input (one shared pre-norm, per Eq. 6's single
    `LayerNorm(S'')` term feeding both branches, and Fig. 3's parallel
    MLP-block/adapter-block diagram off one normalized input). The block
    wrapping this module (see TransformerBlock) does not apply a second
    LayerNorm.
    """

    def __init__(self, dim=EMBED_DIM, mlp_hidden=MLP_HIDDEN, bottleneck=ADAPTER_BOTTLENECK,
                 scale=ADAPTER_SCALE, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = scale
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        mlp_out = self.mlp(normed)
        adapter_out = self.up(self.act(self.down(normed)))
        return mlp_out + self.scale * adapter_out


class TransformerBlock(nn.Module):
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_hidden=MLP_HIDDEN,
                 bottleneck=ADAPTER_BOTTLENECK, adapter_scale=ADAPTER_SCALE, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, proj_dropout=dropout)
        self.adapt_mlp = AdaptMLP(dim, mlp_hidden, bottleneck, adapter_scale, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.adapt_mlp(x)
        return x


class AdaptViT(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_hidden=MLP_HIDDEN, bottleneck=ADAPTER_BOTTLENECK,
                 adapter_scale=ADAPTER_SCALE, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_hidden, bottleneck, adapter_scale, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Not part of the paper's architecture -- a pure memory/compute
        # trade-off for training ViT-L/32 on an 8GB card (task instruction
        # #68: memory adaptations allowed, must not change the model
        # itself, and must be documented). Off by default.
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token, [B, C]


class MTF_AViTK(nn.Module):
    """Full model: Adapt-ViT_L/32 backbone + 2-layer KAN classification head."""

    def __init__(self, num_classes=3, vit_kwargs: dict | None = None, kan_kwargs: dict | None = None):
        super().__init__()
        vit_kwargs = vit_kwargs or {}
        self.backbone = AdaptViT(**vit_kwargs)
        embed_dim = vit_kwargs.get("embed_dim", EMBED_DIM)
        self.classifier = KANClassifier(in_features=embed_dim, hidden=KAN_HIDDEN,
                                         num_classes=num_classes, **(kan_kwargs or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(x)
        return self.classifier(cls)

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


if __name__ == "__main__":
    m = MTF_AViTK()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = m(x)
    print("output shape:", y.shape)
    print("params:", m.num_parameters())
