# -*- coding: utf-8 -*-
r"""
Multi-source Attention CNN ("Multi-Attention-CNN"), reimplemented from
`baselines/multi_source_attention/PAPER_SPEC.md` (Wei et al., RCIM 2024).

Architecture (PAPER_SPEC.md §2, literal Table 2):

    Force branch:  CWT image [B,3,224,224]  -> Conv2D(16,k3,s1,same,ReLU) -> [B,16,224,224]
    Vibration branch: CWT image [B,3,224,224] -> Conv2D(16,k3,s1,same,ReLU) -> [B,16,224,224]
      -> Concatenate (channel dim)                      -> [B,32,224,224]   (Layer1+2)
      -> Channel-Spatial Attention (Eqs 1-18)            -> [B,32,224,224]   (only insertion point)
      -> MaxPool(k3,s2,same)                             -> [B,32,112,112]   (Layer3)
      -> Conv2D(64,k3,s1,same,ReLU)                      -> [B,64,112,112]   (Layer4)
      -> MaxPool(k3,s2,same)                             -> [B,64,56,56]     (Layer5)
      -> Conv2D(128,k3,s1,same,ReLU)                     -> [B,128,56,56]    (Layer6)
      -> MaxPool(k3,s2,same)                             -> [B,128,28,28]    (Layer7)
      -> Flatten -> FC(128)                              -> [B,128]          (Layer8)
      -> Dropout(0.5)                                    -> [B,128]          (Layer9)
      -> FC(3)                                           -> [B,3]            (Layer10, logits)
      -> Softmax (applied externally / via loss)                            (Layer11)

Channel attention: GAP -> FC1(reduce by r) -> ReLU -> FC2(restore) -> Sigmoid
                    (Eqs 1-8; r Missing in paper -> r=16, SE-Net default).
Spatial attention: 1x1 conv channel-compression -> {AvgPool,MaxPool over
                    channel dim} -> concat -> 1x1 conv -> Sigmoid (Eqs 9-15;
                    compression uses the same r=16 for consistency, since
                    the paper gives no separate ratio for this branch).
Combination: X~ = X * s * Ms, BOTH s (channel) and Ms (spatial) computed
             from the SAME pre-attention input X (Eq. 18, parallel, not a
             cascaded/sequential-recompute CBAM) -- see PAPER_SPEC.md
             "Channel+spatial combination order".

The network returns raw logits (not softmax'd) from forward(), matching
standard PyTorch practice for training with nn.CrossEntropyLoss (which
applies log-softmax internally); an explicit softmax is applied only when
producing probability outputs (p_early/p_normal/p_severe), functionally
equivalent to the paper's own Layer 11 Softmax without double-applying it
during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 224  # Explicit (paper §2.4)
REDUCTION_RATIO = 16  # Missing in paper -> SE-Net default (PAPER_SPEC.md "Channel attention reduction ratio r")


class ChannelAttention(nn.Module):
    """Eqs. 1-8: GAP -> FC1 -> ReLU -> FC2 -> Sigmoid -> channel weight s."""

    def __init__(self, channels: int, r: int = REDUCTION_RATIO):
        super().__init__()
        hidden = max(1, channels // r)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return y.view(b, c, 1, 1)  # s


class SpatialAttention(nn.Module):
    """Eqs. 9-15: 1x1 conv channel compression -> {AvgPool,MaxPool} over
    channel dim -> concat -> 1x1 conv -> Sigmoid -> spatial weight Ms."""

    def __init__(self, channels: int, r: int = REDUCTION_RATIO):
        super().__init__()
        compressed = max(1, channels // r)
        self.compress = nn.Conv2d(channels, compressed, kernel_size=1)
        self.merge = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.compress(x)  # [B, C/r, H, W]
        avg_map = y.mean(dim=1, keepdim=True)  # [B,1,H,W]
        max_map = y.max(dim=1, keepdim=True).values  # [B,1,H,W]
        cat = torch.cat([avg_map, max_map], dim=1)  # [B,2,H,W]
        Ms = self.sigmoid(self.merge(cat))  # [B,1,H,W]
        return Ms


class ChannelSpatialAttention(nn.Module):
    """Eq. 18: X~ = X * s * Ms, both branches computed from the same X
    (parallel combination, not a cascaded CBAM)."""

    def __init__(self, channels: int, r: int = REDUCTION_RATIO):
        super().__init__()
        self.channel_att = ChannelAttention(channels, r)
        self.spatial_att = SpatialAttention(channels, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.channel_att(x)
        Ms = self.spatial_att(x)
        return x * s * Ms


class MultiAttentionCNN(nn.Module):
    def __init__(self, num_classes: int = 3, r: int = REDUCTION_RATIO, dropout: float = 0.5,
                 image_size: int = IMAGE_SIZE):
        super().__init__()
        self.force_conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.vib_conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.attn = ChannelSpatialAttention(32, r)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 224 -> 112
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112 -> 56
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56 -> 28

        feat_hw = image_size // 8  # 224 -> 28 after 3x stride-2 pools
        self.fc1 = nn.Linear(128 * feat_hw * feat_hw, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, force_img: torch.Tensor, vib_img: torch.Tensor) -> torch.Tensor:
        f = self.force_conv(force_img)   # [B,16,224,224]
        v = self.vib_conv(vib_img)       # [B,16,224,224]
        x = torch.cat([f, v], dim=1)     # [B,32,224,224]
        x = self.attn(x)                 # only attention insertion point
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)                  # Layer8, no activation stated in Table 2
        x = self.dropout(x)              # Layer9
        x = self.fc2(x)                  # Layer10, logits (Layer11 Softmax applied externally)
        return x

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


if __name__ == "__main__":
    m = MultiAttentionCNN()
    f = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    v = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = m(f, v)
    print("output shape:", y.shape)
    print("params:", m.num_parameters())
    print("softmax sums:", F.softmax(y, dim=1).sum(dim=1))
