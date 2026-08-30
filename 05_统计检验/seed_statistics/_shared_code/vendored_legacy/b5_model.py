# -*- coding: utf-8 -*-
r"""
Vendored verbatim from methods/B5_MultiSource_Attention/code/model.py.
Multi-source Channel-Spatial Attention CNN ("Multi-Attention-CNN"). No
modifications -- torch-only, self-contained.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGE_SIZE = 224
REDUCTION_RATIO = 16


class ChannelAttention(nn.Module):
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
        return y.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, r: int = REDUCTION_RATIO):
        super().__init__()
        compressed = max(1, channels // r)
        self.compress = nn.Conv2d(channels, compressed, kernel_size=1)
        self.merge = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.compress(x)
        avg_map = y.mean(dim=1, keepdim=True)
        max_map = y.max(dim=1, keepdim=True).values
        cat = torch.cat([avg_map, max_map], dim=1)
        return self.sigmoid(self.merge(cat))


class ChannelSpatialAttention(nn.Module):
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

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        feat_hw = image_size // 8
        self.fc1 = nn.Linear(128 * feat_hw * feat_hw, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, force_img: torch.Tensor, vib_img: torch.Tensor) -> torch.Tensor:
        f = self.force_conv(force_img)
        v = self.vib_conv(vib_img)
        x = torch.cat([f, v], dim=1)
        x = self.attn(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
