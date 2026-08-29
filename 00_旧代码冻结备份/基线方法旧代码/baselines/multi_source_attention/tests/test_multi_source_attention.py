# -*- coding: utf-8 -*-
r"""
Unit tests for the Multi-source Attention (Multi-Attention-CNN) baseline
(preprocessing.py + model.py).

Run with:
    python tests/test_multi_source_attention.py
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocessing as P
from model import (
    MultiAttentionCNN, ChannelAttention, SpatialAttention, ChannelSpatialAttention, IMAGE_SIZE,
)


class TestPreprocessing(unittest.TestCase):
    def test_get_middle_region_length(self):
        sig = np.arange(1000 * 7).reshape(1000, 7).astype(float)
        mid = P.get_middle_region(sig)
        self.assertEqual(mid.shape[0], 500)  # central 50%
        self.assertEqual(mid.shape[1], 7)

    def test_cwt_scalogram_shape_and_range(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        mag = P.cwt_scalogram(x, image_size=64, n_scales=64)
        self.assertEqual(mag.shape, (64, 64))
        self.assertTrue(np.all(mag >= -1e-9) and np.all(mag <= 1 + 1e-9))
        self.assertTrue(np.all(np.isfinite(mag)))

    def test_axes_to_rgb_image_shape(self):
        rng = np.random.default_rng(1)
        sig3 = rng.normal(size=(2000, 3))
        img = P.axes_to_rgb_image(sig3, image_size=64)
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_build_sample_real_data(self):
        force_img, vib_img = P.build_sample("C1", 1)
        self.assertEqual(force_img.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(vib_img.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(force_img.dtype, np.uint8)

    def test_original_stage_ranges_cover_1_to_315_no_gap_no_overlap(self):
        for cond in ["C1", "C4", "C6"]:
            covered = set()
            for stage, (lo, hi) in P.ORIGINAL_STAGE_RANGES[cond].items():
                rng_ids = set(range(lo, hi + 1))
                self.assertFalse(covered & rng_ids, f"{cond} stage ranges overlap")
                covered |= rng_ids
            self.assertEqual(covered, set(range(1, 316)), f"{cond} stage ranges don't cover 1..315")

    def test_original_stage_of(self):
        self.assertEqual(P.original_stage_of("C1", 1), "initial")
        self.assertEqual(P.original_stage_of("C1", 47), "initial")
        self.assertEqual(P.original_stage_of("C1", 48), "normal")
        self.assertEqual(P.original_stage_of("C1", 315), "severe")


class TestAttentionModules(unittest.TestCase):
    def test_channel_attention_shape(self):
        ca = ChannelAttention(32, r=16)
        x = torch.randn(2, 32, 16, 16)
        s = ca(x)
        self.assertEqual(s.shape, (2, 32, 1, 1))
        self.assertTrue((s >= 0).all() and (s <= 1).all())  # sigmoid output

    def test_spatial_attention_shape(self):
        sa = SpatialAttention(32, r=16)
        x = torch.randn(2, 32, 16, 16)
        Ms = sa(x)
        self.assertEqual(Ms.shape, (2, 1, 16, 16))
        self.assertTrue((Ms >= 0).all() and (Ms <= 1).all())

    def test_channel_spatial_attention_shape_preserved(self):
        attn = ChannelSpatialAttention(32, r=16)
        x = torch.randn(2, 32, 16, 16)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_attention_gradient_flow(self):
        attn = ChannelSpatialAttention(32, r=16)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        out = attn(x).sum()
        out.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestModel(unittest.TestCase):
    def test_dual_branch_and_fusion_shape(self):
        m = MultiAttentionCNN()
        f = m.force_conv(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        v = m.vib_conv(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(f.shape, (2, 16, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(v.shape, (2, 16, IMAGE_SIZE, IMAGE_SIZE))
        fused = torch.cat([f, v], dim=1)
        self.assertEqual(fused.shape, (2, 32, IMAGE_SIZE, IMAGE_SIZE))

    def test_forward_shape(self):
        m = MultiAttentionCNN()
        f = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        v = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        y = m(f, v)
        self.assertEqual(y.shape, (4, 3))

    def test_softmax_sums_to_one(self):
        m = MultiAttentionCNN()
        f = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        v = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        probs = F.softmax(m(f, v), dim=1)
        np.testing.assert_allclose(probs.sum(dim=1).detach().numpy(), np.ones(4), atol=1e-5)

    def test_no_nan_forward_backward(self):
        m = MultiAttentionCNN()
        f = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        v = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        y = m(f, v)
        self.assertFalse(torch.isnan(y).any())
        loss = F.cross_entropy(y, torch.tensor([0, 1, 2, 1]))
        loss.backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_single_batch_overfit(self):
        torch.manual_seed(0)
        m = MultiAttentionCNN()
        f = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
        v = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.randint(0, 3, (16,))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        losses = []
        for _ in range(80):
            opt.zero_grad()
            out = m(f, v)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        final_acc = (m(f, v).argmax(dim=1) == y).float().mean().item()
        self.assertLess(losses[-1], losses[0] * 0.3, f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}")
        self.assertGreaterEqual(final_acc, 0.9, f"failed to overfit 16 samples, acc={final_acc:.3f}")


if __name__ == "__main__":
    t0 = time.time()
    unittest.main(verbosity=2, exit=False)
    print(f"\nTotal test time: {time.time() - t0:.1f}s")
