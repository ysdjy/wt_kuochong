# -*- coding: utf-8 -*-
r"""
Unit tests for the MTF-AViTK baseline (preprocessing.py + model.py + kan.py).

Run with:
    python tests/test_mtf_avitk.py
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
from kan import KANLinear, KANClassifier
from model import MTF_AViTK, AdaptViT, AdaptMLP, PatchEmbed, MultiHeadSelfAttention


class TestPreprocessing(unittest.TestCase):
    def test_resultant_force(self):
        fx = np.array([3.0, 0.0])
        fy = np.array([4.0, 0.0])
        fz = np.array([0.0, 5.0])
        f = P.resultant_force(np.stack([fx, fy, fz], axis=1))
        np.testing.assert_allclose(f, [5.0, 5.0], atol=1e-9)

    def test_wavelet_denoise_shape_and_finite(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 20, 2000)) + rng.normal(0, 0.1, 2000)
        out = P.wavelet_denoise(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_wavelet_denoise_reduces_noise(self):
        rng = np.random.default_rng(1)
        t = np.linspace(0, 20, 2000)
        clean = np.sin(t)
        noisy = clean + rng.normal(0, 0.3, 2000)
        denoised = P.wavelet_denoise(noisy)
        err_before = np.mean((noisy - clean) ** 2)
        err_after = np.mean((denoised - clean) ** 2)
        self.assertLess(err_after, err_before)

    def test_mtf_encode_shape_and_range(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=2000)
        field_size = 64  # small size for a fast test
        M = P.mtf_encode(x, field_size=field_size, n_bins=8)
        self.assertEqual(M.shape, (field_size, field_size))
        self.assertTrue(np.all(M >= -1e-9) and np.all(M <= 1 + 1e-9))
        self.assertTrue(np.all(np.isfinite(M)))

    def test_mtf_to_rgb_image_shape(self):
        rng = np.random.default_rng(3)
        field = rng.uniform(0, 1, size=(64, 64))
        img = P.mtf_to_rgb_image(field, vit_size=96)
        self.assertEqual(img.shape, (96, 96, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_split_subwindows_count_and_shape(self):
        rng = np.random.default_rng(4)
        force_xyz = rng.normal(size=(10_000, 3))
        subs = P.split_subwindows(force_xyz)
        self.assertEqual(len(subs), 5)
        for s in subs:
            self.assertEqual(s.shape, (2000, 3))

    def test_load_raw_signal_and_build_main_samples(self):
        images = P.build_main_samples("C1", 1)
        self.assertEqual(len(images), 5)
        for img in images:
            self.assertEqual(img.shape, (384, 384, 3))
            self.assertEqual(img.dtype, np.uint8)


class TestKAN(unittest.TestCase):
    def test_kan_linear_shape(self):
        layer = KANLinear(10, 5)
        x = torch.randn(4, 10)
        y = layer(x)
        self.assertEqual(y.shape, (4, 5))

    def test_kan_linear_no_nan(self):
        layer = KANLinear(10, 5)
        x = torch.randn(8, 10) * 5  # values outside default grid_range, tests boundary robustness
        y = layer(x)
        self.assertFalse(torch.isnan(y).any())

    def test_kan_linear_gradient_flow(self):
        layer = KANLinear(10, 5)
        x = torch.randn(4, 10, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for p in layer.parameters():
            self.assertIsNotNone(p.grad)

    def test_kan_classifier_shape(self):
        clf = KANClassifier(in_features=1024, hidden=16, num_classes=3)
        x = torch.randn(4, 1024)
        y = clf(x)
        self.assertEqual(y.shape, (4, 3))


class TestModelComponents(unittest.TestCase):
    def test_patch_embed_shape(self):
        pe = PatchEmbed(img_size=384, patch_size=32, in_chans=3, embed_dim=1024)
        x = torch.randn(2, 3, 384, 384)
        y = pe(x)
        self.assertEqual(y.shape, (2, 144, 1024))

    def test_mhsa_shape(self):
        attn = MultiHeadSelfAttention(dim=1024, num_heads=16)
        x = torch.randn(2, 145, 1024)
        y = attn(x)
        self.assertEqual(y.shape, (2, 145, 1024))

    def test_adaptmlp_shape_and_zero_init_adapter(self):
        block = AdaptMLP(dim=1024, mlp_hidden=256, bottleneck=16, scale=0.1)
        x = torch.randn(2, 5, 1024)
        y = block(x)
        self.assertEqual(y.shape, (2, 5, 1024))
        # Adapter's up-projection is zero-initialized -> adapter branch contributes 0 at init.
        normed = block.norm(x)
        mlp_only = block.mlp(normed)
        self.assertTrue(torch.allclose(y, mlp_only, atol=1e-6))

    def test_small_vit_forward_shape(self):
        vit = AdaptViT(img_size=64, patch_size=16, embed_dim=32, depth=2, num_heads=4, mlp_hidden=64,
                        bottleneck=8, adapter_scale=0.1)
        x = torch.randn(2, 3, 64, 64)
        y = vit(x)
        self.assertEqual(y.shape, (2, 32))


class TestFullModel(unittest.TestCase):
    """Uses a shrunk-down MTF_AViTK (small ViT config) for fast CI-speed tests;
    a full ViT-L/32 forward/backward/param-count check is done separately."""

    def _small_model(self):
        vit_kwargs = dict(img_size=64, patch_size=16, embed_dim=32, depth=2, num_heads=4,
                           mlp_hidden=64, bottleneck=8, adapter_scale=0.1)
        return MTF_AViTK(num_classes=3, vit_kwargs=vit_kwargs, kan_kwargs={})

    def test_forward_shape(self):
        m = self._small_model()
        x = torch.randn(4, 3, 64, 64)
        y = m(x)
        self.assertEqual(y.shape, (4, 3))

    def test_softmax_sums_to_one(self):
        m = self._small_model()
        x = torch.randn(4, 3, 64, 64)
        y = m(x)
        probs = F.softmax(y, dim=1)
        np.testing.assert_allclose(probs.sum(dim=1).detach().numpy(), np.ones(4), atol=1e-5)

    def test_no_nan_forward_backward(self):
        m = self._small_model()
        x = torch.randn(4, 3, 64, 64)
        y = m(x)
        self.assertFalse(torch.isnan(y).any())
        loss = F.cross_entropy(y, torch.tensor([0, 1, 2, 1]))
        loss.backward()
        for p in m.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_full_size_model_shape_and_param_count(self):
        """Slow: builds the real ViT-L/32-scale model once."""
        m = MTF_AViTK()
        x = torch.randn(1, 3, 384, 384)
        y = m(x)
        self.assertEqual(y.shape, (1, 3))
        n_params = m.num_parameters()
        self.assertGreater(n_params, 2.5e8)  # sanity: ViT-L/32 scale, ~300M params

    def test_single_batch_overfit(self):
        """16 synthetic samples; the (small-config) model must memorize them
        given enough steps, per task instruction: if it can't overfit a
        tiny batch, do not proceed to real training."""
        torch.manual_seed(0)
        m = self._small_model()
        x = torch.randn(16, 3, 64, 64)
        y = torch.randint(0, 3, (16,))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        losses = []
        for _ in range(150):
            opt.zero_grad()
            out = m(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        final_acc = (m(x).argmax(dim=1) == y).float().mean().item()
        self.assertLess(losses[-1], losses[0] * 0.3, f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}")
        self.assertGreaterEqual(final_acc, 0.9, f"failed to overfit 16 samples, acc={final_acc:.3f}")


if __name__ == "__main__":
    t0 = time.time()
    unittest.main(verbosity=2, exit=False)
    print(f"\nTotal test time: {time.time() - t0:.1f}s")
