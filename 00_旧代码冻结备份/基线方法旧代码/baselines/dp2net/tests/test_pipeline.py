# -*- coding: utf-8 -*-
r"""
Unit tests for the DP2Net baseline: preprocessing sanity, S/G/F shapes,
MMD/Vst numeric properties, gradient flow through both training stages,
and a single-batch overfit check. Run with:

    python baselines/dp2net/tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
sys.path.insert(0, str(BASE_DIR))

import preprocessing as P  # noqa: E402
from model import SpatialAttention, Generator, WDCNN, mmd_loss, DP2Net  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
def test_lowpass_cutoff_sanity():
    """A pure tone well above the cutoff should be strongly attenuated;
    one well below should pass through mostly intact."""
    fs = P.FS
    t = np.arange(0, 4608) / fs
    low_tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)     # well below 1733Hz
    high_tone = np.sin(2 * np.pi * 10000 * t).astype(np.float32)   # well above 1733Hz
    low_out = P.lowpass_filter(low_tone)
    high_out = P.lowpass_filter(high_tone)
    low_ratio = np.std(low_out) / np.std(low_tone)
    high_ratio = np.std(high_out) / np.std(high_tone)
    assert low_ratio > 0.9, f"low tone over-attenuated: ratio={low_ratio:.3f}"
    assert high_ratio < 0.3, f"high tone not attenuated: ratio={high_ratio:.3f}"


def test_sample_length():
    assert P.SAMPLE_LEN == 4608


def test_k_receptive_field_physics():
    physics_k = (P.FS * 60 / P.N_SPEED) / P.N_TEETH / P.KPOOL
    assert abs(physics_k - P.K_RECEPTIVE) < 2, f"physics k={physics_k:.2f} vs paper k={P.K_RECEPTIVE}"


def test_vst_periodicity_and_range():
    vst = P.build_vst()
    assert vst.shape == (P.SAMPLE_LEN,)
    assert vst.min() >= -1.0 - 1e-6 and vst.max() <= 1.0 + 1e-6
    period = int(round(P.VST_PERIOD_L))
    # periodicity check: the pattern should repeat every ~period samples
    n_periods = len(vst) // period
    assert n_periods >= 10
    one = vst[:period]
    another = vst[period:2 * period]
    assert np.allclose(one, another, atol=1e-5)


def test_stage4_boundary_iv_empirically_zero_documented():
    """PAPER_SPEC.md sec 6b: Stage IV never triggers for C1/C4/C6 in this
    project's real archive. Confirms the documented finding rather than
    silently masking it."""
    for cond in ("C1", "C4", "C6"):
        df = P.assign_paper_native_4stage(cond)
        counts = df["stage4"].value_counts().to_dict()
        assert counts.get("IV", 0) == 0, f"{cond}: expected 0 stage-IV passes, got {counts.get('IV')}"
        assert counts.get("I", 0) > 0 and counts.get("II", 0) > 0 and counts.get("III", 0) > 0


# ---------------------------------------------------------------------------
# Model component tests
# ---------------------------------------------------------------------------
def test_s_output_shape():
    s = SpatialAttention()
    x = torch.randn(4, 1, 4608)
    fa, wa = s(x)
    assert fa.shape == (4, 1, 4608)
    assert wa.min().item() >= 0.0 and wa.max().item() <= 1.0


def test_g_output_shape_and_range():
    g = Generator()
    fa = torch.randn(4, 1, 4608)
    wg = g(fa)
    assert wg.shape == (4, 1, 4608)
    assert not torch.isnan(wg).any()
    assert wg.min().item() >= -1.0 - 1e-4 and wg.max().item() <= 1.0 + 1e-4


def test_adain_finite():
    from model import AdaIN1d
    a = AdaIN1d(4)
    x = torch.randn(4, 4, 100)
    out = a(x)
    assert not torch.isnan(out).any()
    assert out.shape == x.shape


def test_mse_constraint_computable():
    g = Generator()
    fa = torch.randn(2, 1, 4608)
    wg = g(fa)
    vst = torch.from_numpy(P.build_vst()).float().unsqueeze(0).expand(2, -1).unsqueeze(1)
    loss = F.mse_loss(wg, vst)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_mmd_finite_and_zero_for_identical():
    x = torch.randn(16, 10)
    assert mmd_loss(x, x).item() < 1e-4
    y = torch.randn(16, 10) + 5.0
    val = mmd_loss(x, y)
    assert not torch.isnan(val)
    assert val.item() > 0


def test_mmd_gradient_direction():
    """L_G = L_MSE - alpha*L_MMD (Eq.10): gradient step on -MMD should
    push generated features to become MORE different from source (since
    G is minimizing L_G, i.e. maximizing MMD)."""
    torch.manual_seed(0)
    xs = torch.randn(16, 8)
    xg = torch.nn.Parameter(torch.randn(16, 8) * 0.1)
    opt = torch.optim.SGD([xg], lr=0.1)
    mmd_before = mmd_loss(xs, xg.detach()).item()
    for _ in range(20):
        opt.zero_grad()
        loss = -mmd_loss(xs, xg)   # G minimizes L_MSE - alpha*MMD -> the MMD term alone is "-alpha*MMD" to minimize -> maximize MMD
        loss.backward()
        opt.step()
    mmd_after = mmd_loss(xs, xg.detach()).item()
    assert mmd_after > mmd_before, f"MMD should increase (diversity maximized): before={mmd_before:.4f} after={mmd_after:.4f}"


def test_wdcnn_output_and_softmax():
    f = WDCNN(num_classes=3)
    x = torch.randn(4, 1, 4608)
    logits = f(x)
    assert logits.shape == (4, 3)
    probs = F.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    assert not torch.isnan(logits).any()


def test_full_pipeline_gradient_flow_both_stages():
    """Stage 1 (S+F, CE) and Stage 2 (G via L_MSE-alpha*MMD, F via task
    loss) must each produce finite gradients (task instruction #76)."""
    torch.manual_seed(0)
    s, g, f = SpatialAttention(), Generator(), WDCNN(num_classes=3)
    x = torch.randn(4, 1, 4608)
    y = torch.randint(0, 3, (4,))
    vst = torch.from_numpy(P.build_vst()).float().unsqueeze(0)

    # Stage 1
    opt1 = torch.optim.Adam(list(s.parameters()) + list(f.parameters()), lr=1e-3)
    fa, _ = s(x)
    loss1 = F.cross_entropy(f(fa), y)
    opt1.zero_grad()
    loss1.backward()
    gn1 = torch.nn.utils.clip_grad_norm_(list(s.parameters()) + list(f.parameters()), 1e9)
    assert not torch.isnan(gn1)
    opt1.step()

    # Stage 2 - G
    for p in s.parameters():
        p.requires_grad_(False)
    fa, _ = s(x)
    wg = g(fa)
    l_mse = F.mse_loss(wg, vst.expand_as(wg))
    fg = wg * x
    _, emb_s = f(fa, return_features=True)
    _, emb_g = f(fg, return_features=True)
    l_g = l_mse - 20.0 * mmd_loss(emb_s.detach(), emb_g)
    opt_g = torch.optim.Adam(g.parameters(), lr=1e-5)
    opt_g.zero_grad()
    l_g.backward()
    gn_g = torch.nn.utils.clip_grad_norm_(g.parameters(), 1e9)
    assert not torch.isnan(gn_g)
    opt_g.step()

    # Stage 2 - F
    fa2, _ = s(x)
    wg2 = g(fa2).detach()
    fg2 = wg2 * x
    l_task = F.cross_entropy(f(fa2), y) + F.cross_entropy(f(fg2), y)
    opt_f = torch.optim.Adam(f.parameters(), lr=1e-3)
    opt_f.zero_grad()
    l_task.backward()
    gn_f = torch.nn.utils.clip_grad_norm_(f.parameters(), 1e9)
    assert not torch.isnan(gn_f)
    opt_f.step()


def test_dp2net_param_count_sane():
    m = DP2Net()
    n = m.num_parameters()
    assert 10_000 < n < 5_000_000, f"suspiciously sized model: {n} params"


def test_single_batch_overfit_stage1():
    """CPU-feasible single-batch overfit check for Stage 1 (S+F, plain
    CE) -- the cheapest of the two stages to verify convergence on
    (task instruction #77). Stage 2's full two-loss convergence is
    deferred to the GPU tutorial in README.md, same rationale as
    baselines/dynamic_gin_tgp/tests/test_pipeline.py."""
    torch.manual_seed(0)
    s, f = SpatialAttention(), WDCNN(num_classes=3)
    opt = torch.optim.Adam(list(s.parameters()) + list(f.parameters()), lr=1e-3)
    x = torch.randn(8, 1, 4608)
    y = torch.randint(0, 3, (8,))
    losses = []
    for _ in range(25):
        opt.zero_grad()
        fa, _ = s(x)
        logits = f(fa)
        loss = F.cross_entropy(logits, y)
        assert not torch.isnan(loss)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    acc = (f(s(x)[0]).argmax(1) == y).float().mean().item()
    assert acc >= 0.9, f"failed to overfit: final acc={acc:.3f}, losses={losses[::5]}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"running {t.__name__} ...", end=" ", flush=True)
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
