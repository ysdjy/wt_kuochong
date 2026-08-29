# -*- coding: utf-8 -*-
r"""
Unit tests for the Dynamic GIN + TGP baseline: preprocessing sanity,
model shapes, numeric ranges, gradient flow, and a single-batch overfit
check. Run with:

    python -m pytest baselines/dynamic_gin_tgp/tests/test_pipeline.py -v

or directly:

    python baselines/dynamic_gin_tgp/tests/test_pipeline.py
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
from model import (DynamicGIN_TGP, GASFEncoder, TemporalFeatureExtraction,  # noqa: E402
                    build_static_graph, build_dynamic_graph, fuse_and_sparsify)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
def test_raw_segment_shape():
    raw = P.load_raw_pass("C1", 1)
    assert raw.shape[0] == 6, f"expected 6 channels (AE dropped), got {raw.shape[0]}"
    assert raw.shape[1] > 100_000


def test_stable_cut_trims_80k():
    raw = P.load_raw_pass("C1", 1)
    stable = P.stable_cut_region(raw)
    assert stable.shape[1] == raw.shape[1] - 2 * P.TRIM_POINTS


def test_window_shape():
    windows = P.all_portion_windows("C1", 1)
    assert windows.shape == (10, 6, 288)
    assert not np.isnan(windows).any()


def test_sample_counts_protocol_a():
    """PAPER_SPEC.md sec 2: 2520 total samples, 840/tool, exact
    5/2/3-per-run stratified stage counts."""
    total = 0
    for cond in ("C1", "C4", "C6"):
        manifest = P.build_protocol_a_manifest(cond)
        counts = manifest.groupby("stage_original").size().to_dict()
        assert counts["initial"] == 250, counts
        assert counts["normal"] == 320, counts
        assert counts["severe"] == 270, counts
        assert len(manifest) == 840
        total += len(manifest)
    assert total == 2520, total


# ---------------------------------------------------------------------------
# Model component tests
# ---------------------------------------------------------------------------
def test_temporal_extraction_shape():
    m = TemporalFeatureExtraction()
    x = torch.randn(4, 1, 6, 288)
    out = m(x)
    assert out.shape == (4, 1, 24, 288), out.shape


def test_gasf_range_and_shape():
    m = GASFEncoder()
    xsf3 = torch.randn(2, 1, 24, 288)
    gasf = m(xsf3)
    assert gasf.shape == (2, 24, 288, 288)
    assert gasf.max().item() <= 1.0 + 1e-4
    assert gasf.min().item() >= -1.0 - 1e-4
    assert not torch.isnan(gasf).any()


def test_static_adjacency_symmetry():
    xg = torch.randn(4, 24, 64)
    a_static = build_static_graph(xg)
    assert a_static.shape == (24, 24)
    assert torch.allclose(a_static, a_static.t(), atol=1e-5)


def test_dynamic_adjacency_shape():
    xg = torch.randn(4, 24, 64)
    a_dyn = build_dynamic_graph(xg)
    assert a_dyn.shape == (4, 24, 24)


def test_topk_edge_count():
    a_static = torch.rand(24, 24)
    a_dynamic = torch.rand(4, 24, 24)
    mask = fuse_and_sparsify(a_static, a_dynamic, topk=144)
    assert mask.shape == (4, 24, 24)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    # symmetrized top-k with k=144 out of 576 entries -> nonzero count is
    # >= 144 (symmetrization can only add entries, never remove)
    for b in range(4):
        assert mask[b].sum().item() >= 144


def test_full_model_forward_shape_and_no_nan():
    torch.manual_seed(0)
    m = DynamicGIN_TGP(topk=144)
    x = torch.randn(4, 6, 288)
    logits = m(x)
    assert logits.shape == (4, 3)
    assert not torch.isnan(logits).any()
    probs = F.softmax(logits, dim=1)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_param_count_close_to_paper():
    """Paper reports 321,002 total params (Sec 3.4). Sanity tolerance
    +/-5% per task instruction #35."""
    m = DynamicGIN_TGP(topk=144)
    n = m.num_parameters()
    paper_n = 321_002
    rel_diff = abs(n - paper_n) / paper_n
    assert rel_diff < 0.05, f"param count {n} vs paper {paper_n}, rel diff {rel_diff:.3f}"


def test_gradient_flow_main_pathway():
    """Most params must receive gradient; graph_mlp's 4 params and
    tgp3.s_p are documented, expected exceptions (see model.py
    docstrings)."""
    torch.manual_seed(0)
    m = DynamicGIN_TGP(topk=144)
    x = torch.randn(4, 6, 288)
    logits = m(x)
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 1]))
    loss.backward()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    expected_dead = {"graph_mlp.net.0.weight", "graph_mlp.net.0.bias",
                      "graph_mlp.net.2.weight", "graph_mlp.net.2.bias",
                      "tgp3.s_p"}
    assert set(no_grad) == expected_dead, f"unexpected no-grad params: {set(no_grad) - expected_dead}"
    n_total = sum(1 for _ in m.parameters())
    assert len(no_grad) < n_total  # most params ARE trainable


def test_single_batch_overfit_cpu_smoke():
    """CPU-feasible sanity version of the single-batch-overfit check
    (task instruction #77). NOTE: this network's paper-specified
    Conv2d_4 (288 output channels on a ~282x282 spatial map, Table 1)
    costs ~80s/iteration on this CPU-only dev machine (confirmed by
    direct timing) -- a full 60-iteration/90%-accuracy convergence check
    (as run for every other, cheaper baseline in this project) would take
    over an hour here. GPU use is currently policy-blocked (another
    Claude instance's training may be active). This test therefore only
    checks the CPU-cheap invariants: no NaN across several real
    optimizer steps, and the loss trending down -- NOT full convergence.
    A full 60-iter/90%-accuracy GPU overfit run is included as the first
    step of README.md's manual training tutorial and MUST be run (by the
    user, on GPU) before trusting any real Protocol A/B result.

    This test previously caught a real bug: GASF's sin(phi)=sqrt(1-x^2)
    hit exactly 0 at every feature's own min/max sample (by construction
    of the [-1,1] normalization), and sqrt's gradient is infinite at 0,
    producing NaN gradients on the very first backward pass. Fixed in
    model.py's GASFEncoder via clamp_min(1e-8) instead of clamp_min(0.0)
    before the sqrt.
    """
    torch.manual_seed(0)
    m = DynamicGIN_TGP(topk=144)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.0)
    x = torch.randn(4, 6, 288)
    y = torch.randint(0, 3, (4,))
    losses = []
    n_steps = 6
    for _ in range(n_steps):
        opt.zero_grad()
        logits = m(x)
        loss = F.cross_entropy(logits, y)
        assert not torch.isnan(loss), "loss is NaN"
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
        assert not torch.isnan(gn), f"gradient norm is NaN (loss history so far: {losses})"
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease over {n_steps} steps: {losses}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"running {t.__name__} ...", end=" ")
        t()
        print("OK")
    print(f"\nAll {len(tests)} tests passed.")
