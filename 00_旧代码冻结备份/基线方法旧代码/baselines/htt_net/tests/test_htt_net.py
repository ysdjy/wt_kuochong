# -*- coding: utf-8 -*-
r"""
Unit tests for the HTT-Net reimplementation (baselines/htt_net/model.py).

Plain-assertion test runner (no pytest dependency needed, though it also
works fine under pytest). Run directly:

    python baselines/htt_net/tests/test_htt_net.py

All tests use synthetic random data -- they do not require the PHM2010
feature file and validate implementation correctness only, not real-data
performance. See PAPER_SPEC.md and README.md for what was actually
validated against real data (nothing, as of writing: the feature CSV this
project's pipeline depends on is not present on this machine).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from model import (
    HTTNet,
    TokenMerging,
    build_shift_mask,
    pad_to_multiple,
    window_partition,
    window_reverse,
)

torch.manual_seed(0)


def test_forward_shape():
    model = HTTNet(input_dim=45, num_classes=3, embed_dim=32, window_size=3)
    x = torch.randn(8, 12, 45)
    out = model(x)
    assert out.shape == (8, 3), f"expected (8,3), got {tuple(out.shape)}"
    print("test_forward_shape: PASS")


def test_no_nan_forward_and_backward():
    model = HTTNet(input_dim=45, num_classes=3, embed_dim=32, window_size=3)
    x = torch.randn(8, 12, 45, requires_grad=True)
    y = torch.randint(0, 3, (8,))
    out = model(x)
    assert torch.isfinite(out).all(), "forward output contains NaN/Inf"
    loss = F.cross_entropy(out, y)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"
    print("test_no_nan_forward_and_backward: PASS")


def test_softmax_probabilities():
    model = HTTNet(input_dim=45, num_classes=3, embed_dim=32, window_size=3)
    model.eval()
    x = torch.randn(16, 12, 45)
    with torch.no_grad():
        probs = model.predict_proba(x)
    assert probs.shape == (16, 3)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(16), atol=1e-5), row_sums
    assert (probs >= 0).all() and (probs <= 1).all()
    print("test_softmax_probabilities: PASS")


def test_window_partition_reverse_roundtrip():
    x = torch.randn(4, 12, 8)
    windows = window_partition(x, window_size=3)
    assert windows.shape == (4 * 4, 3, 8)  # 12/3 = 4 windows per sample
    back = window_reverse(windows, window_size=3, B=4, L=12)
    assert torch.allclose(x, back), "window partition/reverse is not a lossless roundtrip"
    print("test_window_partition_reverse_roundtrip: PASS")


def test_shifted_window_shapes_and_mask():
    L, window_size, shift_size = 12, 3, 1
    x = torch.randn(2, L, 8)
    shifted = torch.roll(x, shifts=-shift_size, dims=1)
    unshifted_back = torch.roll(shifted, shifts=shift_size, dims=1)
    assert torch.allclose(x, unshifted_back), "shift/reverse-shift roundtrip failed"

    mask = build_shift_mask(L, window_size, shift_size, device=x.device)
    nW = L // window_size
    assert mask.shape == (nW, window_size, window_size)
    # mask values must only ever be 0 (allowed) or -100 (disallowed)
    assert set(torch.unique(mask).tolist()).issubset({0.0, -100.0})
    print("test_shifted_window_shapes_and_mask: PASS")


def test_token_merging_shape():
    merge = TokenMerging(dim=32)
    x = torch.randn(4, 12, 32)
    out = merge(x)
    assert out.shape == (4, 6, 64), f"expected (4,6,64), got {tuple(out.shape)}"

    # odd-length input must be right-padded before merging (L=3 -> pad to 4 -> merge to 2)
    x_odd = torch.randn(4, 3, 32)
    out_odd = merge(x_odd)
    assert out_odd.shape == (4, 2, 64), f"expected (4,2,64), got {tuple(out_odd.shape)}"
    print("test_token_merging_shape: PASS")


def test_pad_to_multiple():
    x = torch.arange(3 * 5).reshape(1, 3, 5).float()
    padded, valid_len = pad_to_multiple(x, 2)
    assert padded.shape == (1, 4, 5)
    assert valid_len == 3
    assert torch.allclose(padded[:, 3, :], padded[:, 2, :]), "padding must repeat the last valid token"
    print("test_pad_to_multiple: PASS")


def test_l12_full_stage_progression_shapes():
    """Sanity-check the L=12 special case end to end (PAPER_SPEC.md sec 4):
    12 -> 6 -> 3 -> (pad)4 -> 2, with 4 stages total, no crashes."""
    model = HTTNet(input_dim=45, num_classes=3, embed_dim=16, window_size=3)
    x = torch.randn(3, 12, 45)
    out = model(x)
    assert out.shape == (3, 3)
    print("test_l12_full_stage_progression_shapes: PASS")


def test_single_batch_overfit():
    """Sanity check per task instructions: model must be able to memorize a
    tiny fixed batch. If this fails, something is architecturally broken
    (label bug, dead attention, masking bug, etc.) and no real training
    should be attempted before fixing it."""
    torch.manual_seed(0)
    model = HTTNet(input_dim=10, num_classes=3, embed_dim=16, window_size=3, dropout=0.0)
    x = torch.randn(24, 12, 10)
    y = torch.randint(0, 3, (24,))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    model.train()
    losses = []
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    acc = (pred == y).float().mean().item()
    assert losses[-1] < 0.05, f"final training loss too high: {losses[-1]:.4f} (expected overfit)"
    assert acc > 0.95, f"single-batch overfit accuracy too low: {acc:.4f}"
    print(f"test_single_batch_overfit: PASS (final_loss={losses[-1]:.4f}, acc={acc:.4f})")


def test_parameter_count_is_reasonable():
    model = HTTNet(input_dim=45, num_classes=3, embed_dim=32, window_size=3)
    n = model.num_parameters()
    assert 1_000 < n < 20_000_000, f"suspicious parameter count: {n}"
    print(f"test_parameter_count_is_reasonable: PASS (n_params={n})")


ALL_TESTS = [
    test_forward_shape,
    test_no_nan_forward_and_backward,
    test_softmax_probabilities,
    test_window_partition_reverse_roundtrip,
    test_shifted_window_shapes_and_mask,
    test_token_merging_shape,
    test_pad_to_multiple,
    test_l12_full_stage_progression_shapes,
    test_single_batch_overfit,
    test_parameter_count_is_reasonable,
]


if __name__ == "__main__":
    failed = []
    for t in ALL_TESTS:
        try:
            t()
        except AssertionError as e:
            failed.append((t.__name__, str(e)))
            print(f"{t.__name__}: FAIL - {e}")
    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)}/{len(ALL_TESTS)} tests FAILED")
        sys.exit(1)
    else:
        print(f"All {len(ALL_TESTS)} tests PASSED")
