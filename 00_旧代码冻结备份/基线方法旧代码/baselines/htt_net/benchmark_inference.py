# -*- coding: utf-8 -*-
r"""
Rigorous inference-time benchmark for the final, frozen HTT-Net checkpoint.
Warms up, repeats multiple trials, reports mean ms/sample and samples/s for
a stated hardware + batch size. No FLOPs library is installed in this
environment (thop/ptflops/fvcore all absent) and none was installed for
this purpose alone, per instructions -- params + wall-clock timing are
reported instead.

Usage:
    python benchmark_inference.py --checkpoint <path> --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import train as T

THIS_DIR = Path(__file__).resolve().parent


def benchmark(checkpoint_path: Path, batch_size: int, n_warmup: int, n_repeat: int, input_dim: int, arch: dict):
    device = T.base.DEVICE
    model = T.HTTNet(input_dim=input_dim, num_classes=3, **arch).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = torch.randn(batch_size, 12, input_dim, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    import numpy as np
    times = np.array(times)
    per_batch_s = float(times.mean())
    per_sample_ms = per_batch_s / batch_size * 1000.0
    samples_per_s = batch_size / per_batch_s

    result = {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "batch_size": batch_size,
        "n_warmup": n_warmup,
        "n_repeat": n_repeat,
        "mean_batch_latency_ms": per_batch_s * 1000.0,
        "std_batch_latency_ms": float(times.std()) * 1000.0,
        "ms_per_sample": per_sample_ms,
        "samples_per_second": samples_per_s,
        "n_params": model.num_parameters(),
        "checkpoint": str(checkpoint_path),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-repeat", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=45)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    arch = {
        "embed_dim": args.embed_dim,
        "depths": (2, 2, 2, 2),
        "num_heads": args.num_heads,
        "window_size": args.window_size,
        "dropout": args.dropout,
    }
    r = benchmark(Path(args.checkpoint), args.batch_size, args.n_warmup, args.n_repeat, args.input_dim, arch)
    print(json.dumps(r, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2), encoding="utf-8")
