#!/usr/bin/env python
"""Downloads any extra pretrained/external assets a method needs beyond the
core PHM2010 data + its own from-scratch-trained checkpoint.

Audit result (2026-08-30, grepped the old project's baselines/ for
`pretrained=True`, `torchvision.models`, `timm.create_model`, `from_pretrained`,
`hub.load` -- zero hits across all of B4-B8): **none of the 9 methods need
any external pretrained weights, CLIP/ViT checkpoints, or graph caches.**
Every model (including B6's 309M-param ViT-L/32) trains entirely from scratch.
B6's "KAN" classifier head is a locally vendored file
(methods/B6_MTF_AViTK/code/kan.py), not a package or external download.

This script is therefore a documented no-op for all 9 methods in this round.
It exists (rather than being omitted) so the interface described in the
project's task spec is honestly present, and so a FUTURE method that does need
an external asset has an obvious place to add a fetcher.

Usage:
    python scripts/download_assets.py --method B6
"""
from __future__ import annotations

import argparse
import sys


NO_EXTRA_ASSETS = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    args = p.parse_args()

    if args.method in NO_EXTRA_ASSETS:
        print(f"[download_assets] {args.method}: no extra pretrained/external assets needed "
              f"(trains entirely from scratch) -- nothing to download.")
        return 0

    print(f"[download_assets] unknown method {args.method!r}, or a new method was added without "
          f"updating this script's asset table.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
