# -*- coding: utf-8 -*-
"""B6 MTF-AViTK smoke test — CPU only, never runs real training.

Levels (run top to bottom, stop and report at the first level that fails or
looks too slow to continue):

  1. import all vendored modules (kan, model, preprocessing, label_utils,
     data_prep, train_core, adapter)
  2. instantiate MTF_AViTK on CPU, report param count
  3. single real image forward pass: build ONE MTF image from real PHM2010
     raw signal (C1, run 1) via the vendored preprocessing pipeline, run it
     through the model once, report output shape + timing
  4. full adapter smoke-test path: MTFAViTKAdapter(...).run(smoke_test=True)
     against a tiny real task cell, writing to tmp/smoke_tests/ (never
     results/), verify it returns status="done" and writes the expected files

Never trains (`train()` is skipped by MethodAdapter.run() whenever
smoke_test=True), never touches results/, never writes a checkpoint.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(METHOD_DIR))
sys.path.insert(0, str(METHOD_DIR / "code"))

# Point preprocessing/label_utils at the OLD project's raw archive for this
# smoke test ONLY (scripts/download_phm2010.py, not this fork's job, will
# populate data/PHM2010/raw/ for real runs) -- explicit env var override,
# never hardcoded in code/.
OLD_PROJECT_ROOT = REPO_ROOT.parent
os.environ.setdefault("PHM2010_ROOT", str(OLD_PROJECT_ROOT / "archive"))


def level1_imports():
    print("[1/4] importing vendored modules...")
    t0 = time.time()
    import kan  # noqa: F401
    import model  # noqa: F401
    import preprocessing  # noqa: F401
    import label_utils  # noqa: F401
    import data_prep  # noqa: F401
    import train_core  # noqa: F401
    from adapter import MTFAViTKAdapter, ADAPTER_CLASS  # noqa: F401
    print(f"    OK ({time.time()-t0:.2f}s)")
    return True


def level2_instantiate():
    print("[2/4] instantiating MTF_AViTK on CPU...")
    import torch
    from model import MTF_AViTK
    t0 = time.time()
    m = MTF_AViTK()
    n_params = m.num_parameters()
    print(f"    OK ({time.time()-t0:.2f}s), n_params={n_params:,} ({n_params/1e6:.1f}M)")
    return m


def level3_single_forward(m):
    print("[3/4] single real-image forward pass (C1 run 1, CPU)...")
    import torch
    import preprocessing as P
    t0 = time.time()
    images = P.build_main_samples("C1", 1)  # 5 sub-window images, [384,384,3] uint8 each
    t_img = time.time() - t0
    print(f"    built {len(images)} real MTF image(s) from raw signal in {t_img:.2f}s, "
          f"shape={images[0].shape} dtype={images[0].dtype}")

    img = torch.from_numpy(images[0].astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0)  # [1,3,384,384]
    m.eval()
    t0 = time.time()
    with torch.no_grad():
        out = m(img)
    t_fwd = time.time() - t0
    print(f"    forward pass OK ({t_fwd:.2f}s), output shape={tuple(out.shape)}")
    assert out.shape == (1, 3), f"expected [1,3], got {tuple(out.shape)}"
    return True


def level4_adapter_smoke_test():
    print("[4/4] full adapter smoke-test path (MTFAViTKAdapter.run(smoke_test=True))...")
    from adapter import MTFAViTKAdapter

    # KNOWN SHARED-CODE BUG (not this method's code) worked around HERE ONLY
    # so this method's own plumbing can still be verified -- see this file's
    # note in the fork report / README.md "Blockers" section.
    # shared/utils/run_meta.py::_torch_cuda_gpu_info() calls
    # torch.cuda.get_device_name(0) whenever torch.cuda.is_available() is
    # True, but on this machine (CUDA_VISIBLE_DEVICES="") is_available()
    # returns True even though device_count()==0, so get_device_name(0)
    # raises AssertionError("Invalid device id"). The real fix belongs in
    # shared/utils/run_meta.py (also check device_count() > 0), which is out
    # of this fork's scope (methods/B6_MTF_AViTK/ only) -- not applied here.
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() == 0:
        torch.cuda.get_device_name = lambda *a, **k: "N/A (CUDA_VISIBLE_DEVICES empty, shared/utils/run_meta.py bug workaround for this smoke test only)"

    out_dir = REPO_ROOT / "tmp" / "smoke_tests" / "B6_MTF_AViTK" / "D1_seed0"
    a = MTFAViTKAdapter(
        task="D1", train_cutters=["C1", "C4"], test_cutter="C6",
        seed=0, preprocess_seed=42, output_dir=out_dir, device="cpu",
        config={"batch_size": 2},
    )
    t0 = time.time()
    result = a.run(resume=False, smoke_test=True)
    dt = time.time() - t0
    print(f"    run() -> {result} ({dt:.2f}s)")
    assert result["status"] == "done", result
    for f in ["predictions.csv", "metrics.json", "run_meta.json", "config_resolved.yaml", "training_log.csv"]:
        p = out_dir / f
        assert p.exists(), f"missing expected output file: {p}"
    print(f"    OK, outputs written under {out_dir} (tmp/, not results/)")
    return True


if __name__ == "__main__":
    level1_imports()
    m = level2_instantiate()
    level3_single_forward(m)
    level4_adapter_smoke_test()
    print("\nALL SMOKE-TEST LEVELS PASSED (CPU only, no training was run).")
