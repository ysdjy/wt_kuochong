# -*- coding: utf-8 -*-
"""CPU-only smoke test for the B5 adapter: prepare -> build_model -> predict
(no training loop, per task spec section 40/45 -- this round does not launch
real training). Verifies the full wiring (label/split logic, image
generation+cache, model forward pass, unified result-schema writing) runs
end to end on a tiny slice of real D1 data.

PHM2010_RAW_ROOT is pointed at the OLD project's live `archive/` for this
smoke test ONLY, since `scripts/download_phm2010.py` (which will populate
`data/PHM2010/raw/` for real use) is a separate, not-yet-built piece of this
task. Production runs must set PHM2010_RAW_ROOT to `data/PHM2010/raw/` (or
rely on its default) once that downloader exists -- this override is a
local, temporary dev/test convenience only, not baked into any non-test file.

Run: python methods/B5_MultiSource_Attention/tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
METHOD_DIR = THIS_DIR.parent
REPO_ROOT = METHOD_DIR.parents[1]
OLD_PROJECT_ROOT = REPO_ROOT.parent

os.environ.setdefault("PHM2010_RAW_ROOT", str(OLD_PROJECT_ROOT / "archive"))
os.environ.setdefault("B5_IMAGE_CACHE_DIR", str(REPO_ROOT / "tmp" / "smoke_tests" / "B5_MultiSource_Attention" / "images_cache"))

sys.path.insert(0, str(METHOD_DIR))
from adapter import MultiSourceAttentionAdapter  # noqa: E402

OUT_DIR = REPO_ROOT / "tmp" / "smoke_tests" / "B5_MultiSource_Attention" / "D1_seed0"


def main():
    adapter = MultiSourceAttentionAdapter(
        task="D1",
        train_cutters=["C1", "C4"],
        test_cutter="C6",
        seed=0,
        preprocess_seed=42,
        output_dir=OUT_DIR,
        device="cpu",
        config={"max_epochs": 1},
    )
    result = adapter.run(resume=False, smoke_test=True)
    print("run() result:", result)
    assert result["status"] == "done", f"smoke test failed: {result}"

    for fname in ["predictions.csv", "training_log.csv", "metrics.json", "run_meta.json", "config_resolved.yaml"]:
        p = OUT_DIR / fname
        assert p.exists(), f"missing expected output file: {p}"
        print(f"  OK  {fname}  ({p.stat().st_size} bytes)")

    import pandas as pd
    preds = pd.read_csv(OUT_DIR / "predictions.csv")
    print(f"predictions.csv: {len(preds)} rows, columns={list(preds.columns)}")
    assert set(["run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]).issubset(preds.columns)
    assert len(preds) == 3, f"expected 3 smoke-test test rows, got {len(preds)}"

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
