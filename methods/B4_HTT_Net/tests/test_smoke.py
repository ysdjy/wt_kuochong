# -*- coding: utf-8 -*-
"""CPU-only smoke test for B4 (HTT-Net): prepare() on real D1 data, build_model(),
one forward pass. Does NOT train (no epoch loop) and does NOT touch results/ --
writes only under tmp/smoke_tests/B4_HTT_Net/ via MethodAdapter.run(smoke_test=True).

Run with:
    python 扩充实验代码/methods/B4_HTT_Net/tests/test_smoke.py
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
METHOD_DIR = THIS_DIR.parent
REPO_ROOT = METHOD_DIR.parents[1]

sys.path.insert(0, str(METHOD_DIR))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from adapter import HTTNetAdapter  # noqa: E402
from phm2010.tasks import test_cutter, train_cutters  # noqa: E402


def main():
    task = "D1"
    out_dir = REPO_ROOT / "tmp" / "smoke_tests" / "B4_HTT_Net" / task
    adapter = HTTNetAdapter(
        task=task,
        train_cutters=train_cutters(task),
        test_cutter=test_cutter(task),
        seed=42,
        preprocess_seed=42,
        output_dir=out_dir,
        device="cpu",
    )
    result = adapter.run(resume=False, smoke_test=True)
    print("run() status:", result)
    assert result["status"] == "done", result

    import pandas as pd

    preds = pd.read_csv(out_dir / "predictions.csv")
    print("predictions.csv shape:", preds.shape)
    print(preds.head(3).to_string())
    assert set(["run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]).issubset(preds.columns)
    assert len(preds) > 0

    print(f"n_selected_features = {len(adapter._selected)}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
