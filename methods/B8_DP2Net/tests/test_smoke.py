"""CPU-only smoke test for B8 (DP2Net-adapted, B-D1). Real data, real S+G+F
instantiation, one untrained S+F forward pass -> real output shapes. No
training loop is exercised. Requires PHM2010_ROOT (see data/README.md).

Usage:
    PHM2010_ROOT=/path/to/archive python tests/test_smoke.py
"""
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_DIR.parents[1]
sys.path.insert(0, str(METHOD_DIR))

from adapter import ADAPTER_CLASS  # noqa: E402


def main():
    out_dir = REPO_ROOT / "tmp" / "smoke_tests" / "B8_DP2Net"
    adapter = ADAPTER_CLASS(
        task="D1",
        train_cutters=["C1", "C4"],
        test_cutter="C6",
        seed=42,
        preprocess_seed=42,
        output_dir=out_dir,
        device="cpu",
        config={"debug_max_runs": 8},
    )
    result = adapter.run(resume=False, smoke_test=True)
    print("run() result:", result)
    assert result["status"] == "done", result
    for fname in ["predictions.csv", "metrics.json", "run_meta.json", "config_resolved.yaml"]:
        p = out_dir / fname
        assert p.exists(), f"missing expected output {p}"
        print(f"OK: {p} ({p.stat().st_size} bytes)")
    import pandas as pd
    preds = pd.read_csv(out_dir / "predictions.csv")
    print(preds)
    assert set(["run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]).issubset(preds.columns)
    assert len(preds) > 0
    print(f"[smoke B8] OK -- {len(preds)} run-level predictions, columns={list(preds.columns)}")


if __name__ == "__main__":
    main()
