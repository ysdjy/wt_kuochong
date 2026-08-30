"""CPU-only plumbing smoke test for B1 (RF).

NOT a real/paper-faithful run -- this exists to prove prepare()/build_model()/
train()/predict()/evaluate() work end-to-end against real D1 data, using a
drastically abbreviated model (n_estimators=5) so it finishes in seconds.
Writes scratch output only under 扩充实验代码/tmp/smoke_tests/B1_RF/, never
under results/.

Run: python methods/B1_RF/tests/smoke_test.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "methods" / "B1_RF"))

from adapter import RFAdapter  # noqa: E402


def main():
    out_dir = REPO_ROOT / "tmp" / "smoke_tests" / "B1_RF"
    adapter = RFAdapter(
        task="D1",
        train_cutters=["C1", "C4"],
        test_cutter="C6",
        seed=42,
        preprocess_seed=42,
        output_dir=out_dir,
        device="cpu",
        config={},
    )
    # Abbreviate the model for smoke-test speed only.
    adapter.N_ESTIMATORS = 5

    result = adapter.run(resume=False, smoke_test=False)
    print("run() result:", result)

    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        print("metrics.json:", metrics_path.read_text(encoding="utf-8"))
    assert result["status"] == "done", f"smoke test FAILED: {result}"
    assert (out_dir / "DONE.flag").exists(), "DONE.flag missing after a 'done' status"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
