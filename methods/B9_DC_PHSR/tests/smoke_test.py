"""CPU-only shape/plumbing smoke test for B9. Does NOT train. Run:
    python methods/B9_DC_PHSR/tests/smoke_test.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from methods.B9_DC_PHSR.adapter import B9DCPHSRAdapter  # noqa: E402
from phm2010.tasks import test_cutter, train_cutters  # noqa: E402

out_dir = REPO_ROOT / "tmp" / "smoke_tests" / "B9_DC_PHSR" / "D1_seed42"

adapter = B9DCPHSRAdapter(
    task="D1", train_cutters=train_cutters("D1"), test_cutter=test_cutter("D1"),
    seed=42, preprocess_seed=42, output_dir=out_dir, device="cpu",
)
result = adapter.run(resume=False, smoke_test=True)
print("RESULT:", result)
assert result["status"] == "done", f"smoke test failed: {result}"

import pandas as pd
df = pd.read_csv(out_dir / "predictions.csv")
print("predictions shape:", df.shape)
print(df.head())
assert len(df) == 304
assert set(["run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]).issubset(df.columns)
# probability-inference columns must be valid probabilities summing to ~1
probsum = df[["p_early", "p_middle", "p_late"]].sum(axis=1)
assert (probsum.sub(1.0).abs() < 1e-6).all(), "final probabilities do not sum to 1"
print("B9 smoke test PASSED")
