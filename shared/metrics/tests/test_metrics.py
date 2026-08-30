"""Unit tests for shared/metrics/metrics.py.

Run with: python -m pytest 扩充实验代码/shared/metrics/tests/test_metrics.py -v
(or: python -m unittest discover -s 扩充实验代码/shared/metrics/tests)
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import (  # noqa: E402
    classification_metrics,
    compute_all_metrics,
    confusion_matrix,
    q_regression_metrics,
    sequence_diagnostics,
    stage_id,
)


class TestStageId(unittest.TestCase):
    def test_string_labels(self):
        self.assertEqual(stage_id("early"), 0)
        self.assertEqual(stage_id("Middle"), 1)
        self.assertEqual(stage_id(" late "), 2)

    def test_int_passthrough(self):
        self.assertEqual(stage_id(0), 0)
        self.assertEqual(stage_id(np.int64(2)), 2)


class TestClassificationMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        truth = ["early", "early", "middle", "middle", "late", "late"]
        pred = list(truth)
        m = classification_metrics(truth, pred)
        self.assertAlmostEqual(m["Acc"], 1.0)
        self.assertAlmostEqual(m["MacroF1"], 1.0)
        self.assertAlmostEqual(m["E_F1"], 1.0)
        self.assertAlmostEqual(m["M_F1"], 1.0)
        self.assertAlmostEqual(m["L_F1"], 1.0)
        self.assertAlmostEqual(m["M_to_E"], 0.0)
        self.assertAlmostEqual(m["M_to_L"], 0.0)

    def test_hand_computed_confusion(self):
        # truth: E E M M M L   (2 early, 3 middle, 1 late)
        # pred:  E M M M E L   -> confusion:
        #   E->E:1 E->M:1
        #   M->M:2 M->E:1
        #   L->L:1
        truth = ["early", "early", "middle", "middle", "middle", "late"]
        pred = ["early", "middle", "middle", "middle", "early", "late"]
        cm = confusion_matrix([stage_id(v) for v in truth], [stage_id(v) for v in pred])
        expected = np.array([
            [1, 1, 0],  # true early -> pred early/middle/late
            [1, 2, 0],  # true middle -> pred early/middle/late
            [0, 0, 1],  # true late -> pred early/middle/late
        ])
        np.testing.assert_array_equal(cm, expected)

        m = classification_metrics(truth, pred)
        # Acc = 4/6
        self.assertAlmostEqual(m["Acc"], 4 / 6)
        # Middle: tp=2, fp=(col M sum - tp)=1-... col M = [1,2,0] sum=3, fp=3-2=1
        # fn = row M sum - tp = 3-2=1 -> precision=2/3, recall=2/3
        self.assertAlmostEqual(m["M_Precision"], 2 / 3)
        self.assertAlmostEqual(m["M_Recall"], 2 / 3)
        # M_to_E = cm[1,0]/middle_total = 1/3
        self.assertAlmostEqual(m["M_to_E"], 1 / 3, places=6)
        self.assertAlmostEqual(m["M_to_L"], 0.0)

    def test_accepts_integer_ids_directly(self):
        truth = [0, 0, 1, 1, 2, 2]
        pred = [0, 0, 1, 1, 2, 2]
        m = classification_metrics(truth, pred)
        self.assertAlmostEqual(m["Acc"], 1.0)


class TestSequenceDiagnostics(unittest.TestCase):
    def test_monotonic_no_reversal_no_jump(self):
        pred = ["early", "early", "middle", "middle", "late", "late"]
        probs = np.array([
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
            [0.0, 0.1, 0.9],
        ])
        d = sequence_diagnostics(pred, probs)
        self.assertEqual(d["Rev"], 0)
        self.assertEqual(d["Jump"], 0)
        self.assertGreater(d["Smooth"], 0.0)

    def test_reversal_and_jump_detected(self):
        # ids: 0 1 0 2  -> diffs: +1, -1, +2  => Rev counts diffs<0 => 1
        # Jump counts abs(diff)>=2 => the +2 step => 1
        pred = ["early", "middle", "early", "late"]
        probs = np.zeros((4, 3))
        d = sequence_diagnostics(pred, probs)
        self.assertEqual(d["Rev"], 1)
        self.assertEqual(d["Jump"], 1)

    def test_single_sample_smooth_is_nan(self):
        d = sequence_diagnostics(["early"], np.array([[1.0, 0.0, 0.0]]))
        self.assertTrue(math.isnan(d["Smooth"]))
        self.assertEqual(d["Rev"], 0)
        self.assertEqual(d["Jump"], 0)


class TestQRegressionMetrics(unittest.TestCase):
    def test_none_returns_nan(self):
        m = q_regression_metrics(None, None)
        self.assertTrue(all(math.isnan(v) for v in m.values()))

    def test_empty_returns_nan(self):
        m = q_regression_metrics([], [])
        self.assertTrue(all(math.isnan(v) for v in m.values()))

    def test_perfect_fit(self):
        q_true = [0.0, 0.5, 1.0]
        q_pred = [0.0, 0.5, 1.0]
        m = q_regression_metrics(q_true, q_pred)
        self.assertAlmostEqual(m["q_MAE"], 0.0)
        self.assertAlmostEqual(m["q_RMSE"], 0.0)
        self.assertAlmostEqual(m["q_R2"], 1.0)

    def test_known_mae_rmse(self):
        q_true = [0.0, 1.0, 2.0, 3.0]
        q_pred = [1.0, 1.0, 2.0, 5.0]
        # errors: 1, 0, 0, 2 -> MAE = 3/4 = 0.75, RMSE = sqrt((1+0+0+4)/4)=sqrt(1.25)
        m = q_regression_metrics(q_true, q_pred)
        self.assertAlmostEqual(m["q_MAE"], 0.75)
        self.assertAlmostEqual(m["q_RMSE"], math.sqrt(1.25))


class TestComputeAllMetrics(unittest.TestCase):
    def test_no_q_leaves_nan_without_faking(self):
        truth = ["early", "middle", "late"]
        pred = ["early", "middle", "late"]
        probs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        m = compute_all_metrics(truth, pred, probs)
        self.assertAlmostEqual(m["Acc"], 1.0)
        self.assertTrue(math.isnan(m["q_MAE"]))


if __name__ == "__main__":
    unittest.main()
