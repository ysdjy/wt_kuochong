import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import (  # noqa: E402
    CUTTERS,
    TASKS,
    assert_no_test_leakage,
    resolve_tasks,
    test_cutter,
    train_cutters,
)


class TestTaskDefinitions(unittest.TestCase):
    def test_d1_d2_d3_definitions(self):
        self.assertEqual(TASKS["D1"], {"train": ["C1", "C4"], "test": "C6"})
        self.assertEqual(TASKS["D2"], {"train": ["C1", "C6"], "test": "C4"})
        self.assertEqual(TASKS["D3"], {"train": ["C4", "C6"], "test": "C1"})

    def test_every_task_covers_all_cutters_with_no_overlap(self):
        for name, spec in TASKS.items():
            assert_no_test_leakage(spec["train"], spec["test"])

    def test_helpers(self):
        self.assertEqual(train_cutters("D2"), ["C1", "C6"])
        self.assertEqual(test_cutter("D2"), "C4")


class TestResolveTasks(unittest.TestCase):
    def test_all(self):
        self.assertEqual(resolve_tasks("all"), ["D1", "D2", "D3"])

    def test_subset_and_case_insensitive(self):
        self.assertEqual(resolve_tasks("d1,d3"), ["D1", "D3"])

    def test_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            resolve_tasks("D1,D5")


class TestLeakageGuard(unittest.TestCase):
    def test_test_cutter_in_train_raises(self):
        with self.assertRaises(AssertionError):
            assert_no_test_leakage(["C1", "C4", "C6"], "C6")

    def test_incomplete_coverage_raises(self):
        with self.assertRaises(AssertionError):
            assert_no_test_leakage(["C1"], "C4")


if __name__ == "__main__":
    unittest.main()
