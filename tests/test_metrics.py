"""
Unit tests for GSAM metrics module (metrics.py).

Tests RFR computation, retrieval precision, concept coverage,
and transfer metrics.
"""

import os
import sys
import unittest
import importlib

# Import metrics directly to avoid gsam/__init__.py which triggers
# heavy imports (tiktoken, openai, etc.)
_spec = importlib.util.spec_from_file_location(
    "gsam.metrics",
    os.path.join(os.path.dirname(__file__), "..", "gsam", "metrics.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_repeated_failure_rate = _mod.compute_repeated_failure_rate
compute_retrieval_precision = _mod.compute_retrieval_precision
compute_concept_coverage = _mod.compute_concept_coverage
compute_transfer_metrics = _mod.compute_transfer_metrics


class TestRepeatedFailureRate(unittest.TestCase):
    """Test RFR computation."""

    def test_empty_history(self):
        result = compute_repeated_failure_rate([])
        self.assertEqual(result["rfr"], 0.0)
        self.assertEqual(result["total_errors"], 0)

    def test_no_errors(self):
        history = [
            {"step": 1, "is_correct": True, "concepts_involved": ["Revenue"]},
            {"step": 2, "is_correct": True, "concepts_involved": ["Expenses"]},
        ]
        result = compute_repeated_failure_rate(history)
        self.assertEqual(result["total_errors"], 0)
        self.assertEqual(result["rfr"], 0.0)

    def test_single_error(self):
        history = [
            {"step": 1, "is_correct": False, "concepts_involved": ["Revenue"],
             "confusion_pairs": []},
        ]
        result = compute_repeated_failure_rate(history)
        self.assertEqual(result["total_errors"], 1)
        self.assertEqual(result["repeated_errors"], 0)

    def test_repeated_errors(self):
        history = [
            {"step": 1, "is_correct": False, "concepts_involved": ["Revenue"],
             "confusion_pairs": []},
            {"step": 2, "is_correct": False, "concepts_involved": ["Revenue"],
             "confusion_pairs": []},
            {"step": 3, "is_correct": False, "concepts_involved": ["Revenue"],
             "confusion_pairs": []},
        ]
        result = compute_repeated_failure_rate(history)
        self.assertEqual(result["total_errors"], 3)
        self.assertEqual(result["repeated_errors"], 2)
        self.assertGreater(result["rfr"], 0)

    def test_different_concept_errors(self):
        history = [
            {"step": 1, "is_correct": False, "concepts_involved": ["Revenue"],
             "confusion_pairs": []},
            {"step": 2, "is_correct": False, "concepts_involved": ["Expenses"],
             "confusion_pairs": []},
        ]
        result = compute_repeated_failure_rate(history)
        self.assertEqual(result["total_errors"], 2)
        self.assertEqual(result["repeated_errors"], 0)
        self.assertEqual(result["unique_error_patterns"], 2)

    def test_confusion_pair_repetition(self):
        history = [
            {"step": 1, "is_correct": False, "concepts_involved": [],
             "confusion_pairs": [["NetIncome", "ComprehensiveIncome"]]},
            {"step": 2, "is_correct": False, "concepts_involved": [],
             "confusion_pairs": [["NetIncome", "ComprehensiveIncome"]]},
        ]
        result = compute_repeated_failure_rate(history)
        self.assertEqual(result["total_errors"], 2)
        self.assertEqual(result["repeated_errors"], 1)


class TestRetrievalPrecision(unittest.TestCase):
    """Test retrieval precision computation."""

    def test_empty_logs(self):
        result = compute_retrieval_precision([])
        self.assertEqual(result["mean_precision"], 0.0)
        self.assertEqual(result["total_tasks"], 0)

    def test_perfect_precision(self):
        logs = [
            {"retrieved_count": 5, "referenced_count": 5, "precision": 1.0,
             "retrieval_time_s": 0.1},
        ]
        result = compute_retrieval_precision(logs)
        self.assertEqual(result["mean_precision"], 1.0)

    def test_mixed_precision(self):
        logs = [
            {"retrieved_count": 10, "referenced_count": 5, "precision": 0.5,
             "retrieval_time_s": 0.1},
            {"retrieved_count": 10, "referenced_count": 8, "precision": 0.8,
             "retrieval_time_s": 0.2},
        ]
        result = compute_retrieval_precision(logs)
        self.assertAlmostEqual(result["mean_precision"], 0.65, places=2)
        self.assertEqual(result["total_tasks"], 2)

    def test_retrieval_time(self):
        logs = [
            {"retrieved_count": 5, "referenced_count": 5, "precision": 1.0,
             "retrieval_time_s": 0.1},
            {"retrieved_count": 5, "referenced_count": 5, "precision": 1.0,
             "retrieval_time_s": 0.3},
        ]
        result = compute_retrieval_precision(logs)
        self.assertAlmostEqual(result["mean_retrieval_time_s"], 0.2, places=2)


class TestConceptCoverage(unittest.TestCase):
    """Test concept coverage computation."""

    def test_from_stats(self):
        stats = {"concept_coverage": 0.75}
        self.assertEqual(compute_concept_coverage(stats), 0.75)

    def test_missing_key(self):
        self.assertEqual(compute_concept_coverage({}), 0.0)


class TestTransferMetrics(unittest.TestCase):
    """Test transfer metric computation."""

    def test_positive_transfer(self):
        result = compute_transfer_metrics(
            source_accuracy=0.8,
            target_accuracy_with_transfer=0.7,
            target_accuracy_without_transfer=0.5,
        )
        self.assertAlmostEqual(result["delta_transfer"], 0.2, places=5)
        self.assertTrue(result["positive_transfer"])
        self.assertFalse(result["negative_transfer"])

    def test_negative_transfer(self):
        result = compute_transfer_metrics(
            source_accuracy=0.8,
            target_accuracy_with_transfer=0.4,
            target_accuracy_without_transfer=0.5,
        )
        self.assertAlmostEqual(result["delta_transfer"], -0.1, places=5)
        self.assertFalse(result["positive_transfer"])
        self.assertTrue(result["negative_transfer"])

    def test_no_transfer(self):
        result = compute_transfer_metrics(
            source_accuracy=0.8,
            target_accuracy_with_transfer=0.5,
            target_accuracy_without_transfer=0.5,
        )
        self.assertEqual(result["delta_transfer"], 0.0)
        self.assertFalse(result["positive_transfer"])
        self.assertFalse(result["negative_transfer"])


if __name__ == "__main__":
    unittest.main()
