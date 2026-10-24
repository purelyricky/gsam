"""Tests for experiment tracking."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiment_tracker import ExperimentTracker, flatten_metrics  # noqa: E402


class TestFlattenMetrics(unittest.TestCase):
    def test_nested_numeric_leaves_only(self):
        data = {"a": {"b": 1, "c": "text", "d": {"e": 2.5}}, "flag": True, "n": 3}
        self.assertEqual(flatten_metrics(data), {"a.b": 1.0, "a.d.e": 2.5, "n": 3.0})

    def test_non_dict_returns_empty(self):
        self.assertEqual(flatten_metrics("x"), {})


class TestExperimentTracker(unittest.TestCase):
    def test_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ExperimentTracker(tmp, "run1", backend="")
            tracker.log_params({"mode": "online"})
            flat = tracker.log_metrics({"test": {"accuracy": 0.5}}, step=3)
            tracker.finish()
            self.assertEqual(flat, {"test.accuracy": 0.5})
            with open(os.path.join(tmp, "tracking.jsonl")) as f:
                records = [json.loads(line) for line in f]
            self.assertEqual([r["type"] for r in records], ["params", "metrics"])
            self.assertEqual(records[1]["step"], 3)
            self.assertEqual(records[0]["run"], "run1")

    def test_missing_remote_backend_falls_back_to_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"wandb": None}):
                tracker = ExperimentTracker(tmp, "run2", backend="wandb")
            self.assertIsNone(tracker._remote)
            tracker.log_metrics({"x": 1})
            self.assertTrue(os.path.exists(os.path.join(tmp, "tracking.jsonl")))

    def test_remote_backend_receives_metrics(self):
        fake = mock.MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"wandb": fake}):
                tracker = ExperimentTracker(tmp, "run3", backend="wandb")
                tracker.log_metrics({"acc": 0.9}, step=1)
                tracker.finish()
        fake.init.return_value.log.assert_called_once_with({"acc": 0.9}, step=1)
        fake.init.return_value.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
