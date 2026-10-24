"""Tests for checkpoint / resume helpers."""

import json
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import resume_utils  # noqa: E402

LOG = logging.getLogger("test_resume")


class TestFindLatestCheckpoint(unittest.TestCase):
    def test_picks_highest_step_with_extension(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("pb_step_2.txt", "pb_step_10.txt", "pb_step_99.json", "notes.txt"):
                open(os.path.join(d, name), "w").close()
            best = resume_utils.find_latest_checkpoint(d, ".txt")
            self.assertEqual(os.path.basename(best), "pb_step_10.txt")

    def test_missing_directory_returns_none(self):
        self.assertIsNone(resume_utils.find_latest_checkpoint("/no/such/dir", ".txt"))


class TestProgress(unittest.TestCase):
    def test_write_and_apply_offline(self):
        with tempfile.TemporaryDirectory() as d:
            resume_utils.write_progress(d, mode="offline", epoch=3, step=7)
            state = resume_utils.apply_progress_files(d, resume_utils.default_resume_state(), LOG)
            self.assertEqual((state["start_epoch"], state["start_step"]), (3, 7))

    def test_online_resumes_at_next_window(self):
        with tempfile.TemporaryDirectory() as d:
            resume_utils.write_progress(d, timestamp="t", mode="online", window=4, global_step=120)
            with open(os.path.join(d, "partial_online_results.json"), "w") as f:
                json.dump({"a": 1}, f)
            state = resume_utils.apply_progress_files(d, resume_utils.default_resume_state(), LOG)
            self.assertEqual(state["start_window"], 5)
            self.assertEqual(state["start_global_step"], 120)
            self.assertEqual(state["prior_online_results"], {"a": 1})

    def test_empty_directory_keeps_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            state = resume_utils.apply_progress_files(d, resume_utils.default_resume_state(), LOG)
            self.assertEqual(state, resume_utils.default_resume_state())


if __name__ == "__main__":
    unittest.main()
