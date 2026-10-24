"""Tests for the structured logging configuration."""

import io
import json
import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import log_config  # noqa: E402


class TestLogConfig(unittest.TestCase):
    def tearDown(self):
        log_config.configure_logging(level="INFO", fmt="kv")

    def test_key_value_output_and_level(self):
        stream = io.StringIO()
        log_config.configure_logging(level="WARNING", fmt="kv", stream=stream)
        log = log_config.get_logger("unit")
        log.info("hidden")
        log.warning("shown")
        out = stream.getvalue()
        self.assertNotIn("hidden", out)
        self.assertIn("level=WARNING", out)
        self.assertIn('logger=gsam.unit msg="shown"', out)

    def test_json_output(self):
        stream = io.StringIO()
        log_config.configure_logging(level="INFO", fmt="json", stream=stream)
        log_config.get_logger("unit").info("hello %s", "world")
        record = json.loads(stream.getvalue().strip())
        self.assertEqual(record["msg"], "hello world")
        self.assertEqual(record["level"], "INFO")

    def test_configure_is_idempotent(self):
        log_config.configure_logging()
        log_config.configure_logging()
        self.assertEqual(len(logging.getLogger("gsam").handlers), 1)


if __name__ == "__main__":
    unittest.main()
