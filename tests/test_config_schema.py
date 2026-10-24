"""Tests for experiment config validation."""

import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_schema import ConfigError, validate_config  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "configs")


def good():
    return {
        "system": "gsam", "task_name": "finer", "mode": "online", "api_provider": "mock",
        "generator_model": "m", "reflector_model": "m", "curator_model": "m",
        "num_epochs": 1, "merge_threshold": 0.9, "no_ontology": False,
    }


class TestConfigSchema(unittest.TestCase):
    def test_valid_config_passes(self):
        cfg = good()
        self.assertIs(validate_config(cfg), cfg)

    def test_all_shipped_configs_are_valid(self):
        paths = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
        self.assertTrue(paths)
        for path in paths:
            with open(path) as f:
                validate_config(json.load(f))

    def test_negative_epochs_rejected(self):
        cfg = good()
        cfg["num_epochs"] = -1
        with self.assertRaisesRegex(ConfigError, "num_epochs"):
            validate_config(cfg)

    def test_missing_required_field_rejected(self):
        cfg = good()
        del cfg["generator_model"]
        with self.assertRaisesRegex(ConfigError, "generator_model"):
            validate_config(cfg)

    def test_unknown_mode_rejected(self):
        cfg = good()
        cfg["mode"] = "sideways"
        with self.assertRaisesRegex(ConfigError, "mode"):
            validate_config(cfg)

    def test_bad_threshold_and_bool_flags_rejected(self):
        cfg = good()
        cfg["merge_threshold"] = 1.5
        cfg["no_ontology"] = "yes"
        with self.assertRaises(ConfigError) as ctx:
            validate_config(cfg)
        self.assertIn("merge_threshold", str(ctx.exception))
        self.assertIn("no_ontology", str(ctx.exception))

    def test_bool_is_not_an_int(self):
        cfg = good()
        cfg["num_epochs"] = True
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_non_object_rejected(self):
        with self.assertRaises(ConfigError):
            validate_config([1, 2])

    def test_config_error_is_value_error(self):
        self.assertTrue(issubclass(ConfigError, ValueError))


if __name__ == "__main__":
    unittest.main()
