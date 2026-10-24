"""
Lightweight validation for experiment config JSON files (experiments/configs/*.json).

Kept dependency-free and import-light so entry-point scripts can validate a config
before any heavy module (torch, sentence-transformers, API clients) is imported.
"""
from typing import Any, Dict

VALID_SYSTEMS = ("ace", "gsam")
VALID_MODES = ("offline", "online", "eval_only")
VALID_PROVIDERS = ("sambanova", "together", "openai", "modal", "mock")
VALID_TASKS = ("finer", "formula")

REQUIRED_STRINGS = ("system", "task_name", "mode", "api_provider",
                    "generator_model", "reflector_model", "curator_model")
POSITIVE_INTS = ("num_epochs", "max_num_rounds", "curator_frequency", "eval_steps", "save_steps",
                 "online_eval_frequency", "max_tokens", "playbook_token_budget", "test_workers",
                 "prune_frequency", "retrieval_depth")
BOOLEANS = ("no_ontology", "no_failure_cascades", "embedding_only_retrieval", "untyped_edges",
            "no_multi_epoch_refinement", "use_bulletpoint_analyzer")


class ConfigError(ValueError):
    """Raised when an experiment config is malformed."""


def validate_config(config: Any) -> Dict[str, Any]:
    """Validate an experiment config dict and return it unchanged; raise ConfigError otherwise."""
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a JSON object, got {type(config).__name__}")

    errors = []
    for key in REQUIRED_STRINGS:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{key}' is required and must be a non-empty string")

    for key, allowed in (("system", VALID_SYSTEMS), ("mode", VALID_MODES),
                         ("api_provider", VALID_PROVIDERS), ("task_name", VALID_TASKS)):
        value = config.get(key)
        if isinstance(value, str) and value and value not in allowed:
            errors.append(f"'{key}' must be one of {list(allowed)}, got '{value}'")

    for key in POSITIVE_INTS:
        if key in config:
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"'{key}' must be a positive integer, got {value!r}")

    if "merge_threshold" in config:
        value = config["merge_threshold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            errors.append(f"'merge_threshold' must be a number in (0, 1], got {value!r}")

    for key in BOOLEANS:
        if key in config and not isinstance(config[key], bool):
            errors.append(f"'{key}' must be true or false, got {config[key]!r}")

    if errors:
        raise ConfigError("Invalid experiment config:\n  - " + "\n  - ".join(errors))
    return config
