# Changelog

## [Unreleased]
### Added
- Offline `mock` API provider (`mock_llm.py`) and end-to-end GSAM/ACE tests that need no API keys.
- `config_schema.py`: experiment configs are validated before a run is launched.
- `experiment_tracker.py`: per-run `tracking.jsonl` plus optional wandb/mlflow mirroring (`GSAM_TRACKER`).
- `uv.lock`, Dependabot, mypy and `pip check` in CI, 60% coverage gate.

### Changed
- Oversized modules split into focused files (`ace/online_training.py`, `ace/sample_training.py`,
  `gsam/training_loops.py`, `gsam/sample_training.py`, `gsam/graph_*.py`, `gsam/core/reflection_extraction.py`).
- Star imports, unused imports/variables and bare excepts removed; pyflakes rules enforced by ruff.

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff lint, unit tests with coverage.
- `requirements-lock.txt` pinned dependency set and `requirements-dev.txt`.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` for one-command startup.
- `pyproject.toml` with ruff and coverage configuration.
- `log_config.py`: structured (key=value / JSON) logging via `GSAM_LOG_LEVEL` / `GSAM_LOG_FORMAT`.
- Tests for `GraphRetriever`, `GraphConstructor`, the generator pipeline and logging.
- `.env.example`, `CONTRIBUTING.md`, `REPRODUCIBILITY.md`.

### Changed
- `print()` calls in `ace/`, `gsam/`, `playbook_utils.py` and `logger.py` replaced by the `gsam` logger.

## [0.1.0]
- Initial GSAM implementation, ACE baseline and FiNER / XBRL Formula evaluation.
