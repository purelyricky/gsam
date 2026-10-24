# Changelog

## [Unreleased]
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
