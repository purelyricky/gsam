# Contributing

1. Create a virtualenv (Python 3.11+) and install: `pip install -r requirements-lock.txt -r requirements-dev.txt`
2. Copy `.env.example` to `.env` only if you intend to run experiments (tests need no API keys).
3. Run checks before opening a PR:
   ```bash
   ruff check gsam ace eval tests
   python -m unittest discover -s tests -v
   ```
4. Keep each feature or fix in its own small commit together with the tests that pin its behaviour.
   Do not mix formatting, refactors and features in one commit.
5. Use the `gsam` logger (`from log_config import get_logger`) instead of `print()`.
6. New dependencies go in `requirements.txt`; regenerate `requirements-lock.txt` with
   `pip install -r requirements.txt && pip freeze > requirements-lock.txt`.
