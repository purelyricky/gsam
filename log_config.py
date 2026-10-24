"""
Structured logging configuration for GSAM / ACE.

Provides a single named logger hierarchy rooted at ``gsam`` that emits
key=value lines (or JSON when ``GSAM_LOG_FORMAT=json``). Level is controlled
by the ``GSAM_LOG_LEVEL`` environment variable (default ``INFO``).
"""
import json
import logging
import os
import sys

ROOT_LOGGER_NAME = "gsam"


class KeyValueFormatter(logging.Formatter):
    """Render records as ``ts=... level=... logger=... msg="..."``."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage().replace('"', "'")
        line = (
            f'ts={self.formatTime(record, "%Y-%m-%dT%H:%M:%S")} '
            f'level={record.levelname} logger={record.name} msg="{message}"'
        )
        if record.exc_info:
            line += ' exc="' + self.formatException(record.exc_info).replace('"', "'") + '"'
        return line


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level=None, fmt=None, stream=None) -> logging.Logger:
    """Configure (idempotently) and return the root ``gsam`` logger."""
    level_name = (level or os.environ.get("GSAM_LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.environ.get("GSAM_LOG_FORMAT", "kv")).lower()

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(getattr(logging, level_name, logging.INFO))
    root.propagate = False

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else KeyValueFormatter())
    root.addHandler(handler)
    return root


def get_logger(name: str = None) -> logging.Logger:
    """Return a child of the ``gsam`` logger, configuring it on first use."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not root.handlers:
        configure_logging()
    if not name or name == ROOT_LOGGER_NAME:
        return root
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
