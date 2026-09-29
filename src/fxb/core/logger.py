"""Logging for FXB: console + rotating file, ISO UTC timestamps, idempotent.

Usage::

    from fxb.core.logger import get_logger
    log = get_logger(__name__)

Never use ``print()``; never log secrets (use ``Settings.redacted()``).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

from fxb.core.paths import LOGS_DIR

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


class _UtcFormatter(logging.Formatter):
    """Formatter that renders ``asctime`` as ISO-8601 UTC (Z suffix)."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("UTC"))
        return dt.strftime(datefmt or _DATEFMT)


def _configure_root(level: str) -> None:
    root = logging.getLogger("fxb")
    if getattr(root, "_fxb_configured", False):
        # Already configured: only refresh the level, never duplicate handlers.
        root.setLevel(level)
        return
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(_UtcFormatter(_FORMAT, _DATEFMT))
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "fxb.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(_UtcFormatter(_FORMAT, _DATEFMT))
    root.addHandler(file_handler)

    root._fxb_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return the namespaced logger ``fxb.<name>`` (console + rotating file).

    Idempotent: repeated calls never add duplicate handlers. Log level comes
    from the ``LOG_LEVEL`` environment variable (default ``INFO``).
    """
    import os

    _configure_root(os.environ.get("LOG_LEVEL", "INFO").upper())
    short = name.removeprefix("fxb.")
    return logging.getLogger(f"fxb.{short}" if short else "fxb")
