"""Settings and logging conventions for FXB.

- ``get_logger(name)`` returns a module logger with a consistent format.
- ``Settings`` is a plain frozen-ish config object (env-overridable in later
  prompts); the backtest harness uses its own ``BacktestConfig`` dataclass.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    root = logging.getLogger("fxb")
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("FXB_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("fxb") else f"fxb.{name}")


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Prompt 1 live pipeline uses these; the backtester keeps
    its own config so runs stay reproducible."""

    data_dir: str = "data"
    candle_db: str = "data/candles.db"
    backtest_db: str = "data/backtest.db"
    log_level: str = "INFO"


DEFAULT_SETTINGS = Settings()
