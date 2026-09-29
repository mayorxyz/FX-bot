"""Canonical filesystem locations for FXB.

Everything is resolved relative to this file so the package works from any
working directory: ``PROJECT_ROOT`` is the repository root (the parent of
``src/``). ``DATA_DIR`` and ``LOGS_DIR`` are created on first import because
the store and logger write into them immediately.
"""
from __future__ import annotations

from pathlib import Path

#: Repository root: .../fxb  (parent of the ``src`` directory).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

#: Source tree root (used by tests/tooling).
SRC_DIR: Path = PROJECT_ROOT / "src"

#: All runtime data lives here (SQLite databases, exports...). Never commit it.
DATA_DIR: Path = PROJECT_ROOT / "data"

#: Rotating log files live here. Never commit it.
LOGS_DIR: Path = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
