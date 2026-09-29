"""Candle data providers (prompt 1 conventions).

A provider exposes ``get_candles(symbol, timeframe) -> DataFrame`` returning the
FULL historical candle frame for a symbol/timeframe (columns per
``fxb.core.frames.CANDLE_COLUMNS``). Backtests and strategies must never mutate
these frames; the MarketContext hands out bounded copies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CandleProvider(ABC):
    """Read-only source of candle frames."""

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        ...

    def has(self, symbol: str, timeframe: str) -> bool:
        try:
            return len(self.get_candles(symbol, timeframe)) > 0
        except Exception:
            return False


class DictProvider(CandleProvider):
    """In-memory provider keyed by (symbol, timeframe). Used by tests/backtests."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]):
        self._frames = {k: v.copy() for k, v in frames.items()}

    def get_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol.upper(), timeframe.upper())
        if key not in self._frames:
            raise KeyError(f"No candles for {key}")
        return self._frames[key].copy()
