"""Abstract data provider interface.

Every provider returns ONLY fully closed bars with tz-aware UTC ``ts_open``
and the standard candle schema (see :mod:`fxb.data.validate`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Source of historical candles for one symbol/timeframe at a time."""

    #: Human-readable provider name (used in logs).
    name: str = "provider"

    @abstractmethod
    def fetch_candles(self, symbol: str, tf: str,
                      start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
        """Return closed candles with ``start_utc <= ts_open < end_utc``.

        Columns follow :data:`fxb.data.validate.CANDLE_COLUMNS`; timestamps
        are tz-aware UTC bar OPEN times. Implementations must never return a
        bar that has not closed yet.
        """
        raise NotImplementedError
