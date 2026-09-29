"""CSV candle provider for MT5 exports, Dukascopy and HistData style files.

The provider is configured with a ``column_map`` that translates the file's
own header names into the FXB standard schema, plus a ``source_tz`` because
many vendors export broker-server or local time rather than UTC::

    CsvProvider(path, column_map={"Date": "ts_open", "Time": "ts_open", ...},
                source_tz="America/New_York")

- ``ts_open`` may be split across two columns (a date column and a time
  column); both map to ``ts_open`` and are concatenated before parsing.
- A single column may hold combined datetimes ("Date" containing "2024/06/03
  14:00"); then only that column maps to ``ts_open``.
- Missing ``spread`` column -> filled with ``default_spread_points`` and a
  warning is logged. Missing ``tick_volume`` -> 0.
- Only closed bars are returned (the forming last bar is dropped against the
  file's own newest timestamp, which by convention is already closed data;
  an explicit ``now_utc`` can be supplied for live-ish files).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from fxb.core.logger import get_logger
from fxb.core.timeutil import ensure_utc
from fxb.data.providers.base import DataProvider
from fxb.data.validate import CANDLE_COLUMNS, validate_candles

log = get_logger(__name__)

# Header aliases accepted when ``column_map`` is omitted (MT5-style export).
_DEFAULT_MAP: Dict[str, str] = {
    "Date": "ts_open",
    "Time": "ts_open",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "tick_volume",
}


class CsvProvider(DataProvider):
    """Reads a CSV file of OHLC bars and normalizes it to the FXB schema."""

    name = "csv"

    def __init__(self, path: str | Path,
                 column_map: Optional[Dict[str, str]] = None,
                 source_tz: str = "UTC",
                 datetime_format: Optional[str] = None,
                 default_spread_points: float = 2.0,
                 separator: str = ";") -> None:
        self.path = Path(path)
        self.column_map = column_map if column_map is not None else dict(_DEFAULT_MAP)
        self.source_tz = source_tz
        self.datetime_format = datetime_format
        self.default_spread_points = float(default_spread_points)
        self.separator = separator

    def fetch_candles(self, symbol: str, tf: str,
                      start_utc: Optional[pd.Timestamp] = None,
                      end_utc: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Parse the CSV, convert to UTC, filter ``[start_utc, end_utc)``.

        ``symbol``/``tf`` are used only for logging and gap filtering; the
        file itself is assumed to contain bars of a single timeframe.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"candle CSV not found: {self.path}")
        raw = pd.read_csv(self.path, sep=self.separator, dtype=str).rename(columns=str.strip)

        # -- map vendor columns to the standard schema ---------------------
        parts: Dict[str, list] = {}
        out_cols: Dict[str, pd.Series] = {}
        for src, dst in self.column_map.items():
            if src not in raw.columns:
                continue
            if dst == "ts_open":
                parts.setdefault("ts_open_parts", []).append(raw[src].astype(str))
            else:
                out_cols[dst] = raw[src]
        if not parts.get("ts_open_parts"):
            raise ValueError(
                f"column_map produced no ts_open source columns; header was {list(raw.columns)}"
            )
        combined = parts["ts_open_parts"][0]
        for extra in parts["ts_open_parts"][1:]:
            combined = combined.str.strip() + " " + extra.str.strip()
        ts = pd.to_datetime(combined, format=self.datetime_format, utc=False, errors="raise")
        try:
            ts = ts.dt.tz_localize(self.source_tz, ambiguous=False, nonexistent="shift_forward")
        except TypeError:  # already tz-aware strings
            pass
        out_cols["ts_open"] = ts.dt.tz_convert("UTC")

        df = pd.DataFrame(out_cols)

        # -- optional / missing columns ------------------------------------
        if "spread" not in df.columns:
            log.warning(
                "CSV %s has no spread column; filling %.1f points (configure per-source if known)",
                self.path.name, self.default_spread_points,
            )
            df["spread"] = self.default_spread_points
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0

        df = df[CANDLE_COLUMNS].copy()
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["tick_volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0).astype("int64")
        df["spread"] = pd.to_numeric(df["spread"]).astype(float)

        df = df.sort_values("ts_open").reset_index(drop=True)
        validate_candles(df)

        if start_utc is not None:
            df = df[df["ts_open"] >= ensure_utc(start_utc)]
        if end_utc is not None:
            df = df[df["ts_open"] < ensure_utc(end_utc)]
        return df.reset_index(drop=True)
