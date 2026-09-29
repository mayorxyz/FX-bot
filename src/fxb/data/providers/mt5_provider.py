"""MetaTrader5 candle provider (Windows-only, lazy import).

``MetaTrader5`` is imported INSIDE methods only — importing this module on
any platform is safe. Bar times coming from MT5 are in *server* time; the
configured ``MT5_SERVER_UTC_OFFSET_HOURS`` is subtracted to recover true UTC
(server_time = UTC + offset), and the still-forming last bar is dropped.

The terminal connection details come from :class:`fxb.core.config.Settings`
(never logged). For offline testing a fake module object can be injected via
the constructor (``Mt5Provider(mt5=<fake>)``).
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, Optional

import pandas as pd

from fxb.core.config import Settings, load_settings
from fxb.core.logger import get_logger
from fxb.core.timeutil import ensure_utc, timeframe_to_timedelta
from fxb.data.providers.base import DataProvider
from fxb.data.validate import CANDLE_COLUMNS, validate_candles

log = get_logger(__name__)

_TF_ATTR = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


class Mt5UnavailableError(RuntimeError):
    """Raised when the MetaTrader5 package/terminal is not available."""


def _import_mt5() -> ModuleType:
    try:
        import MetaTrader5 as mt5  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise Mt5UnavailableError(
            "MetaTrader5 package is not installed (pip install MetaTrader5; "
            "Windows only). Use CsvProvider or a stored dataset otherwise."
        ) from exc
    return mt5


class Mt5Provider(DataProvider):
    """Fetches closed candles from a running MetaTrader 5 terminal."""

    name = "mt5"

    def __init__(self, settings: Optional[Settings] = None,
                 mt5: Optional[ModuleType] = None) -> None:
        #: Injected fake module for tests; real import happens lazily otherwise.
        self._mt5_override = mt5
        self.settings = settings or load_settings()
        self._initialized = False

    def _mod(self) -> ModuleType:
        if self._mt5_override is not None:
            return self._mt5_override
        if sys.platform != "win32":
            raise Mt5UnavailableError(
                f"MetaTrader5 requires Windows (current platform: {sys.platform})"
            )
        return _import_mt5()

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> None:
        """Connect to the terminal using credentials from Settings."""
        mt5 = self._mod()
        kwargs: Dict[str, Any] = {}
        if self.settings.mt5_login:
            kwargs.update(
                login=int(self.settings.mt5_login),
                password=self.settings.mt5_password or "",
                server=self.settings.mt5_server or "",
            )
        if not mt5.initialize(**kwargs):
            raise Mt5UnavailableError(f"MT5 initialize() failed: {mt5.last_error()}")
        self._initialized = True
        log.info("MT5 terminal initialized (server=%s)", self.settings.mt5_server or "<default>")

    def shutdown(self) -> None:
        """Disconnect from the terminal (safe to call twice)."""
        if self._initialized:
            self._mod().shutdown()
            self._initialized = False

    def __enter__(self) -> "Mt5Provider":
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # -- data ---------------------------------------------------------------
    def fetch_candles(self, symbol: str, tf: str,
                      start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
        """Return closed BID candles with ``start <= ts_open < end`` in UTC.

        Server-time bar stamps are shifted by ``-MT5_SERVER_UTC_OFFSET_HOURS``
        to obtain true UTC; the unclosed forming bar is removed.
        """
        if not self._initialized:
            self.initialize()
        mt5 = self._mod()
        key = tf.strip().upper()
        if key not in _TF_ATTR:
            raise ValueError(f"unsupported timeframe {tf!r}")
        tf_const = getattr(mt5, _TF_ATTR[key])
        # MT5 stamps bars with *server-time* values encoded as epoch seconds.
        # Request a range padded by 1 day on each side so that, after the
        # server->UTC shift below, the full [start, end) span is covered even
        # for extreme offsets.
        pad = pd.Timedelta(days=1)
        req_start = (ensure_utc(start_utc) - pad).to_pydatetime()
        req_end = (ensure_utc(end_utc) + pad).to_pydatetime()
        rates = mt5.copy_rates_range(symbol.upper(), tf_const, req_start, req_end)
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed for {symbol} {tf}: {mt5.last_error()}")
        if len(rates) == 0:
            empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDLE_COLUMNS})
            empty["ts_open"] = pd.to_datetime([], utc=True)
            return empty[CANDLE_COLUMNS]

        df = pd.DataFrame([dict(r) for r in rates])
        # Server time = UTC + offset  =>  true_utc = stamp - offset.
        offset = pd.Timedelta(hours=float(self.settings.mt5_server_utc_offset_hours))
        naive = pd.to_datetime(df["time"], unit="s", utc=True)
        df["ts_open"] = (naive - offset).tz_convert("UTC")
        df = df.rename(columns={"tick_volume": "tick_volume"})
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["tick_volume"] = pd.to_numeric(df["tick_volume"]).astype("int64")
        df["spread"] = pd.to_numeric(df["spread"]).astype(float)
        df = df[CANDLE_COLUMNS].sort_values("ts_open").reset_index(drop=True)

        # Filter to the requested window after the correction.
        start_ts, end_ts = ensure_utc(start_utc), ensure_utc(end_utc)
        df = df[(df["ts_open"] >= start_ts) & (df["ts_open"] < end_ts)].reset_index(drop=True)

        # Drop the still-forming bar: closed iff now >= ts_open + duration.
        now = pd.Timestamp.now(tz="UTC")
        dur = pd.Timedelta(timeframe_to_timedelta(key))
        df = df[df["ts_open"] + dur <= now].reset_index(drop=True)
        validate_candles(df)
        return df

    def symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Broker metadata used later with ``instruments.override_spec``."""
        if not self._initialized:
            self.initialize()
        info = self._mod().symbol_info_get(symbol.upper())
        if info is None:
            raise RuntimeError(f"symbol_info_get({symbol}) failed: {self._mod().last_error()}")
        return {
            "point": float(info.point),
            "digits": int(info.digits),
            "contract_size": float(info.trade_contract_size),
            "volume_min": float(info.volume_min),
            "volume_step": float(info.volume_step),
            "volume_max": float(info.volume_max),
        }
