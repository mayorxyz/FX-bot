"""Strategy base class and the *bounded* MarketContext.

Look-ahead safety contract:
- ``MarketContext`` is constructed for one primary-timeframe bar close ("now").
  It stores **sliced copies** of candle frames containing ONLY bars that are
  fully closed at or before ``now``. For higher timeframes it uses prompt 1's
  ``last_closed_htf_index`` rule so a partial HTF bar is never exposed.
- Strategies receive no other data channel from the harness; to see a future
  bar they would have to smuggle a reference themselves (which
  ``fxb.backtest.audit`` is designed to catch).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from fxb.core.config import get_logger
from fxb.core.frames import last_closed_htf_index
from fxb.core.timeutil import tf_minutes, to_utc
from fxb.data.providers import CandleProvider
from fxb.signals.models import Signal

log = get_logger("signals.strategy_base")


class MarketContext:
    """Read-only, point-in-time view of market data at ``now``.

    ``now`` is the CLOSE time of the last closed primary bar. All frames handed
    out are private copies truncated at that instant.
    """

    __slots__ = ("_primary_tf", "_now", "_frames", "_symbols")

    def __init__(self, provider: CandleProvider, symbols, primary_tf: str, now):
        self._primary_tf = primary_tf.upper()
        self._now = to_utc(now)
        self._symbols = tuple(s.upper() for s in symbols)
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}
        # Build bounded slices eagerly: structurally impossible to touch futures.
        for sym in self._symbols:
            try:
                full = provider.get_candles(sym, self._primary_tf)
            except KeyError:
                continue
            sl = full[full["ts_open"] <= self._now - _tf_td(self._primary_tf)].copy()
            self._frames[(sym, self._primary_tf)] = sl.reset_index(drop=True)

    # ------------------------------------------------------------------ api
    @property
    def now(self) -> pd.Timestamp:
        return self._now

    @property
    def primary_timeframe(self) -> str:
        return self._primary_tf

    @property
    def symbols(self) -> tuple:
        return self._symbols

    def candles(self, symbol: str, timeframe: str | None = None, n: int | None = None) -> pd.DataFrame:
        """Last ``n`` CLOSED bars at or before ``now`` (copy).

        - ``timeframe`` defaults to the primary timeframe.
        - If ``timeframe`` != primary, only fully-closed HTF bars are returned
          (``last_closed_htf_index`` rule); LTF data is NOT resampled on the fly
          — the provider must supply HTF frames (prompt 1 resampler does this).
        """
        sym = symbol.upper()
        tf = (timeframe or self._primary_tf).upper()
        key = (sym, tf)
        if key not in self._frames:
            raise KeyError(
                f"MarketContext has no frame for {key}; declare it in "
                f"Strategy.required_timeframes"
            )
        df = self._frames[key]
        if n is not None:
            n = int(n)
            if n <= 0:
                return df.iloc[0:0].copy()
            df = df.iloc[-n:]
        return df.copy()

    def last_bar(self, symbol: str, timeframe: str | None = None):
        df = self.candles(symbol, timeframe)
        return None if len(df) == 0 else df.iloc[-1]


def _tf_td(tf: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=tf_minutes(tf))


# Rebuild HTF slices lazily per context: strategies may request extra tfs via
# required_timeframes; the engine pre-seeds them using this helper.
def build_context(provider: CandleProvider, symbols, primary_tf: str, now,
                  extra_tfs=()) -> MarketContext:
    ctx = MarketContext(provider, symbols, primary_tf, now)
    # attach HTF bounded frames using the closed-HTF rule
    for sym in ctx.symbols:
        for tf in extra_tfs:
            tf = tf.upper()
            if tf == primary_tf.upper() or (sym, tf) in ctx._frames:
                continue
            try:
                full = provider.get_candles(sym, tf)
            except KeyError:
                continue
            idx = last_closed_htf_index(full, tf, now)
            sliced = full.iloc[: idx + 1].copy() if idx >= 0 else full.iloc[0:0].copy()
            ctx._frames[(sym, tf)] = sliced.reset_index(drop=True)
    return ctx


class Strategy(ABC):
    """Base class for signal generators.

    Subclasses declare ``name``, the timeframes they need (primary plus any
    HTFs, which the engine pre-slices into the context) and a warmup length.
    ``on_bar`` is called once per closed primary bar with a bounded context and
    must return a (possibly empty) list of Signals. It must be a pure function
    of the context: identical contexts -> identical signals (the audit relies
    on this).
    """

    name: str = "unnamed"
    #: primary timeframe first, then HTFs, e.g. ("M15", "H1")
    required_timeframes: tuple = ()
    #: minimum number of closed primary bars needed before on_bar is useful
    warmup_bars: int = 1

    @abstractmethod
    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        """Generate signals for the just-closed bar ``ctx.now``."""
        raise NotImplementedError

    # convenience -------------------------------------------------------
    @property
    def primary_timeframe(self) -> str:
        if not self.required_timeframes:
            raise ValueError(f"{self.name}: required_timeframes not set")
        return self.required_timeframes[0]

    @property
    def htf_timeframes(self) -> tuple:
        return tuple(self.required_timeframes[1:])

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Strategy {self.name} tfs={self.required_timeframes}>"
