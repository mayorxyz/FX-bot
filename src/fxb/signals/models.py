"""Signal data model — the single contract between strategies and consumers
(backtester, alerting). Frozen dataclass; validated on construction.

Conventions:
- All timestamps are tz-aware UTC. ``ts_signal`` is the CLOSE time of the bar
  that generated the signal (i.e. the moment it becomes known).
- Prices are BID prices (entry/stop/TP for a LONG refer to executions at bid
  or ask per the fill model in ``fxb.backtest.fills``; the Signal itself only
  stores the *level* the trader should act on).
- ``signal_id`` is a deterministic hash of (symbol, timeframe, ts_signal,
  direction, entry_price) so re-running a strategy reproduces identical ids.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from fxb.core.timeutil import to_utc


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


def _require_utc(ts, name: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"{name} must be tz-aware UTC, got naive timestamp {ts!r}")
    return t.tz_convert("UTC") if t.utcoffset().total_seconds() != 0 else t


def compute_signal_id(symbol: str, timeframe: str, ts_signal, direction,
                      entry_price) -> str:
    """Deterministic short hash of the signal identity tuple."""
    payload = json.dumps(
        {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "ts_signal": _require_utc(ts_signal, "ts_signal").isoformat(),
            "direction": Direction(direction).value,
            "entry_price": round(float(entry_price), 8),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Signal:
    symbol: str
    timeframe: str
    ts_signal: pd.Timestamp          # close time of generating bar (UTC)
    direction: Direction
    entry_price: float
    stop_price: float
    take_profit_prices: tuple        # 1..3 price levels (floats), ordered by distance
    entry_type: EntryType = EntryType.MARKET
    expiry_bars: int = 5             # unfilled LIMIT cancelled after N bars
    max_hold_bars: int | None = None
    score: float = 1.0               # 0..1
    reasons: tuple = ()              # tuple[str] of human-readable reasons
    meta: tuple = ()                 # tuple of (key, value) pairs -> read via .meta_dict
    signal_id: str = ""

    def __post_init__(self):
        # --- normalisation (object is frozen, so use object.__setattr__) ---
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "timeframe", self.timeframe.upper())
        object.__setattr__(self, "ts_signal", _require_utc(self.ts_signal, "ts_signal"))
        object.__setattr__(self, "direction", Direction(self.direction))
        object.__setattr__(self, "entry_type", EntryType(self.entry_type))
        object.__setattr__(self, "take_profit_prices",
                           tuple(float(p) for p in self.take_profit_prices))
        object.__setattr__(self, "reasons", tuple(str(r) for r in self.reasons))
        if isinstance(self.meta, dict):
            object.__setattr__(self, "meta",
                               tuple(sorted((str(k), v) for k, v in self.meta.items())))
        else:
            object.__setattr__(self, "meta", tuple(tuple(m) for m in self.meta))

        # --- validation ---
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        for name, p in (("entry_price", self.entry_price), ("stop_price", self.stop_price)):
            if not (float(p) > 0):
                raise ValueError(f"{name} must be > 0, got {p}")
        tps = self.take_profit_prices
        if not (1 <= len(tps) <= 3):
            raise ValueError(f"take_profit_prices must hold 1..3 levels, got {len(tps)}")
        for p in tps:
            if not (float(p) > 0):
                raise ValueError(f"take-profit price must be > 0, got {p}")
        risk = self.risk_distance
        if risk <= 0:
            side = "below" if self.direction is Direction.LONG else "above"
            raise ValueError(
                f"stop must be {side} entry for {self.direction.value}: "
                f"entry={self.entry_price} stop={self.stop_price}"
            )
        for i, p in enumerate(tps):
            rr = self.rr_to_tp(i)
            if rr <= 0:
                side = "above" if self.direction is Direction.LONG else "below"
                raise ValueError(
                    f"take_profit_prices[{i}]={p} on wrong side of entry "
                    f"(must be {side} for {self.direction.value})"
                )
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(f"score must be within [0, 1], got {self.score}")
        if int(self.expiry_bars) < 1:
            raise ValueError(f"expiry_bars must be >= 1, got {self.expiry_bars}")
        if self.max_hold_bars is not None and int(self.max_hold_bars) < 1:
            raise ValueError(f"max_hold_bars must be >= 1 or None, got {self.max_hold_bars}")
        expected = compute_signal_id(self.symbol, self.timeframe, self.ts_signal,
                                     self.direction, self.entry_price)
        if self.signal_id == "":
            object.__setattr__(self, "signal_id", expected)
        elif self.signal_id != expected:
            raise ValueError(
                f"signal_id {self.signal_id!r} does not match deterministic hash "
                f"{expected!r} of (symbol, tf, ts_signal, direction, entry)"
            )

    # ------------------------------------------------------------------ props
    @property
    def risk_distance(self) -> float:
        """Absolute price distance from entry to stop (>0 when valid)."""
        if self.direction is Direction.LONG:
            return float(self.entry_price) - float(self.stop_price)
        return float(self.stop_price) - float(self.entry_price)

    def rr_to_tp(self, i: int = 0) -> float:
        """Reward:Risk ratio to the i-th take-profit level."""
        tp = float(self.take_profit_prices[i])
        reward = (tp - self.entry_price) if self.direction is Direction.LONG \
            else (self.entry_price - tp)
        return reward / self.risk_distance

    @property
    def meta_dict(self) -> dict:
        return dict(self.meta)

    def with_meta(self, **kwargs) -> "Signal":
        """Return a copy with extra meta entries (signal_id preserved)."""
        merged = self.meta_dict | kwargs
        return Signal(
            symbol=self.symbol, timeframe=self.timeframe, ts_signal=self.ts_signal,
            direction=self.direction, entry_price=self.entry_price,
            stop_price=self.stop_price, take_profit_prices=self.take_profit_prices,
            entry_type=self.entry_type, expiry_bars=self.expiry_bars,
            max_hold_bars=self.max_hold_bars, score=self.score,
            reasons=self.reasons, meta=merged, signal_id=self.signal_id,
        )
