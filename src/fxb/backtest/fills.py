"""Execution model for a human trader following FXB signals.

Data conventions (prompt 1): bars carry BID OHLC and ``spread`` in POINTS.
``ask = bid + spread * point``. Exit valuation is on BID levels except SHORT
stops, which trigger against ASK (the trader buys back at ask).

Fill rules (hand-verified in tests/test_fills.py):

Entries
- MARKET: fill at the OPEN of the first actionable bar — the bar AFTER the
  signal bar; never the signal bar itself — at ask for LONG / bid for SHORT,
  plus adverse ``entry_slippage_pips_market`` slippage.
- LIMIT: fills on the first actionable bar whose range crosses the level
  (LONG: ask_low <= entry; SHORT: bid_high >= entry). Fill price = limit
  price, unless the bar GAP-THROUGHS the level (opens already beyond it),
  then the fill happens at the gap open (worse side only). Default no slip.
- A pending LIMIT not filled within ``signal.expiry_bars`` actionable bars is
  cancelled with reason EXPIRED (a Trade row is still emitted with r = 0).

Exits
- LONG stop triggers when bid_low <= stop; LONG TP when bid_high >= tp.
- SHORT stop triggers when ask_high >= stop; SHORT TP when ask_low <= tp.
- Gap through STOP -> fill at bar open (worse). Gap through TP -> fill AT the
  TP price (resting take-profit assumed placed before the gap; documented).
- SAME-BAR AMBIGUITY: if one bar touches both stop and TP, the STOP is assumed
  hit FIRST (conservative). This also applies on the entry/fill bar.
- Partial TPs: default fractions are (1.0,) for one TP, (0.5, 0.5) for two,
  (0.4, 0.3, 0.3) for three. When ``move_stop_to_breakeven_after_tp1`` is set,
  hitting TP1 moves the stop to the entry fill from the NEXT bar onward (never
  retroactive within the same bar).
- ``max_hold_bars``: close at that bar's CLOSE (bid for LONG / ask for SHORT),
  reason TIME. End of data force-closes open trades at the last bar's close
  with reason END_OF_DATA.

Costs converted to R on a 1-lot basis:
- commission per round turn (default 7 USD) split half/half between legs;
  pip value per lot on a 1-lot standard basis is contract_size * pip_size /
  pip_size-pip = 10 USD/pip, so commission_r = (c/2) / (risk_pips * 10)/leg.
- adverse slippage pips on market entries and on stop exits.
- cost_r = commission_r + entry_slip_r + realized exit_slip_r; net = gross - cost.
- MAE/MFE measured from the entry FILL price to intra-bar extremes (adverse leg
  uses the price actually traded against: bid for LONG, ask for SHORT).

Weekends
- Bars whose open instant falls while the market is closed (per timeutil) do
  not trigger fills. Trades whose holding interval spans the Friday-22:00 ->
  Sunday-22:00 UTC gap are flagged ``weekend_hold = True``.

Timestamps: ts_entry / ts_exit are the OPEN time (UTC) of the bar on which the
fill occurred. Session tag by UTC hour of ts_entry: ASIA [0,7), LONDON [7,12),
OVERLAP [12,17), NEWYORK [17,22), CLOSED otherwise (documented approximation).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import pandas as pd

from fxb.core.instruments import get_instrument
from fxb.core.timeutil import crosses_weekend, is_market_open, session_of, tf_to_timedelta
from fxb.signals.models import Direction, EntryType, Signal


class ExitReason(str, Enum):
    STOP = "STOP"
    TP1 = "TP1"
    TP2 = "TP2"
    TP3 = "TP3"
    TIME = "TIME"
    EXPIRED = "EXPIRED"
    END_OF_DATA = "END_OF_DATA"


@dataclass(frozen=True)
class FillConfig:
    commission_per_lot_round_turn: float = 7.0   # account currency (USD), round turn
    entry_slippage_pips_limit: float = 0.0
    entry_slippage_pips_market: float = 0.3
    stop_exit_slippage_pips: float = 0.3
    move_stop_to_breakeven_after_tp1: bool = False
    respect_market_closed: bool = True           # skip fills on closed-market bars

    @staticmethod
    def from_dict(d: dict) -> "FillConfig":
        keys = {f.name for f in fields(FillConfig)}
        return FillConfig(**{k: v for k, v in d.items() if k in keys})


DEFAULT_FILL_CONFIG = FillConfig()

#: standard-lot pip value per 1.0 lot (USD for majors quoted vs USD-like logic):
#: contract_size(100000) * pip_size / 1 pip = 10 per pip per lot.
_LOT_PIP_VALUE = 10.0

_EPS = 1e-12


def default_fractions(n_tps: int) -> tuple:
    """Default partial-close fractions: 1 TP -> (1.0,), 2 -> (.5,.5), 3 -> (.4,.3,.3)."""
    if n_tps == 1:
        return (1.0,)
    if n_tps == 2:
        return (0.5, 0.5)
    return (0.4, 0.3, 0.3)


TRADE_COLUMNS = [
    "signal_id", "symbol", "direction", "ts_signal", "ts_entry", "entry_fill",
    "ts_exit", "exit_price", "exit_reason", "r_multiple_gross", "cost_r",
    "r_multiple_net", "bars_held", "mae_r", "mfe_r", "session",
    "weekend_hold", "closed", "fraction_remaining",
]


@dataclass
class Trade:
    """Mutable accumulator for one simulated position; emits a dict via to_row()."""
    signal: Signal
    ts_entry: pd.Timestamp | None = None
    entry_fill: float | None = None
    ts_exit: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    r_gross: float = 0.0
    cost_r: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0
    weekend_hold: bool = False
    closed: bool = False
    # internals for partial / breakeven management
    stop: float = 0.0
    risk_distance: float = 0.0
    fractions: tuple = (1.0,)
    next_tp: int = 0
    realized_gross: float = 0.0
    entry_leg_cost_r: float = 0.0
    realized_exit_slip_pips: float = 0.0
    action_bars_seen: int = 0

    def to_row(self) -> dict:
        s = self.signal
        return {
            "signal_id": s.signal_id,
            "symbol": s.symbol,
            "direction": s.direction.value,
            "ts_signal": s.ts_signal,
            "ts_entry": self.ts_entry,
            "entry_fill": self.entry_fill,
            "ts_exit": self.ts_exit,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "r_multiple_gross": round(self.r_gross, 10),
            "cost_r": round(self.cost_r, 10),
            "r_multiple_net": round(self.r_gross - self.cost_r, 10),
            "bars_held": self.bars_held,
            "mae_r": round(self.mae_r, 10),
            "mfe_r": round(self.mfe_r, 10),
            "session": session_of(self.ts_entry) if self.ts_entry is not None else None,
            "weekend_hold": bool(self.weekend_hold),
            "closed": bool(self.closed),
            "fraction_remaining": 0.0 if self.closed else float(
                1.0 - sum(self.fractions[: self.next_tp])),
        }


# --------------------------------------------------------------------- helpers
def _pip(symbol: str) -> float:
    return get_instrument(symbol).pip_size


def _point(symbol: str) -> float:
    return get_instrument(symbol).point


def _spread_price(symbol: str, spread_points) -> float:
    return float(spread_points) * _point(symbol)


def _round(x: float) -> float:
    """Round to 8 decimals so results are byte-stable across runs."""
    return round(float(x), 8)


def _bar_is_tradable(row, timeframe: str) -> bool:
    o = pd.Timestamp(row["ts_open"])
    return is_market_open(o) or is_market_open(o + tf_to_timedelta(timeframe))


# --------------------------------------------------------------------- engine
class PositionSimulator:
    """Steps one Trade through primary-timeframe bars, one bar per call.

    The engine creates a simulator for each accepted signal and calls
    ``process_bar(bar)`` for every actionable bar (strictly after the signal
    bar). ``process_bar`` returns the Trade when it becomes final (filled-and-
    closed, or expired); ``finalize(last_bar)`` closes anything still open at
    end of data.
    """

    def __init__(self, trade: Trade, config: FillConfig = DEFAULT_FILL_CONFIG,
                 timeframe: str = "H1"):
        self.trade = trade
        self.cfg = config
        self.tf = timeframe.upper()
        self._be_pending = False     # TP1 hit this bar -> arm breakeven next bar

    # cost helpers -------------------------------------------------------
    def _risk_pips(self) -> float:
        return self.trade.risk_distance / _pip(self.trade.signal.symbol)

    def _commission_half_r(self) -> float:
        return (self.cfg.commission_per_lot_round_turn / 2.0) / (
            _LOT_PIP_VALUE * self._risk_pips())

    def _pips_to_r(self, pips: float) -> float:
        return pips / self._risk_pips()

    # main step ----------------------------------------------------------
    def process_bar(self, row) -> "Trade | None":
        t = self.trade
        if t.closed:
            return t
        if self.cfg.respect_market_closed and not _bar_is_tradable(row, self.tf):
            return None  # no fills while the market is closed
        if t.ts_entry is None:
            return self._try_enter(row)
        # regular management bar: arm pending breakeven BEFORE evaluating exits
        self._arm_breakeven_if_pending()
        self._update_extremes(row)
        return self._manage(row)

    # entry --------------------------------------------------------------
    def _try_enter(self, row) -> "Trade | None":
        t, cfg = self.trade, self.cfg
        sig = t.signal
        t.action_bars_seen += 1
        fill_price = None
        slip_pips = 0.0
        sp = _spread_price(sig.symbol, row["spread"])

        if sig.entry_type is EntryType.MARKET:
            slip_pips = cfg.entry_slippage_pips_market
            if sig.direction is Direction.LONG:
                fill_price = row["open"] + sp + slip_pips * _pip(sig.symbol)
            else:
                fill_price = row["open"] - slip_pips * _pip(sig.symbol)
        else:  # LIMIT
            if sig.direction is Direction.LONG:
                ask_low = row["low"] + sp
                if ask_low <= sig.entry_price + _EPS:
                    if row["open"] + sp < sig.entry_price - _EPS:
                        fill_price = row["open"] + sp   # gap down through buy limit
                    else:
                        fill_price = sig.entry_price
            else:
                if row["high"] >= sig.entry_price - _EPS:
                    if row["open"] > sig.entry_price + _EPS:
                        fill_price = row["open"]        # gap up through sell limit
                    else:
                        fill_price = sig.entry_price
            if fill_price is None:  # not crossed this bar
                if t.action_bars_seen >= int(sig.expiry_bars):
                    return self._close(row, ExitReason.EXPIRED, None, 0.0, 0.0)
                return None
            slip_pips = cfg.entry_slippage_pips_limit

        t.ts_entry = pd.Timestamp(row["ts_open"])
        t.entry_fill = _round(fill_price)
        t.stop = float(sig.stop_price)
        t.risk_distance = sig.risk_distance
        t.fractions = default_fractions(len(sig.take_profit_prices))
        t.entry_leg_cost_r = self._commission_half_r() + self._pips_to_r(slip_pips)
        # The entry bar can also exit: conservative stop-first rule applies.
        return self._manage(row, entry_bar=True)

    # exits ---------------------------------------------------------------
    def _update_extremes(self, row):
        t, sig = self.trade, self.trade.signal
        rd = t.risk_distance
        if rd <= 0:
            return
        if sig.direction is Direction.LONG:
            mae = (t.entry_fill - row["low"]) / rd
            mfe = (row["high"] - t.entry_fill) / rd
        else:
            ask_high = row["high"] + _spread_price(sig.symbol, row["spread"])
            mae = (ask_high - t.entry_fill) / rd
            mfe = (t.entry_fill - row["low"]) / rd
        t.mae_r = min(t.mae_r, _round(mae))
        t.mfe_r = max(t.mfe_r, _round(mfe))

    def _manage(self, row, entry_bar: bool = False) -> "Trade | None":
        t, cfg = self.trade, self.cfg
        sig = t.signal
        long = sig.direction is Direction.LONG
        pip = _pip(sig.symbol)
        rd = t.risk_distance
        sp = _spread_price(sig.symbol, row["spread"])

        # ---- stop touch (conservative: evaluated first) ----
        if long:
            if row["low"] <= t.stop + _EPS:
                trigger = min(row["open"], t.stop)      # gap => worse open
                exec_price = trigger - cfg.stop_exit_slippage_pips * pip
                return self._close(row, ExitReason.STOP, exec_price,
                                   (exec_price - t.entry_fill) / rd,
                                   cfg.stop_exit_slippage_pips)
        else:
            ask_high = row["high"] + sp
            if ask_high >= t.stop - _EPS:
                trigger = max(row["open"] + sp, t.stop)  # gap => worse open
                exec_price = trigger + cfg.stop_exit_slippage_pips * pip
                return self._close(row, ExitReason.STOP, exec_price,
                                   (t.entry_fill - exec_price) / rd,
                                   cfg.stop_exit_slippage_pips)

        # ---- take-profit touches ----
        while t.next_tp < len(sig.take_profit_prices):
            tp = float(sig.take_profit_prices[t.next_tp])
            touched = (long and row["high"] >= tp - _EPS) or \
                      ((not long) and row["low"] <= tp + _EPS)
            if not touched:
                break
            frac = t.fractions[t.next_tp]
            leg_r = ((tp - t.entry_fill) if long else (t.entry_fill - tp)) / rd
            t.realized_gross += leg_r * frac
            t.next_tp += 1
            if cfg.move_stop_to_breakeven_after_tp1 and t.next_tp == 1:
                self._be_pending = True
            if t.next_tp >= len(sig.take_profit_prices):
                return self._close(row, ExitReason(f"TP{t.next_tp}"), tp,
                                   0.0, 0.0, add_realized=True)
        if entry_bar:
            # TP1 taken on the entry bar: breakeven arms right after this bar
            self._arm_breakeven_if_pending()

        # ---- time stop ----
        if sig.max_hold_bars is not None and t.action_bars_seen >= int(sig.max_hold_bars):
            px = row["close"] if long else row["close"] + sp
            leg_r = ((px - t.entry_fill) if long else (t.entry_fill - px)) / rd
            return self._close(row, ExitReason.TIME, px, leg_r, 0.0)
        return None

    def _arm_breakeven_if_pending(self):
        if self._be_pending and self.trade.next_tp >= 1:
            be = self.trade.entry_fill
            if self.trade.signal.direction is Direction.LONG:
                self.trade.stop = max(self.trade.stop, be)
            else:
                self.trade.stop = min(self.trade.stop, be)
            self._be_pending = False

    def _close(self, row, reason: ExitReason, price, leg_r: float,
               slip_pips: float, add_realized: bool = False) -> Trade:
        """Mark the trade closed and return it (final row producer)."""
        t = self.trade
        if not add_realized:
            frac_left = 1.0 - sum(t.fractions[: t.next_tp])
            t.realized_gross += leg_r * frac_left
        if reason is ExitReason.STOP:
            t.realized_exit_slip_pips += slip_pips * (
                1.0 - sum(t.fractions[: t.next_tp]))
        t.r_gross = _round(t.realized_gross)
        t.cost_r = _round(t.entry_leg_cost_r + self._commission_half_r()
                          + self._pips_to_r(t.realized_exit_slip_pips))
        t.exit_price = _round(price) if price is not None else None
        t.exit_reason = reason
        t.ts_exit = pd.Timestamp(row["ts_open"])
        t.bars_held = t.action_bars_seen
        t.closed = True
        t.weekend_hold = crosses_weekend(pd.Timestamp(t.signal.ts_signal),
                                         pd.Timestamp(row["ts_open"]))
        return t

    # finalize ------------------------------------------------------------
    def finalize(self, last_row) -> Trade:
        """Close whatever remains at end of data (or expire a pending order)."""
        t = self.trade
        if t.closed:
            return t
        sig = t.signal
        if t.ts_entry is None:
            return self._close(last_row, ExitReason.EXPIRED, None, 0.0, 0.0)
        long = sig.direction is Direction.LONG
        px = last_row["close"] if long else \
            last_row["close"] + _spread_price(sig.symbol, last_row["spread"])
        leg_r = ((px - t.entry_fill) if long else (t.entry_fill - px)) / t.risk_distance
        return self._close(last_row, ExitReason.END_OF_DATA, px, leg_r, 0.0)
