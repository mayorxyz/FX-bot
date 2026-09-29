"""Bar-by-bar backtest engine.

Loop invariant (look-ahead safety): at each primary-timeframe bar B with close
time ``now``, the strategy sees a bounded MarketContext containing only bars
closed at or before ``now``; signals generated there become actionable only on
bars strictly after B (the fill simulator is never fed the signal's own bar).

Portfolio rules (configurable via BacktestConfig):
- max_concurrent_trades: cap on simultaneously open positions (pending limit
  orders count as open slots once accepted).
- max_trades_per_symbol: default 1.
- max_currency_exposure: at most N open trades sharing a currency (base or
  quote) in the same *exposure sense* — a LONG pair exposes its base currency
  and is short its quote; a SHORT pair exposes its quote. Two EURUSD longs and
  one EURJPY long all share EUR exposure (see _currency_exposures).
Skipped signals are recorded in the signals frame with
status = SKIPPED_PORTFOLIO_RULE (never silently dropped) and logged.

Determinism: no randomness anywhere in the engine; floats rounded to 8 dp in
fills; identical inputs -> byte-identical output frames.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, asdict
from typing import Iterable

import pandas as pd

from fxb.core.config import get_logger
from fxb.core.timeutil import tf_to_timedelta, to_utc
from fxb.backtest.fills import (DEFAULT_FILL_CONFIG, FillConfig, PositionSimulator,
                                Trade, TRADE_COLUMNS)
from fxb.data.providers import CandleProvider, DictProvider
from fxb.signals.models import Signal
from fxb.signals.strategy_base import Strategy, build_context

log = get_logger("backtest.engine")

SIGNAL_STATUS_ACCEPTED = "ACCEPTED"
SIGNAL_STATUS_SKIPPED = "SKIPPED_PORTFOLIO_RULE"


def sig_ts_open(sig: Signal, tf: str) -> pd.Timestamp:
    """OPEN time of the bar whose CLOSE produced ``sig`` (ts_signal - tf duration)."""
    return pd.Timestamp(sig.ts_signal) - tf_to_timedelta(tf)


@dataclass(frozen=True)
class BacktestConfig:
    start: str | None = None            # inclusive UTC lower bound (bar close time)
    end: str | None = None              # exclusive UTC upper bound
    max_concurrent_trades: int = 3
    max_trades_per_symbol: int = 1
    max_currency_exposure: int = 2
    seed: int = 0                       # reserved (engine itself has no randomness)
    fill: FillConfig = DEFAULT_FILL_CONFIG

    @staticmethod
    def from_dict(d: dict) -> "BacktestConfig":
        dd = dict(d)
        fill = FillConfig.from_dict(dd.pop("fill", {}))
        keys = {f.name for f in fields(BacktestConfig)} - {"fill"}
        return BacktestConfig(fill=fill, **{k: v for k, v in dd.items() if k in keys})

    def snapshot(self) -> dict:
        d = asdict(self)
        return d

    def config_hash(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, default=str,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class BacktestResult:
    trades: pd.DataFrame          # columns TRADE_COLUMNS (+ status for context)
    signals: pd.DataFrame         # every signal emitted, with ACCEPT/SKIP status
    skipped: pd.DataFrame         # convenience view of skipped signals
    config: dict                  # config snapshot incl. hash
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            h = hashlib.sha256()
            h.update(json.dumps(self.config, sort_keys=True, default=str).encode())
            n = f"{len(self.trades)}/{len(self.signals)}".encode()
            h.update(n)
            self.run_id = h.hexdigest()[:12]


def _signal_row(sig: Signal, status: str, skip_reason: str | None) -> dict:
    return {
        "signal_id": sig.signal_id,
        "symbol": sig.symbol,
        "timeframe": sig.timeframe,
        "ts_signal": sig.ts_signal,
        "direction": sig.direction.value,
        "entry_type": sig.entry_type.value,
        "entry_price": sig.entry_price,
        "stop_price": sig.stop_price,
        "take_profit_prices": list(sig.take_profit_prices),
        "expiry_bars": sig.expiry_bars,
        "max_hold_bars": sig.max_hold_bars,
        "score": sig.score,
        "reasons": list(sig.reasons),
        "status": status,
        "skip_reason": skip_reason,
    }


def _currency_exposures(open_trades: list[Trade]) -> dict[str, int]:
    """Count open trades exposing each currency (LONG -> base exposed, SHORT ->
    quote exposed; both directions also expose the counter through P&L but the
    documented rule here counts directional exposure of the traded currency)."""
    counts: dict[str, int] = {}
    for t in open_trades:
        base, quote = t.signal.symbol[:3], t.signal.symbol[3:6]
        from fxb.signals.models import Direction
        cur = base if t.signal.direction is Direction.LONG else quote
        counts[cur] = counts.get(cur, 0) + 1
    return counts


class _OpenBook:
    def __init__(self):
        self.sims: list[PositionSimulator] = []

    def active(self) -> list[Trade]:
        return [s.trade for s in self.sims]

    def add(self, sim: PositionSimulator):
        self.sims.append(sim)

    def remove_closed(self):
        self.sims = [s for s in self.sims if not s.trade.closed]


def run_backtest(strategy: Strategy, provider_or_store, symbols: Iterable[str],
                 tf: str, start=None, end=None,
                 config: BacktestConfig | None = None) -> BacktestResult:
    """Run ``strategy`` over ``symbols`` on primary timeframe ``tf``.

    ``provider_or_store`` must expose ``get_candles(symbol, timeframe)``
    (a CandleProvider or anything duck-typed like DictProvider).
    """
    cfg = config or BacktestConfig()
    symbols = tuple(s.upper() for s in symbols)
    tf = tf.upper()
    provider = provider_or_store
    lo = to_utc(start) if start is not None else None
    hi = to_utc(end) if end is not None else None

    # reference symbol series drives the event loop (first symbol with data)
    ref = None
    for s in symbols:
        df = provider.get_candles(s, tf)
        if len(df):
            ref = s
            break
    if ref is None:
        raise ValueError("provider returned no candles for any requested symbol")

    primary = provider.get_candles(ref, tf)
    ts_close = primary["ts_open"] + tf_to_timedelta(tf)
    mask = pd.Series(True, index=primary.index)
    if lo is not None:
        mask &= ts_close >= lo
    if hi is not None:
        mask &= ts_close < hi
    loop_idx = list(primary.index[mask])

    book = _OpenBook()
    trade_rows: list[dict] = []
    signal_rows: list[dict] = []
    extra_tfs = tuple(strategy.htf_timeframes)

    for pos_i, i in enumerate(loop_idx):
        now = ts_close.iloc[i]
        ctx = build_context(provider, symbols, tf, now, extra_tfs=extra_tfs)
        # 1) manage existing orders/positions on THIS bar (bars after their signal)
        row = primary.iloc[i]
        for sym in symbols:
            sdf = provider.get_candles(sym, tf)
            sclose = sdf["ts_open"] + tf_to_timedelta(tf)
            j = int((sclose <= now).values.nonzero()[0][-1]) \
                if (sclose <= now).any() else -1
            if j < 0:
                continue
            srow = sdf.iloc[j]
            for sim in list(book.sims):
                if sim.trade.signal.symbol != sym or sim.trade.closed:
                    continue
                # A signal generated at bar close (ts_signal == this bar's close
                # time) is actionable only from the NEXT bar onward.
                if sig_ts_open(sim.trade.signal, tf) >= sclose.iloc[j]:
                    continue  # cannot act on the signal's own bar
                finished = sim.process_bar(srow)
                if finished is not None:
                    trade_rows.append(finished.to_row())
        book.remove_closed()

        # 2) ask the strategy for signals at this bar close
        try:
            new_signals = strategy.on_bar(ctx) or []
        except Exception as exc:  # strategy bugs should be loud but not fatal
            log.error("strategy %s raised at %s: %s", strategy.name, now, exc)
            raise
        for sig in new_signals:
            reason = _portfolio_reject(sig, book, cfg, symbols, provider, tf, now)
            if reason:
                signal_rows.append(_signal_row(sig, SIGNAL_STATUS_SKIPPED, reason))
                log.info("SKIPPED_PORTFOLIO_RULE %s %s (%s) at %s: %s",
                         sig.symbol, sig.direction.value, sig.signal_id, now, reason)
                continue
            signal_rows.append(_signal_row(sig, SIGNAL_STATUS_ACCEPTED, None))
            sim = PositionSimulator(Trade(signal=sig), cfg.fill, timeframe=tf)
            book.add(sim)

    # 3) end of data: finalize everything against each symbol's last looped bar
    last_rows = {}
    for sym in symbols:
        sdf = provider.get_candles(sym, tf)
        sclose = sdf["ts_open"] + tf_to_timedelta(tf)
        sel = sclose <= (to_utc(hi) - tf_to_timedelta(tf) if hi is not None else sclose.iloc[-1])
        idx = int(sel.values.nonzero()[0][-1]) if sel.any() else None
        if idx is not None:
            last_rows[sym] = sdf.iloc[idx]
    for sim in list(book.sims):
        if not sim.trade.closed:
            sym = sim.trade.signal.symbol
            fin = sim.finalize(last_rows[sym])
            trade_rows.append(fin.to_row())

    trades_df = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    if len(trades_df):
        trades_df = trades_df.sort_values(["ts_signal", "signal_id"]).reset_index(drop=True)
    sigs_df = pd.DataFrame(
        signal_rows,
        columns=["signal_id", "symbol", "timeframe", "ts_signal", "direction",
                 "entry_type", "entry_price", "stop_price", "take_profit_prices",
                 "expiry_bars", "max_hold_bars", "score", "reasons", "status",
                 "skip_reason"])
    if len(sigs_df):
        sigs_df = sigs_df.sort_values(["ts_signal", "signal_id"]).reset_index(drop=True)
    skipped = sigs_df[sigs_df["status"] == SIGNAL_STATUS_SKIPPED].copy() \
        if len(sigs_df) else sigs_df
    snap = cfg.snapshot()
    snap["config_hash"] = cfg.config_hash()
    snap.update({"strategy": strategy.name, "symbols": list(symbols), "tf": tf,
                 "n_trades": len(trades_df), "n_signals": len(sigs_df)})
    return BacktestResult(trades=trades_df, signals=sigs_df, skipped=skipped,
                          config=snap)


def _portfolio_reject(sig: Signal, book: _OpenBook, cfg: BacktestConfig,
                      symbols, provider, tf, now) -> str | None:
    """Return a human-readable rejection reason or None to accept."""
    open_trades = book.active()
    if len(open_trades) >= cfg.max_concurrent_trades:
        return (f"max_concurrent_trades={cfg.max_concurrent_trades} reached "
                f"({len(open_trades)} open)")
    same_sym = [t for t in open_trades if t.signal.symbol == sig.symbol]
    if len(same_sym) >= cfg.max_trades_per_symbol:
        return f"max_trades_per_symbol={cfg.max_trades_per_symbol} for {sig.symbol}"
    base, quote = sig.symbol[:3], sig.symbol[3:6]
    from fxb.signals.models import Direction
    cur = base if sig.direction is Direction.LONG else quote
    counts = _currency_exposures(open_trades)
    if counts.get(cur, 0) >= cfg.max_currency_exposure:
        return f"max_currency_exposure={cfg.max_currency_exposure} for {cur}"
    return None
