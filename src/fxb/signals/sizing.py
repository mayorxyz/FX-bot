"""Position sizing for human traders following signals (alert support).

The backtester reports in R-multiples and does not need lots; this module
exists so future alerting can attach a ``suggested_lots`` figure to each
signal. Pure functions, no state.
"""
from __future__ import annotations

from fxb.core.instruments import pip_value_per_lot, round_lots


def suggested_lots(symbol: str, account_balance: float, risk_pct: float,
                   entry: float, stop: float,
                   quote_to_account_rate: float = 1.0,
                   lot_step: float = 0.01) -> float:
    """Lot size risking ``risk_pct`` (fraction, e.g. 0.01 = 1%) of balance.

    lots = (balance * risk_pct) / (risk_distance_in_pips * pip_value_per_lot)

    ``quote_to_account_rate`` is units of account currency per 1 unit of the
    pair's QUOTE currency (1.0 when the account currency == quote currency).
    Rounded DOWN to the broker step; returns 0.0 when the risk distance or
    resulting size is unusable (caller should then suppress the alert).
    """
    if account_balance <= 0 or risk_pct <= 0:
        raise ValueError("account_balance and risk_pct must be > 0")
    risk_price = abs(float(entry) - float(stop))
    if risk_price <= 0:
        raise ValueError("entry and stop must differ (risk distance > 0)")
    from fxb.core.instruments import get_instrument
    pip_size = get_instrument(symbol).pip_size
    risk_pips = risk_price / pip_size
    pip_value = pip_value_per_lot(symbol, quote_to_account_rate=quote_to_account_rate)
    raw = (account_balance * risk_pct) / (risk_pips * pip_value)
    return round_lots(raw, step=lot_step)
