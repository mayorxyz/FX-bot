"""Instrument metadata: point/pip sizes, contract sizes and pip value per lot.

All values are static reference data for the majors (documented approximations;
in production these would be refreshed from broker specs).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    point: float          # smallest quoted price increment
    pip_size: float       # size of one pip in quote currency units
    contract_size: int    # units of base currency in 1.0 lot
    default_spread_points: float  # fallback spread when bar data has none


#: JPY-quoted pairs have pip = 0.01, everything else here has pip = 0.0001.
_INSTRUMENTS: dict[str, InstrumentSpec] = {}


def _add(symbol: str, point: float, pip_size: float, contract_size: int, spread: float):
    _INSTRUMENTS[symbol] = InstrumentSpec(symbol, point, pip_size, contract_size, spread)


_add("EURUSD", 0.00001, 0.0001, 100_000, 1.2)
_add("GBPUSD", 0.00001, 0.0001, 100_000, 1.6)
_add("AUDUSD", 0.00001, 0.0001, 100_000, 1.5)
_add("NZDUSD", 0.00001, 0.0001, 100_000, 2.0)
_add("USDCAD", 0.00001, 0.0001, 100_000, 1.8)
_add("USDCHF", 0.00001, 0.0001, 100_000, 1.9)
_add("USDJPY", 0.001, 0.01, 100_000, 1.4)
_add("EURJPY", 0.001, 0.01, 100_000, 1.7)
_add("GBPJPY", 0.001, 0.01, 100_000, 2.4)
_add("EURGBP", 0.00001, 0.0001, 100_000, 1.5)


def get_instrument(symbol: str) -> InstrumentSpec:
    s = symbol.upper()
    if s not in _INSTRUMENTS:
        raise KeyError(f"Unknown instrument '{symbol}'")
    return _INSTRUMENTS[s]


def known_symbols() -> list[str]:
    return sorted(_INSTRUMENTS)


def pip_value_per_lot(symbol: str, account_currency: str = "USD",
                      quote_to_account_rate: float = 1.0) -> float:
    """USD (or ``account_currency``) value of a 1-pip move on a 1.0-lot position.

    For XXX/quote pairs the pip value is ``contract_size * pip_size`` expressed in
    the quote currency; converting to the account currency multiplies by
    ``quote_to_account_rate`` (units of account currency per 1 unit of quote).

    Example: EURUSD -> 100000 * 0.0001 * 1.0 = 10.0 USD per pip per lot.
    """
    spec = get_instrument(symbol)
    return spec.contract_size * spec.pip_size * quote_to_account_rate


def round_lots(lots: float, step: float = 0.01, min_lots: float = 0.01,
               max_lots: float = 100.0) -> float:
    """Round a raw lot size down to the broker step, clamped to [min, max]."""
    if lots <= 0:
        return 0.0
    stepped = int(lots / step + 1e-9) * step
    stepped = max(min_lots, min(max_lots, stepped))
    return round(stepped, 8)
