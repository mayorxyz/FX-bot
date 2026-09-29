"""Instrument reference data: pip/point sizes, contract specs, lot rounding.

Conventions:
- ``point_size`` is the broker's smallest quoted increment; ``pip_size`` is
  10 points. JPY-quoted pairs: pip = 0.01, digits = 3. All others here:
  pip = 0.0001, digits = 5.
- Candle prices stored by FXB are BID prices with ``spread`` in points
  (ask = bid + spread * point_size). See :mod:`fxb.data.store`.
- Broker specs can differ from these defaults; :func:`override_spec` lets a
  later step replace fields from ``Mt5Provider.symbol_info``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Dict


@dataclass(frozen=True)
class InstrumentSpec:
    """Static description of one tradable symbol."""

    symbol: str
    base: str
    quote: str
    pip_size: float
    point_size: float
    digits: int
    contract_size: int = 100_000
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0


def _spec(symbol: str, base: str, quote: str, jpy: bool) -> InstrumentSpec:
    pip = 0.01 if jpy else 0.0001
    return InstrumentSpec(
        symbol=symbol, base=base, quote=quote,
        pip_size=pip, point_size=pip / 10.0, digits=3 if jpy else 5,
    )


#: Default registry for the majors requested in the spec.
_REGISTRY: Dict[str, InstrumentSpec] = {
    s.symbol: s
    for s in [
        _spec("EURUSD", "EUR", "USD", False),
        _spec("GBPUSD", "GBP", "USD", False),
        _spec("USDJPY", "USD", "JPY", True),
        _spec("USDCHF", "USD", "CHF", False),
        _spec("AUDUSD", "AUD", "USD", False),
        _spec("USDCAD", "USD", "CAD", False),
        _spec("NZDUSD", "NZD", "USD", False),
        _spec("EURJPY", "EUR", "JPY", True),
        _spec("GBPJPY", "GBP", "JPY", True),
        _spec("EURGBP", "EUR", "GBP", False),
        _spec("AUDJPY", "AUD", "JPY", True),
    ]
}


class UnknownInstrumentError(KeyError):
    """Raised when a symbol has no registered instrument spec."""


def get_spec(symbol: str) -> InstrumentSpec:
    """Return the :class:`InstrumentSpec` for ``symbol`` (case-insensitive)."""
    key = symbol.strip().upper()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise UnknownInstrumentError(
            f"unknown symbol {symbol!r}; known: {sorted(_REGISTRY)}"
        ) from None


def override_spec(symbol: str, **spec_fields: object) -> InstrumentSpec:
    """Replace fields of a registered spec (e.g. after reading broker info).

    Returns the new frozen spec now stored in the registry. Only existing
    dataclass fields may be overridden.
    """
    current = get_spec(symbol)
    allowed = {f.name for f in fields(InstrumentSpec)}
    bad = set(spec_fields) - allowed
    if bad:
        raise ValueError(f"unknown InstrumentSpec field(s): {sorted(bad)}")
    new = replace(current, **spec_fields)
    _REGISTRY[new.symbol.upper()] = new
    return new


def reset_specs() -> None:
    """Restore factory defaults (used by tests around :func:`override_spec`)."""
    global _REGISTRY
    _REGISTRY = dict(_FACTORY)


#: Snapshot of the factory registry, taken before any overrides can occur.
_FACTORY = dict(_REGISTRY)


def price_to_pips(symbol: str, price_diff: float) -> float:
    """Convert an absolute price difference to pips for ``symbol``."""
    return price_diff / get_spec(symbol).pip_size


def pips_to_price(symbol: str, pips: float) -> float:
    """Convert a pip count to an absolute price difference for ``symbol``."""
    return pips * get_spec(symbol).pip_size


def pip_value_per_lot(symbol: str, quote_to_account_rate: float) -> float:
    """Value of one pip for 1.0 lot, expressed in the account currency.

    ``quote_to_account_rate`` converts 1 unit of the quote currency into the
    account currency (1.0 when quote == account currency). The caller is
    responsible for supplying a correct rate (e.g. USDJPY / 150 for a USD
    account). Formula: ``contract_size * pip_size * quote_to_account_rate``.
    """
    if quote_to_account_rate <= 0:
        raise ValueError("quote_to_account_rate must be > 0")
    spec = get_spec(symbol)
    return spec.contract_size * spec.pip_size * quote_to_account_rate


def round_lots(symbol: str, lots: float) -> float:
    """Floor ``lots`` to the lot step and clamp to [min_lot, max_lot].

    Returns ``0.0`` when the floored size is below ``min_lot`` (i.e. the
    requested size cannot be traded).
    """
    spec = get_spec(symbol)
    if lots <= 0:
        return 0.0
    steps = math.floor(lots / spec.lot_step + 1e-9)
    value = steps * spec.lot_step
    if value < spec.min_lot - 1e-12:
        return 0.0
    return min(round(value, 8), spec.max_lot)
