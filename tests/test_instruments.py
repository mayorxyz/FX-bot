"""Tests for fxb.core.instruments."""
import pytest

from fxb.core.instruments import (
    UnknownInstrumentError,
    get_spec,
    override_spec,
    pips_to_price,
    pip_value_per_lot,
    price_to_pips,
    reset_specs,
    round_lots,
)


def test_registry_complete():
    for sym in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
                "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY"]:
        spec = get_spec(sym)
        assert spec.symbol == sym


def test_eurusd_pip_math():
    spec = get_spec("EURUSD")
    assert spec.pip_size == 0.0001
    assert spec.point_size == 0.00001
    assert spec.digits == 5
    assert price_to_pips("EURUSD", 0.0015) == pytest.approx(15.0)
    assert pips_to_price("EURUSD", 15.0) == pytest.approx(0.0015)


def test_usdjpy_pip_math():
    spec = get_spec("USDJPY")
    assert spec.pip_size == 0.01
    assert spec.point_size == 0.001
    assert spec.digits == 3
    assert price_to_pips("USDJPY", 0.25) == pytest.approx(25.0)
    assert pips_to_price("USDJPY", 25.0) == pytest.approx(0.25)


def test_unknown_symbol_raises_clear_error():
    with pytest.raises(UnknownInstrumentError, match="unknown symbol"):
        get_spec("NOTAPAIR")


def test_pip_value_per_lot_eurusd_usd_account():
    # 100000 * 0.0001 * 1.0 = 10 USD per pip per lot
    assert pip_value_per_lot("EURUSD", 1.0) == pytest.approx(10.0)


def test_pip_value_per_lot_usdjpy_usd_account():
    # 100000 * 0.01 * (1/150) = 6.6667 USD when USDJPY = 150
    assert pip_value_per_lot("USDJPY", 1.0 / 150.0) == pytest.approx(20.0 / 3.0)


def test_pip_value_rejects_bad_rate():
    with pytest.raises(ValueError):
        pip_value_per_lot("EURUSD", 0.0)


def test_round_lots_below_min_returns_zero():
    assert round_lots("EURUSD", 0.009) == 0.0
    assert round_lots("EURUSD", 0.0) == 0.0
    assert round_lots("EURUSD", -1.0) == 0.0


def test_round_lots_step_flooring():
    assert round_lots("EURUSD", 0.129) == pytest.approx(0.12)
    assert round_lots("EURUSD", 1.234) == pytest.approx(1.23)


def test_round_lots_clamps_above_max():
    assert round_lots("EURUSD", 123.4) == pytest.approx(100.0)


def test_override_and_reset_spec():
    try:
        new = override_spec("EURUSD", pip_size=0.00001, digits=6)
        assert get_spec("EURUSD").pip_size == 0.00001
        with pytest.raises(ValueError):
            override_spec("EURUSD", not_a_field=1)
    finally:
        reset_specs()
    assert get_spec("EURUSD").pip_size == 0.0001
