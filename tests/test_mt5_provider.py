"""Offline tests for fxb.data.providers.mt5_provider using an injected fake module.

MetaTrader5 is NEVER imported here; a fake module object stands in for it,
and the real package's absence must not break this test file.
"""
import pandas as pd
import pytest

from fxb.core.config import Settings
from fxb.data.providers.mt5_provider import Mt5Provider


class FakeMt5:
    """Mimics the pieces of MetaTrader5 that Mt5Provider touches."""

    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60

    def __init__(self, rates):
        self._rates = rates
        self.last_request = None

    def initialize(self, **kwargs):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (0, "fake")

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        self.last_request = (symbol, timeframe, date_from, date_to)
        return self._rates


def settings_with(offset_hours: float) -> Settings:
    return Settings(
        db_path="data/test.db",
        symbols=["EURUSD"],
        timeframes=["M15"],
        account_currency="USD",
        risk_pct=0.01,
        max_portfolio_heat=0.06,
        mt5_login=None,
        mt5_password=None,
        mt5_server=None,
        mt5_server_utc_offset_hours=offset_hours,
        d1_anchor="17:00 America/New_York",
        log_level="INFO",
    )


def _rate(epoch_s, o=1.0, h=1.1, l=0.9, c=1.05, vol=10, spread=2):
    return {"time": epoch_s, "open": o, "high": h, "low": l, "close": c,
            "tick_volume": vol, "spread": spread}


def test_no_mt5_import_at_module_import_time():
    import sys
    import fxb.data.providers.mt5_provider as mod
    # The module itself must not have imported the real package.
    src = open(mod.__file__, encoding="utf-8").read()
    assert "\nimport MetaTrader5" not in src  # only inside _import_mt5()
    assert "MetaTrader5 as mt5" in src.split("def _import_mt5")[1]


def test_offset_plus2_shifts_bars_to_correct_utc():
    """REGRESSION: server stamps are UTC+2 wall-clock encoded as epochs.

    A bar stamped 2024-06-10 02:15 server time is really 00:15 UTC.
    """
    srv_0015z = int(pd.Timestamp("2024-06-10 02:15", tz="UTC").timestamp())
    srv_0100z = int(pd.Timestamp("2024-06-10 03:00", tz="UTC").timestamp())
    fake = FakeMt5([_rate(srv_0015z), _rate(srv_0100z)])
    prov = Mt5Provider(settings=settings_with(2.0), mt5=fake)

    df = prov.fetch_candles(
        "EURUSD", "M15",
        start_utc=pd.Timestamp("2024-06-10 00:00", tz="UTC"),
        end_utc=pd.Timestamp("2024-06-10 02:00", tz="UTC"),
    )
    assert list(df["ts_open"]) == [
        pd.Timestamp("2024-06-10 00:15", tz="UTC"),
        pd.Timestamp("2024-06-10 01:00", tz="UTC"),
    ]


def test_request_range_padded_by_one_day():
    fake = FakeMt5([])
    prov = Mt5Provider(settings=settings_with(2.0), mt5=fake)
    start = pd.Timestamp("2024-06-10 00:00", tz="UTC")
    end = pd.Timestamp("2024-06-11 00:00", tz="UTC")
    df = prov.fetch_candles("EURUSD", "M15", start, end)
    req_from, req_to = fake.last_request[2], fake.last_request[3]
    assert pd.Timestamp(req_from, tz="UTC") == start - pd.Timedelta(days=1)
    assert pd.Timestamp(req_to, tz="UTC") == end + pd.Timedelta(days=1)
    assert len(df) == 0


def test_unclosed_bar_dropped_and_out_of_range_filtered():
    now = pd.Timestamp.now(tz="UTC")
    closed_open = (now.floor("15min") - pd.Timedelta(minutes=15))
    forming_open = now.floor("15min")
    # Bars BEFORE today (server epoch = UTC epoch here, offset 0) so the
    # window filter keeps them; plus one still-forming bar inside the window.
    rates = [
        _rate(int(closed_open.timestamp())),
        _rate(int(forming_open.timestamp())),   # closes in the future -> dropped
    ]
    fake = FakeMt5(rates)
    prov = Mt5Provider(settings=settings_with(0.0), mt5=fake)
    df = prov.fetch_candles("EURUSD", "M15",
                            start_utc=closed_open - pd.Timedelta(hours=1),
                            end_utc=now + pd.Timedelta(minutes=30))
    assert len(df) == 1
    assert df["ts_open"].iloc[0] == closed_open
    # out-of-window bars filtered: request far in the past returns empty
    df2 = prov.fetch_candles("EURUSD", "M15",
                             start_utc=pd.Timestamp("2020-01-01", tz="UTC"),
                             end_utc=pd.Timestamp("2020-01-02", tz="UTC"))
    assert len(df2) == 0


def test_negative_offset_shifts_other_way():
    srv = int(pd.Timestamp("2024-06-10 23:15", tz="UTC").timestamp())  # server 23:15, UTC-2 => 01:15Z next day? No: stamp - (-2h) = +2h
    fake = FakeMt5([_rate(srv)])
    prov = Mt5Provider(settings=settings_with(-2.0), mt5=fake)
    df = prov.fetch_candles("EURUSD", "M15",
                            start_utc=pd.Timestamp("2024-06-11 00:00", tz="UTC"),
                            end_utc=pd.Timestamp("2024-06-11 06:00", tz="UTC"))
    assert df["ts_open"].iloc[0] == pd.Timestamp("2024-06-11 01:15", tz="UTC")
