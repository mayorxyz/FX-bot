"""Tests for fxb.data.providers.csv_provider (offline, temp files)."""
import pandas as pd
import pytest

from fxb.data.providers.csv_provider import CsvProvider


MT5_STYLE = (
    "Time;Open;High;Low;Close;Volume;Spread\n"
    "2024.06.10 00:00;1.00000;1.00100;0.99900;1.00050;120;15\n"
    "2024.06.10 00:15;1.00050;1.00150;1.00000;1.00100;98;15\n"
)


def write(tmp_path, text, name="bars.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_mt5_style_default_map(tmp_path):
    p = write(tmp_path, MT5_STYLE)
    prov = CsvProvider(p, column_map={"Time": "ts_open", "Open": "open",
                                      "High": "high", "Low": "low",
                                      "Close": "close", "Volume": "tick_volume",
                                      "Spread": "spread"},
                      source_tz="UTC", datetime_format="%Y.%m.%d %H:%M")
    df = prov.fetch_candles("EURUSD", "M15")
    assert len(df) == 2
    assert str(df["ts_open"].dt.tz) == "UTC"
    assert df["spread"].iloc[0] == 15.0
    assert df["tick_volume"].iloc[1] == 98


def test_non_utc_source_tz_converted_to_utc(tmp_path):
    """Times stamped in America/New_York (EDT, UTC-4 in June) must shift +4h."""
    text = (
        "Date;Time;Open;High;Low;Close;Volume\n"
        "2024.06.10;00:00;1.00000;1.00100;0.99900;1.00050;120\n"
        "2024.06.10;01:00;1.00050;1.00150;1.00000;1.00100;98\n"
    )
    p = write(tmp_path, text)
    prov = CsvProvider(
        p,
        column_map={"Date": "ts_open", "Time": "ts_open", "Open": "open",
                    "High": "high", "Low": "low", "Close": "close",
                    "Volume": "tick_volume"},
        source_tz="America/New_York",
        datetime_format="%Y.%m.%d %H:%M",
    )
    df = prov.fetch_candles("EURUSD", "H1")
    assert len(df) == 2
    # NY midnight EDT == 04:00 UTC next day... 2024-06-10 00:00 EDT -> 04:00Z same? No: EDT=UTC-4 so 00:00 EDT = 04:00 UTC on 2024-06-10.
    assert df["ts_open"].iloc[0] == pd.Timestamp("2024-06-10 04:00", tz="UTC")
    assert df["ts_open"].iloc[1] == pd.Timestamp("2024-06-10 05:00", tz="UTC")
    # spread column missing -> default filled
    assert (df["spread"] == prov.default_spread_points).all()


def test_missing_tick_volume_defaults_zero(tmp_path):
    text = (
        "Time;Open;High;Low;Close\n"
        "2024.06.10 00:00;1.0;1.1;0.9;1.05\n"
    )
    p = write(tmp_path, text)
    prov = CsvProvider(p, column_map={"Time": "ts_open", "Open": "open",
                                      "High": "high", "Low": "low",
                                      "Close": "close"},
                      source_tz="UTC", datetime_format="%Y.%m.%d %H:%M")
    df = prov.fetch_candles("EURUSD", "H1")
    assert df["tick_volume"].iloc[0] == 0


def test_range_filtering(tmp_path):
    p = write(tmp_path, MT5_STYLE)
    prov = CsvProvider(p, column_map={"Time": "ts_open", "Open": "open",
                                      "High": "high", "Low": "low",
                                      "Close": "close", "Volume": "tick_volume",
                                      "Spread": "spread"},
                      source_tz="UTC", datetime_format="%Y.%m.%d %H:%M")
    df = prov.fetch_candles("EURUSD", "M15",
                            start_utc=pd.Timestamp("2024-06-10 00:15", tz="UTC"),
                            end_utc=pd.Timestamp("2024-06-10 23:00", tz="UTC"))
    assert len(df) == 1
    assert df["ts_open"].iloc[0] == pd.Timestamp("2024-06-10 00:15", tz="UTC")


def test_missing_file_raises(tmp_path):
    prov = CsvProvider(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        prov.fetch_candles("EURUSD", "M15")
