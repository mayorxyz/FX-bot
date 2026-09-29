"""Tests for fxb.data.store (SQLite candle store)."""
import pandas as pd
import pytest

from conftest import grid, make_candles
from fxb.data.store import CandleStore
from fxb.data.validate import CandleValidationError


def df5():
    return make_candles(grid(15, "2024-06-10 08:00", 5))


@pytest.fixture()
def store(tmp_path):
    s = CandleStore(tmp_path / "candles.db")
    yield s
    s.close()


def test_upsert_idempotent_twice_same_count(store):
    df = df5()
    assert store.upsert_candles("EURUSD", "M15", df) == 5
    assert store.upsert_candles("EURUSD", "M15", df) == 5
    assert store.count("EURUSD", "M15") == 5


def test_primary_key_dedupes_conflicting_values(store):
    df = df5()
    store.upsert_candles("EURUSD", "M15", df)
    changed = df.copy()
    changed.loc[0, "close"] = 1.5
    store.upsert_candles("EURUSD", "M15", changed)
    out = store.get_candles("EURUSD", "M15")
    assert len(out) == 5
    assert out["close"].iloc[0] == 1.5  # replaced, not duplicated


def test_range_query_inclusive_start_exclusive_end(store):
    store.upsert_candles("EURUSD", "M15", df5())
    start = pd.Timestamp("2024-06-10 08:15", tz="UTC")
    end = pd.Timestamp("2024-06-10 08:45", tz="UTC")
    out = store.get_candles("EURUSD", "M15", start=start, end=end)
    assert list(out.index) == [start, pd.Timestamp("2024-06-10 08:30", tz="UTC")]


def test_get_candles_index_utc_sorted_named(store):
    store.upsert_candles("gbpusd", "h1", make_candles(grid(60, "2024-06-10 00:00", 3)))
    out = store.get_candles("GBPUSD", "H1")
    assert out.index.name == "ts_open"
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing
    assert len(out) == 3


def test_last_ts_and_count_empty(store):
    assert store.last_ts("EURUSD", "M15") is None
    assert store.count("EURUSD", "M15") == 0
    store.upsert_candles("EURUSD", "M15", df5())
    assert store.last_ts("EURUSD", "M15") == pd.Timestamp("2024-06-10 09:00", tz="UTC")


def test_delete_range(store):
    store.upsert_candles("EURUSD", "M15", df5())
    n = store.delete_range("EURUSD", "M15",
                           start=pd.Timestamp("2024-06-10 08:30", tz="UTC"),
                           end=pd.Timestamp("2024-06-10 09:00", tz="UTC"))
    assert n == 2
    assert store.count("EURUSD", "M15") == 3


def test_validation_rejects_bad_bar(store):
    bad = df5()
    bad.loc[0, "high"] = 0.5  # high < low
    with pytest.raises(CandleValidationError):
        store.upsert_candles("EURUSD", "M15", bad)
    assert store.count("EURUSD", "M15") == 0  # nothing written


def test_context_manager_and_wal(tmp_path):
    path = tmp_path / "ctx.db"
    with CandleStore(path) as s:
        mode = s._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"
        s.upsert_candles("EURUSD", "M15", df5())
    # connection closed on exit; reopening sees the data
    with CandleStore(path) as s2:
        assert s2.count("EURUSD", "M15") == 5
