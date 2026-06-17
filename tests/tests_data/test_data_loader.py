"""Tests for DataLoader. All external I/O (pandas readers, requests, yfinance) is
mocked so nothing touches the disk or network."""
import pandas as pd
import pytest

from src.data import data_loader
from src.data.data_loader import DataLoader


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader()


def test_load_parquet(loader: DataLoader, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda p: sentinel)
    assert loader.load_parquet("x.parquet") is sentinel


def test_load_csv(loader: DataLoader, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(data_loader.pd, "read_csv", lambda p: sentinel)
    assert loader.load_csv("x.csv") is sentinel


def test_load_api(loader: DataLoader, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict]:
            return [{"a": 1}, {"a": 2}]

    monkeypatch.setattr(data_loader.requests, "get", lambda url, timeout: FakeResp())
    df = loader.load_api("http://example.com")
    assert list(df["a"]) == [1, 2]


def test_load_yfinance(loader: DataLoader, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = pd.DataFrame({"Close": [1.0]})
    monkeypatch.setattr(data_loader.yfinance, "download", lambda **kwargs: sentinel)
    out = loader.load_yfinance("^SPX", "2024-01-01", "2024-02-01", "1d")
    assert out is sentinel


def test_load_spx_uses_cache_when_path_hits(
    loader: DataLoader, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = pd.DataFrame({"cached": [1]})
    monkeypatch.setattr(data_loader.pd, "read_parquet", lambda p: cached)
    # yfinance must NOT be called on a cache hit
    monkeypatch.setattr(data_loader.yfinance, "download",
                        lambda **kwargs: pytest.fail("network hit on cache"))
    out = loader.load_spx("2024-01-01", "2024-02-01", "1d", path="cache.parquet")
    assert out is cached


def test_load_spx_falls_back_when_cache_raises(
    loader: DataLoader, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_p: str) -> pd.DataFrame:
        raise FileNotFoundError

    downloaded = pd.DataFrame({"net": [1]})
    monkeypatch.setattr(data_loader.pd, "read_parquet", _raise)
    monkeypatch.setattr(data_loader.yfinance, "download", lambda **kwargs: downloaded)
    out = loader.load_spx("2024-01-01", "2024-02-01", "1d", path="missing.parquet")
    assert out is downloaded


def test_load_spx_no_path_goes_to_network(
    loader: DataLoader, monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = pd.DataFrame({"net": [1]})
    monkeypatch.setattr(data_loader.yfinance, "download", lambda **kwargs: downloaded)
    out = loader.load_spx("2024-01-01", "2024-02-01", "1d")
    assert out is downloaded


def test_load_option_chain(loader: DataLoader, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = pd.DataFrame({"strike": [100.0, 110.0], "lastPrice": [5.0, 2.0]})
    puts = pd.DataFrame({"strike": [100.0, 110.0], "lastPrice": [4.0, 6.0]})

    class FakeChain:
        def __init__(self) -> None:
            self.calls = calls
            self.puts = puts
            # underlying quote captured alongside every expiry (spot + snapshot epoch)
            self.underlying = {"regularMarketPrice": 105.0,
                               "regularMarketTime": 1_780_000_000}

    class FakeTicker:
        def __init__(self, ticker: str) -> None:
            self.options = ["2026-06-19", "2026-06-26", "2026-07-17"]

        def option_chain(self, expiry: str) -> FakeChain:
            return FakeChain()

    monkeypatch.setattr(data_loader.yfinance, "Ticker", FakeTicker)
    # wide day-window so selection isn't dependent on the run date; the ladder then
    # spreads its picks across the expiries instead of taking the nearest ones.
    df = loader.load_option_chain(ticker="^SPX", n_maturities=2,
                                  min_days=0, max_days=100_000)

    # 2 expiries x {call, put} x 2 strikes = 8 rows
    assert len(df) == 8
    assert set(df["type"]) == {"call", "put"}
    # even spread of 2 across 3 expiries -> first and last, not the nearest two
    assert set(df["expiry"]) == {"2026-06-19", "2026-07-17"}
    assert df["spot"].iloc[0] == 105.0
