"""Tests for the ingestor implementations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cps.infrastructure.ingestors import (
    CCXTIngestorConfig,
    CCXTPoller,
    CsvIngestor,
    SyntheticIngestor,
    YFinanceConfig,
    YFinanceIngestor,
    default_exchange_factory,
    fetch_yfinance_prices,
    pivot_to_price_frame,
    resolve_exchange_factory,
)


class TestSyntheticIngestor:
    def test_fetch_shape_and_positivity(self):
        frame = SyntheticIngestor(days=20, assets=4, seed=3).fetch()
        assert frame.shape == (20, 4)
        assert (frame > 0).all().all()

    def test_deterministic_with_seed(self):
        first = SyntheticIngestor(seed=42).fetch()
        second = SyntheticIngestor(seed=42).fetch()
        assert first.equals(second)

    def test_validates_dimensions(self):
        with pytest.raises(ValueError):
            SyntheticIngestor(days=0)
        with pytest.raises(ValueError):
            SyntheticIngestor(assets=0)


class TestCsvIngestor:
    def test_fetch_returns_price_frame(self, prices_csv: Path):
        frame = CsvIngestor(str(prices_csv), date_col="date").fetch()
        assert frame.shape == (20, 3)
        assert frame.index.name == "date"

    def test_missing_date_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("x,a\n1,10\n", encoding="utf-8")
        with pytest.raises(ValueError):
            CsvIngestor(str(path), date_col="date").fetch()


class TestYFinanceIngestor:
    def test_fetch_returns_dataframe(self):
        yfinance = pytest.importorskip("yfinance")

        def fake_download(symbols, **kwargs):
            dates = pd.date_range("2024-01-01", periods=5, freq="D")
            columns = pd.MultiIndex.from_product(
                [["Open", "High", "Low", "Close", "Adj Close", "Volume"], symbols],
                names=["field", "Ticker"],
            )
            return pd.DataFrame(np.ones((len(dates), len(columns))), index=dates, columns=columns)

        with patch.object(yfinance, "download", side_effect=fake_download):
            frame = YFinanceIngestor(YFinanceConfig(symbols=("BTC-USD",), period="1mo")).fetch()
        assert list(frame.columns) == ["BTC-USD"]
        assert frame.shape[0] == 5

    def test_free_function(self):
        yfinance = pytest.importorskip("yfinance")

        def fake_download(symbols, **kwargs):
            dates = pd.date_range("2024-01-01", periods=3, freq="D")
            columns = pd.MultiIndex.from_product(
                [["Close"], symbols], names=["field", "Ticker"]
            )
            return pd.DataFrame(np.ones((3, 1)), index=dates, columns=columns)

        with patch.object(yfinance, "download", side_effect=fake_download):
            frame = fetch_yfinance_prices(symbols=("AAPL",), start="2024-01-01", end="2024-01-10")
        assert frame.shape == (3, 1)

    def test_rejects_empty_symbols(self):
        pytest.importorskip("yfinance")
        with pytest.raises(ValueError):
            YFinanceIngestor(YFinanceConfig(symbols=()))

    def test_rejects_missing_window(self):
        pytest.importorskip("yfinance")
        with pytest.raises(ValueError):
            YFinanceIngestor(YFinanceConfig(symbols=("BTC-USD",)))


class TestCCXTPoller:
    class _FakeExchange:
        def __init__(self):
            self.calls: list[str] = []

        def fetch_ohlcv(self, symbol, timeframe="1m", limit=5):
            self.calls.append(symbol)
            return [[1_700_000_000_000 + i * 60_000, 100.0 + i, 101, 99, 100.5, 10] for i in range(limit)]

    def test_poll_once_writes_csv(self, tmp_path):
        exchange = self._FakeExchange()
        config = CCXTIngestorConfig(
            exchange_id="fake",
            symbols=("BTC/USDT", "ETH/USDT"),
            output_csv=tmp_path / "prices.csv",
            timeframe="1m",
            limit=3,
            max_iterations=1,
            interval_seconds=0.0,
            exchange_factory=lambda exchange_id: exchange,
        )
        poller = CCXTPoller(config)
        appended = poller.poll_once()
        assert not appended.empty
        assert (tmp_path / "prices.csv").exists()
        assert exchange.calls == ["BTC/USDT", "ETH/USDT"]

    def test_run_executes_iterations(self, tmp_path):
        counter = {"n": 0}

        class _Sequenced:
            def fetch_ohlcv(self, symbol, timeframe="1m", limit=5):
                counter["n"] += 1
                return [[1_700_000_000_000 + (counter["n"] - 1) * 60_000, 100, 101, 99, 100.5, 10]]

        config = CCXTIngestorConfig(
            exchange_id="fake",
            symbols=("BTC/USDT",),
            output_csv=tmp_path / "prices.csv",
            timeframe="1m",
            limit=1,
            max_iterations=3,
            interval_seconds=0.0,
            exchange_factory=lambda exchange_id: _Sequenced(),
            sleep=lambda _seconds: None,
        )
        completed = CCXTPoller(config).run(max_iterations=3)
        assert completed == 3
        assert counter["n"] == 3

    def test_rejects_invalid_interval(self):
        with pytest.raises(ValueError):
            CCXTIngestorConfig(
                symbols=("BTC/USDT",), output_csv=None, interval_seconds=-1.0
            )

    def test_rejects_empty_symbols(self):
        with pytest.raises(ValueError):
            CCXTIngestorConfig(symbols=(), output_csv=None)

    def test_rejects_unsupported_timeframe(self):
        with pytest.raises(ValueError):
            CCXTIngestorConfig(symbols=("BTC/USDT",), output_csv=None, timeframe="3y")

    def test_default_factory_is_resolvable(self):
        factory = resolve_exchange_factory(None)
        assert factory is default_exchange_factory

    def test_resolve_exchange_factory_returns_passed(self):
        def sentinel(exchange_id):
            return None

        assert resolve_exchange_factory(sentinel) is sentinel


class TestPivotToPriceFrame:
    def test_pivot_long_to_wide(self, tmp_path):
        path = tmp_path / "long.csv"
        ts = 1_700_000_000_000
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [ts, ts + 60_000, ts, ts + 60_000], unit="ms", utc=True
                ),
                "symbol": ["BTC/USDT", "BTC/USDT", "ETH/USDT", "ETH/USDT"],
                "close": [100.0, 101.0, 50.0, 51.0],
            }
        ).to_csv(path, index=False)
        pivot = pivot_to_price_frame(path)
        assert list(pivot.columns) == ["BTC/USDT", "ETH/USDT"]

    def test_missing_value_column(self, tmp_path):
        path = tmp_path / "long.csv"
        pd.DataFrame({"date": [], "symbol": []}).to_csv(path, index=False)
        with pytest.raises(ValueError):
            pivot_to_price_frame(path, value_col="vwap")
