from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ccxt = pytest.importorskip("ccxt")

from cps.realtime import CCXTPollerConfig, pivot_to_price_frame, poll_once, run_polling_loop  # noqa: E402
from cps.resilience import RetryConfig  # noqa: E402


class _FakeExchange:
    def __init__(self, candles_by_symbol: dict[str, list[list[float]]] | None = None) -> None:
        self._candles = candles_by_symbol or {}
        self.calls: list[str] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 5):
        self.calls.append(symbol)
        return self._candles.get(symbol, [])


def _fake_candles(symbol: str, count: int = 3) -> list[list[float]]:
    base_ts = 1_700_000_000_000
    return [[base_ts + i * 60_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i] for i in range(count)]


def test_poll_once_writes_csv(tmp_path: Path):
    exchange = _FakeExchange({"BTC/USDT": _fake_candles("BTC/USDT"), "ETH/USDT": _fake_candles("ETH/USDT")})
    config = CCXTPollerConfig(
        exchange_id="fake",
        symbols=("BTC/USDT", "ETH/USDT"),
        output_csv=tmp_path / "prices.csv",
        timeframe="1m",
        limit=3,
        max_iterations=1,
        interval_seconds=0.0,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("cps.realtime._build_exchange", lambda exchange_id: exchange)
        appended = poll_once(config)
    assert not appended.empty
    csv_path = tmp_path / "prices.csv"
    assert csv_path.exists()
    frame = pd.read_csv(csv_path)
    assert set(frame["symbol"]) == {"BTC/USDT", "ETH/USDT"}


def test_poll_once_dedupes_existing_rows(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    base_ts = 1_700_000_000_000
    initial = pd.DataFrame(
        {
            "date": pd.to_datetime([base_ts], unit="ms", utc=True),
            "symbol": ["BTC/USDT"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    )
    initial.to_csv(csv_path, index=False)

    exchange = _FakeExchange({"BTC/USDT": _fake_candles("BTC/USDT", count=2)})
    config = CCXTPollerConfig(
        exchange_id="fake",
        symbols=("BTC/USDT",),
        output_csv=csv_path,
        timeframe="1m",
        limit=2,
        max_iterations=1,
        interval_seconds=0.0,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("cps.realtime._build_exchange", lambda exchange_id: exchange)
        poll_once(config)

    frame = pd.read_csv(csv_path)
    assert len(frame) == 2
    assert frame["symbol"].nunique() == 1


def test_poll_once_requires_symbols(tmp_path: Path):
    config = CCXTPollerConfig(
        exchange_id="fake",
        symbols=(),
        output_csv=tmp_path / "out.csv",
    )
    with pytest.raises(ValueError):
        poll_once(config)


def test_poll_once_validates_interval(tmp_path: Path):
    config = CCXTPollerConfig(
        exchange_id="fake",
        symbols=("BTC/USDT",),
        output_csv=tmp_path / "out.csv",
        interval_seconds=-1.0,
    )
    with pytest.raises(ValueError):
        poll_once(config)


def test_poll_once_validates_timeframe(tmp_path: Path):
    config = CCXTPollerConfig(
        exchange_id="fake",
        symbols=("BTC/USDT",),
        output_csv=tmp_path / "out.csv",
        timeframe="3y",
    )
    with pytest.raises(ValueError):
        poll_once(config)


def test_run_polling_loop_executes_iterations(tmp_path: Path):
    base_ts = 1_700_000_000_000
    iteration_calls = {"n": 0}

    class _SequencedExchange:
        def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 5):
            iteration_calls["n"] += 1
            offset = (iteration_calls["n"] - 1) * 60_000
            return [[base_ts + offset, 100.0, 101.0, 99.0, 100.5, 10.0]]

    config = CCXTPollerConfig(
        exchange_id="fake",
        symbols=("BTC/USDT",),
        output_csv=tmp_path / "prices.csv",
        timeframe="1m",
        limit=1,
        max_iterations=3,
        interval_seconds=0.0,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("cps.realtime._build_exchange", lambda exchange_id: _SequencedExchange())
        monkeypatch.setattr("cps.realtime.time.sleep", lambda _seconds: None)
        completed = run_polling_loop(config, max_iterations=3)
    assert completed == 3
    frame = pd.read_csv(tmp_path / "prices.csv")
    assert len(frame) == 3


def test_run_polling_loop_rejects_zero_iterations():
    with pytest.raises(ValueError):
        run_polling_loop(CCXTPollerConfig(), max_iterations=0)


def test_pivot_to_price_frame(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    base_ts = 1_700_000_000_000
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([base_ts, base_ts + 60_000, base_ts, base_ts + 60_000], unit="ms", utc=True),
            "symbol": ["BTC/USDT", "BTC/USDT", "ETH/USDT", "ETH/USDT"],
            "open": [100.0, 101.0, 50.0, 51.0],
            "high": [102.0, 103.0, 52.0, 53.0],
            "low": [99.0, 100.0, 49.0, 50.0],
            "close": [100.5, 101.5, 50.5, 51.5],
            "volume": [10.0, 11.0, 5.0, 5.5],
        }
    )
    frame.to_csv(csv_path, index=False)
    pivot = pivot_to_price_frame(csv_path)
    assert list(pivot.columns) == ["BTC/USDT", "ETH/USDT"]
    assert pivot.shape[0] == 2


def test_pivot_to_price_frame_missing_value_column(tmp_path: Path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame({"date": [], "symbol": [], "open": []}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        pivot_to_price_frame(csv_path, value_col="vwap")


def test_realtime_cli_invokes_poller(tmp_path: Path, monkeypatch):
    from cps import cli

    csv_path = tmp_path / "prices.csv"
    argv = [
        "--exchange",
        "fake",
        "--symbols",
        "BTC/USDT",
        "--output-csv",
        str(csv_path),
        "--max-iterations",
        "1",
        "--interval-seconds",
        "0",
    ]
    fake_exchange = _FakeExchange({"BTC/USDT": _fake_candles("BTC/USDT", count=1)})
    monkeypatch.setattr("cps.realtime._build_exchange", lambda exchange_id: fake_exchange)
    monkeypatch.setattr("cps.realtime.time.sleep", lambda _seconds: None)

    exit_code = cli.realtime_main(argv)
    assert exit_code == 0
    assert csv_path.exists()


def test_realtime_cli_rejects_empty_symbols():
    from cps import cli

    with pytest.raises(SystemExit):
        cli.realtime_main(["cps-realtime", "--symbols", "", "--output-csv", "x.csv"])


def test_realtime_cli_uses_retry_config(tmp_path: Path, monkeypatch):
    from cps import cli

    csv_path = tmp_path / "prices.csv"
    captured: dict[str, RetryConfig] = {}

    def fake_run(config: CCXTPollerConfig, max_iterations: int | None = None) -> int:
        captured["retry"] = config.retry
        return 1

    monkeypatch.setattr("cps.realtime.run_polling_loop", fake_run)
    cli.realtime_main(
        [
            "--symbols",
            "BTC/USDT",
            "--output-csv",
            str(csv_path),
            "--max-attempts",
            "7",
            "--initial-backoff",
            "0.5",
        ]
    )
    assert captured["retry"].max_attempts == 7
    assert captured["retry"].initial_backoff_seconds == 0.5
