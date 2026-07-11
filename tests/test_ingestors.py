from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

yfinance = pytest.importorskip("yfinance")

from cps.ingestors import (  # noqa: E402
    YFinanceIngestorConfig,
    fetch_yfinance_prices,
    fetch_yfinance_symbols,
)


def _fake_download(
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = False,
    threads: bool = True,
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], symbols],
        names=["field", "Ticker"],
    )
    rows = np.ones((len(dates), len(columns)))
    return pd.DataFrame(rows, index=dates, columns=columns)


def test_fetch_yfinance_prices_basic():
    config = YFinanceIngestorConfig(symbols=("BTC-USD", "ETH-USD"), period="1mo")
    with patch.object(yfinance, "download", side_effect=_fake_download) as mock_download:
        frame = fetch_yfinance_prices(config)
    assert mock_download.called
    assert list(frame.columns) == ["BTC-USD", "ETH-USD"]
    assert frame.shape[0] == 5
    assert (frame > 0).all().all()


def test_fetch_yfinance_symbols_helper():
    with patch.object(yfinance, "download", side_effect=_fake_download):
        frame = fetch_yfinance_symbols(("AAPL",), start="2024-01-01", end="2024-01-10")
    assert frame.shape[1] == 1
    assert "AAPL" in frame.columns


def test_fetch_yfinance_rejects_empty_symbols():
    with pytest.raises(ValueError):
        fetch_yfinance_prices(YFinanceIngestorConfig(symbols=()))


def test_fetch_yfinance_requires_period_or_window():
    with pytest.raises(ValueError):
        fetch_yfinance_prices(YFinanceIngestorConfig(symbols=("BTC-USD",)))


def test_fetch_yfinance_raises_on_missing_symbols():
    def fake_download_missing(*args: object, **kwargs: object) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL"]],
            names=["field", "Ticker"],
        )
        return pd.DataFrame(np.ones((len(dates), len(columns))), index=dates, columns=columns)

    with (
        patch.object(yfinance, "download", side_effect=fake_download_missing),
        pytest.raises(ValueError, match="no data for symbols"),
    ):
        fetch_yfinance_prices(YFinanceIngestorConfig(symbols=("AAPL", "MISSING"), period="1mo"))


def test_cli_source_yfinance_writes_csv(tmp_path, monkeypatch):
    from cps import cli

    out_dir = tmp_path / "out"
    ingest_csv = tmp_path / "prices.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-portfolio",
            "--source",
            "yfinance",
            "--symbols",
            "BTC-USD,ETH-USD",
            "--period",
            "5d",
            "--output-dir",
            str(out_dir),
            "--run-dir",
            str(out_dir / "runs"),
            "--train-window-days",
            "5",
            "--corr-window-days",
            "3",
            "--rebalance-step-days",
            "2",
            "--horizons",
            "1",
            "--forecast-method",
            "naive",
            "--consensus-runs",
            "2",
            "--min-assets",
            "2",
            "--max-assets",
            "5",
            "--max-volatility-annual",
            "5.0",
            "--ingest-output-csv",
            str(ingest_csv),
        ],
    )

    def fake_fetch(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(
            np.linspace(100.0, 110.0, num=10).reshape(-1, 1).repeat(len(symbols), axis=1),
            index=dates,
            columns=symbols,
        )

    with patch("cps.ingestors.fetch_yfinance_symbols", side_effect=fake_fetch):
        cli.main()

    assert ingest_csv.exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "trades.csv").exists()


def test_cli_yfinance_missing_symbols(monkeypatch, tmp_path, capsys):
    from cps import cli

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-portfolio",
            "--source",
            "yfinance",
            "--output-dir",
            str(out_dir),
            "--run-dir",
            str(out_dir / "runs"),
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
    captured = capsys.readouterr()
    assert "--symbols is required" in captured.err


def test_cli_csv_source_requires_path(monkeypatch, tmp_path, capsys):
    from cps import cli

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crypto-portfolio",
            "--source",
            "csv",
            "--output-dir",
            str(out_dir),
            "--run-dir",
            str(out_dir / "runs"),
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
    captured = capsys.readouterr()
    assert "--prices-csv is required" in captured.err


def test_cli_resolve_source_logic():
    from cps import cli

    args = type("Args", (), {"source": "auto", "symbols": "", "prices_csv": ""})()
    assert cli._resolve_source(args) == "synthetic"

    args.symbols = "BTC-USD"
    assert cli._resolve_source(args) == "yfinance"

    args.symbols = ""
    args.prices_csv = "x.csv"
    assert cli._resolve_source(args) == "csv"

    args.prices_csv = ""
    args.source = "yfinance"
    assert cli._resolve_source(args) == "yfinance"
