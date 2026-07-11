"""Real-time OHLCV ingestion via ccxt REST polling.

This module provides a small polling loop that pulls the latest OHLCV candle
for a list of symbols from a single ccxt exchange and appends results to a
long-form CSV file on disk. The poller is intentionally simple:

* No asyncio / threading primitives; :func:`poll_once` does a single REST
  call and writes the row(s). :func:`run_polling_loop` runs it in a
  loop with a bounded retry.
* Each row is de-duplicated by ``(timestamp, symbol)`` so partial
  overlaps do not corrupt the CSV.
* The output CSV matches the format consumed by
  :func:`cps.data.load_price_data` so the rest of the system can ingest
  it without modification.

Install the optional extra with::

    pip install 'crypto-portfolio-system[realtime]'

Typical use::

    from cps.realtime import CCXTPollerConfig, run_polling_loop
    config = CCXTPollerConfig(
        exchange_id="binance",
        symbols=("BTC/USDT", "ETH/USDT"),
        output_csv=Path("prices.csv"),
        interval_seconds=60,
        timeframe="1m",
    )
    run_polling_loop(config, max_iterations=10)

Long-form CSV contract
----------------------
Each poll appends one row per ``(timestamp, symbol)`` pair::

    date,symbol,open,high,low,close,volume
    2024-05-01T00:00:00+00:00,BTC/USDT,...

The ``date`` column is ISO-8601 with timezone. Use
:func:`pivot_to_price_frame` to convert the long-form frame into the
wide price matrix that the pipeline expects.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .resilience import RetryConfig, execute_with_retry


def _require_ccxt() -> None:
    """Lazy guard for the optional ``ccxt`` dependency.

    Raises:
        RuntimeError: With a message instructing the caller to install
            the ``[realtime]`` extra.
    """
    try:
        import ccxt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise RuntimeError(
            "The ccxt real-time ingestor requires the 'ccxt' package. "
            "Install the optional extra with: pip install 'crypto-portfolio-system[realtime]'"
        ) from exc


@dataclass(frozen=True)
class CCXTPollerConfig:
    """Configuration for the ccxt-based real-time ingestor.

    Attributes:
        exchange_id: ccxt exchange identifier (e.g. ``"binance"``).
        symbols: Tuple of ccxt-formatted symbols
            (e.g. ``("BTC/USDT", "ETH/USDT")``). At least one is
            required.
        output_csv: Destination long-form CSV path. The poller creates
            parent directories as needed.
        timeframe: ccxt candle timeframe. Must be one of the supported
            literals (``"1m"`` ... ``"1M"``). Defaults to ``"1m"``.
        interval_seconds: Sleep between successive iterations. Use
            ``0`` for back-to-back polling. Defaults to ``60.0``.
        limit: Number of candles requested per REST call. Defaults to
            ``5``.
        max_iterations: Default iteration count when
            :func:`run_polling_loop` is invoked without an explicit
            override. Defaults to ``1`` so a misconfigured scheduler
            cannot accidentally hammer the exchange.
        date_col: Name of the timestamp column written to the CSV.
            Defaults to ``"date"``.
        retry: :class:`cps.resilience.RetryConfig` applied to each REST
            call.
    """

    exchange_id: str = "binance"
    symbols: tuple[str, ...] = field(default_factory=tuple)
    output_csv: Path | None = None
    timeframe: str = "1m"
    interval_seconds: float = 60.0
    limit: int = 5
    max_iterations: int = 1
    date_col: str = "date"
    retry: RetryConfig = field(default_factory=lambda: RetryConfig(max_attempts=3, initial_backoff_seconds=1.0))


def _build_exchange(exchange_id: str) -> Any:
    """Construct a ccxt exchange instance.

    ``enableRateLimit=True`` is set so ccxt self-throttles requests to
    respect exchange-side rate limits -- without this, a busy poller
    quickly accrues HTTP 429 responses.

    Args:
        exchange_id: ccxt exchange identifier (e.g. ``"binance"``).

    Returns:
        An instantiated ccxt exchange object.
    """
    import ccxt

    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def _fetch_ohlcv(exchange: Any, symbol: str, timeframe: str, limit: int) -> list[list[Any]]:
    """Thin wrapper around ``exchange.fetch_ohlcv``.

    Centralised so retry / error handling around the REST call lives in
    exactly one place.

    Args:
        exchange: ccxt exchange instance.
        symbol: Symbol string (e.g. ``"BTC/USDT"``).
        timeframe: Candle timeframe.
        limit: Maximum number of candles to return.

    Returns:
        List of ``[timestamp, open, high, low, close, volume]`` rows as
        returned by ccxt.
    """
    return list(exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))


def _fetch_for_symbol(exchange: Any, symbol: str, timeframe: str, limit: int) -> Callable[[], list[list[Any]]]:
    """Return a zero-argument callable that fetches one symbol.

    The closure pattern lets :func:`cps.resilience.execute_with_retry`
    invoke a fresh call per attempt without carrying stale closure
    state. Returning a named ``Callable`` (rather than a lambda) keeps
    mypy happy about the return type.
    """

    def _do_fetch() -> list[list[Any]]:
        return _fetch_ohlcv(exchange, symbol, timeframe, limit)

    return _do_fetch


def poll_once(config: CCXTPollerConfig) -> pd.DataFrame:
    """Run a single polling iteration and return the rows that were appended.

    The CSV at ``config.output_csv`` is read (if present), the latest candles
    for each symbol are appended, the merged frame is written back to disk,
    and the rows added during this iteration are returned.

    Args:
        config: :class:`CCXTPollerConfig` driving this iteration.

    Returns:
        ``pd.DataFrame`` of the rows appended during this iteration. May
        be empty when every symbol returned no data.

    Raises:
        ValueError: When ``symbols`` is empty, ``interval_seconds < 0``,
            or ``timeframe`` is not one of the ccxt-supported values.
    """
    _require_ccxt()
    if not config.symbols:
        raise ValueError("At least one symbol is required")
    if config.interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")
    if config.timeframe not in {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"}:
        raise ValueError(f"Unsupported timeframe: {config.timeframe}")

    exchange = _build_exchange(config.exchange_id)
    frames: list[pd.DataFrame] = []
    for symbol in config.symbols:
        candles = execute_with_retry(_fetch_for_symbol(exchange, symbol, config.timeframe, config.limit), config.retry)
        if not candles:
            continue
        rows = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        rows[config.date_col] = pd.to_datetime(rows["timestamp"], unit="ms", utc=True)
        rows["symbol"] = symbol
        frames.append(rows)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[[config.date_col, "symbol", "open", "high", "low", "close", "volume"]]

    if config.output_csv is not None:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if config.output_csv.exists():
            existing = pd.read_csv(config.output_csv)
            # Force the existing ``date`` column to a UTC-aware datetime
            # so the dedupe key matches the freshly fetched rows
            # (``existing`` is read as a string otherwise).
            existing[config.date_col] = pd.to_datetime(existing[config.date_col], utc=True)
            combined = pd.concat([existing, combined], ignore_index=True)
        # Dedup on ``(timestamp, symbol)`` so repeated polls don't pile
        # up duplicate rows; ``keep="last"`` prefers the freshest copy.
        combined = combined.drop_duplicates(subset=[config.date_col, "symbol"], keep="last")
        combined = combined.sort_values([config.date_col, "symbol"]).reset_index(drop=True)
        combined.to_csv(config.output_csv, index=False)

    return combined


def run_polling_loop(config: CCXTPollerConfig, max_iterations: int | None = None) -> int:
    """Run the poller in a loop until ``max_iterations`` is reached.

    Args:
        config: :class:`CCXTPollerConfig` driving the loop.
        max_iterations: Iteration count override. When ``None``, the
            value is taken from ``config.max_iterations``.

    Returns:
        Number of iterations actually executed.

    Raises:
        ValueError: When ``iterations < 1``.
    """
    iterations = max_iterations if max_iterations is not None else config.max_iterations
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    completed = 0
    for iteration_index in range(iterations):
        del iteration_index  # Loop variable is intentional; suppress ruff's F841.
        poll_once(config)
        completed += 1
        if completed < iterations:
            # Sleep between iterations but never after the last one --
            # the caller may want to chain into other work immediately.
            time.sleep(config.interval_seconds)
    return completed


def pivot_to_price_frame(csv_path: str | Path, *, date_col: str = "date", value_col: str = "close") -> pd.DataFrame:
    """Read the long-form ccxt CSV and pivot to a wide price frame.

    The output has one column per symbol with the selected value column
    and is indexed by the parsed date column. This matches the format
    consumed by :func:`cps.data.load_price_data` and
    :func:`cps.pipeline.run_pipeline`.

    Args:
        csv_path: Path to the long-form CSV produced by the poller.
        date_col: Name of the timestamp column. Defaults to ``"date"``.
        value_col: Name of the column to pivot (e.g. ``"close"``,
            ``"Adj Close"``). Defaults to ``"close"``.

    Returns:
        ``pd.DataFrame`` indexed by date with one column per symbol.
        Missing values are forward-filled and any leading ``NaN`` rows
        are dropped.

    Raises:
        ValueError: When ``value_col`` is not present in the CSV.
    """
    frame = pd.read_csv(csv_path, parse_dates=[date_col])
    if value_col not in frame.columns:
        raise ValueError(f"Column '{value_col}' not present in {csv_path}")
    pivot = frame.pivot_table(index=date_col, columns="symbol", values=value_col, aggfunc="last")
    pivot = pivot.sort_index()
    pivot = pivot.ffill().dropna(how="all")
    return pivot
