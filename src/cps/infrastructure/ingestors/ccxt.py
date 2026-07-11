"""ccxt-based real-time OHLCV poller."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from ...config.settings import CCXT_RATE_LIMIT_OPTION, CCXT_SUPPORTED_TIMEFRAMES
from ...domain.protocols import ExchangeFactory, SleepCallable
from ...infrastructure.resilience import RetryPolicy, execute_with_retry, require_optional
from ...infrastructure.stores import LongFormCsvStore


def default_exchange_factory(exchange_id: str) -> Any:
    """Build a ccxt exchange instance with rate-limit self-throttling."""
    ccxt = require_optional("ccxt", "realtime")
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": CCXT_RATE_LIMIT_OPTION})


def default_sleep(seconds: float) -> None:
    """Default ``sleep`` callable: forwards to :func:`time.sleep`."""
    time.sleep(seconds)


def resolve_exchange_factory(factory: ExchangeFactory | None) -> ExchangeFactory:
    """Return ``factory`` or :func:`default_exchange_factory` when ``factory`` is ``None``."""
    return factory if factory is not None else default_exchange_factory


def resolve_sleep(sleep: SleepCallable | None) -> SleepCallable:
    """Return ``sleep`` or :func:`default_sleep` when ``sleep`` is ``None``."""
    return sleep if sleep is not None else default_sleep


@dataclass(frozen=True)
class CCXTIngestorConfig:
    """Configuration for the ccxt-backed real-time ingestor."""

    exchange_id: str = "binance"
    symbols: tuple[str, ...] = field(default_factory=tuple)
    output_csv: Path | None = None
    timeframe: str = "1m"
    interval_seconds: float = 60.0
    limit: int = 5
    max_iterations: int = 1
    date_col: str = "date"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    exchange_factory: ExchangeFactory | None = None
    sleep: SleepCallable | None = None

    def __post_init__(self) -> None:
        """Validate invariants that should hold before the poller runs."""
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        if self.interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        if self.timeframe not in CCXT_SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")


class CCXTPoller:
    """Stateless polling loop for one ccxt exchange."""

    name: ClassVar[str] = "ccxt"

    def __init__(self, config: CCXTIngestorConfig) -> None:
        """Initialise the poller with a config validated at construction."""
        self.__config = config
        self.__exchange_factory = resolve_exchange_factory(config.exchange_factory)
        self.__sleep = resolve_sleep(config.sleep)
        self.__store = (
            LongFormCsvStore(config.output_csv, date_col=config.date_col)
            if config.output_csv is not None
            else None
        )

    @property
    def config(self) -> CCXTIngestorConfig:
        """Return the poller's configuration."""
        return self.__config

    def poll_once(self) -> pd.DataFrame:
        """Run a single polling iteration and return the rows added."""
        exchange = self.__exchange_factory(self.__config.exchange_id)
        frames: list[pd.DataFrame] = []
        for symbol in self.__config.symbols:
            candles = self.__fetch_with_retry(exchange, symbol)
            if not candles:
                continue
            rows = pd.DataFrame(
                candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            rows[self.__config.date_col] = pd.to_datetime(rows["timestamp"], unit="ms", utc=True)
            rows["symbol"] = symbol
            frames.append(rows)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined = combined[
            [self.__config.date_col, "symbol", "open", "high", "low", "close", "volume"]
        ]
        if self.__store is not None:
            return self.__store.append(combined)
        return combined

    def run(self, max_iterations: int | None = None) -> int:
        """Run the poller in a loop until ``max_iterations`` is reached."""
        iterations = max_iterations if max_iterations is not None else self.__config.max_iterations
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        completed = 0
        for _ in range(iterations):
            self.poll_once()
            completed += 1
            if completed < iterations:
                self.__sleep(self.__config.interval_seconds)
        return completed

    def __fetch_with_retry(self, exchange: Any, symbol: str) -> list[list[Any]]:
        """Fetch candles for one symbol, applying the retry policy."""
        return list(
            execute_with_retry(
                lambda sym=symbol: exchange.fetch_ohlcv(
                    sym, timeframe=self.__config.timeframe, limit=self.__config.limit
                ),
                self.__config.retry,
            )
        )


def pivot_to_price_frame(
    csv_path: str | Path,
    *,
    date_col: str = "date",
    value_col: str = "close",
) -> pd.DataFrame:
    """Read the long-form ccxt CSV and pivot to a wide price frame."""
    frame = pd.read_csv(csv_path, parse_dates=[date_col])
    if value_col not in frame.columns:
        raise ValueError(f"Column '{value_col}' not present in {csv_path}")
    pivot = frame.pivot_table(index=date_col, columns="symbol", values=value_col, aggfunc="last")
    pivot = pivot.sort_index()
    pivot = pivot.ffill().dropna(how="all")
    return pivot


__all__ = [
    "CCXTIngestorConfig",
    "CCXTPoller",
    "default_exchange_factory",
    "default_sleep",
    "pivot_to_price_frame",
    "resolve_exchange_factory",
    "resolve_sleep",
]