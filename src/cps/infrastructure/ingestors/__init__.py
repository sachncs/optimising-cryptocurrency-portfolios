"""Ingestor implementations and registry."""

from .ccxt import (
    CCXTIngestorConfig,
    CCXTPoller,
    default_exchange_factory,
    default_sleep,
    pivot_to_price_frame,
    resolve_exchange_factory,
    resolve_sleep,
)
from .csv import CsvIngestor
from .synthetic import SyntheticIngestor
from .yfinance import (
    YFinanceConfig,
    YFinanceField,
    YFinanceIngestor,
    YFinanceInterval,
    fetch_yfinance_prices,
)

__all__ = [
    "CCXTIngestorConfig",
    "CCXTPoller",
    "CsvIngestor",
    "SyntheticIngestor",
    "YFinanceConfig",
    "YFinanceField",
    "YFinanceIngestor",
    "YFinanceInterval",
    "default_exchange_factory",
    "default_sleep",
    "fetch_yfinance_prices",
    "pivot_to_price_frame",
    "resolve_exchange_factory",
    "resolve_sleep",
]
