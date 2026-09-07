"""Yahoo! Finance ingestor.

Returns a wide price frame indexed by date with one column per
symbol, all values strictly positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import pandas as pd

from ...infrastructure.resilience import require_optional

YFinanceInterval = Literal["1d", "1h", "1m", "5m", "15m", "30m", "60m", "1wk", "1mo"]
YFinanceField = Literal["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass(frozen=True)
class YFinanceConfig:
    """Configuration for the Yahoo! Finance ingestor."""

    symbols: tuple[str, ...]
    start: str | None = None
    end: str | None = None
    period: str | None = None
    interval: YFinanceInterval = "1d"
    field: YFinanceField = "Close"
    auto_adjust: bool = False

    def __post_init__(self) -> None:
        """Validate that the config has the inputs needed to fetch anything."""
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        if self.start is None and self.end is None and self.period is None:
            raise ValueError("One of 'start', 'end', or 'period' must be supplied")


class YFinanceIngestor:
    """Fetch price history from Yahoo! Finance for the configured symbols."""

    name: ClassVar[str] = "yfinance"

    def __init__(self, config: YFinanceConfig) -> None:
        """Initialise the ingestor with a :class:`YFinanceConfig`."""
        self.__config = config

    def fetch(self) -> pd.DataFrame:
        """Return the fetched price frame.

        Raises:
            ValueError: On invalid configuration or partial fetches.
            RuntimeError: When the yfinance package is not installed.
        """
        return fetch_yfinance_prices(self.__config)


def fetch_yfinance_prices(config: YFinanceConfig | None = None, /, **kwargs: object) -> pd.DataFrame:
    """Fetch price history for one or more symbols from Yahoo! Finance.

    Args:
        config: Optional :class:`YFinanceConfig`.
        **kwargs: When ``config`` is ``None``, supply the symbols and
            other parameters here.

    Returns:
        ``pd.DataFrame`` indexed by ``pd.Timestamp`` with one column
        per symbol.

    Raises:
        ValueError: When ``symbols`` is empty, ``period``/``start``/``end``
            are all missing, the yfinance response is empty, one or more
            symbols are missing, or any value is non-positive.
        RuntimeError: When yfinance is not installed.
    """
    if config is None:
        if "symbols" not in kwargs:
            raise ValueError("At least one symbol is required")
        config = YFinanceConfig(**kwargs)  # type: ignore[arg-type]  # kwargs is filtered at call sites

    yf = require_optional("yfinance", "ingestors")

    if not config.symbols:
        raise ValueError("At least one symbol is required")
    if config.start is None and config.end is None and config.period is None:
        raise ValueError("One of 'start', 'end', or 'period' must be supplied")

    download_kwargs: dict[str, object] = {
        "interval": config.interval,
        "auto_adjust": config.auto_adjust,
        "progress": False,
        "threads": True,
    }
    if config.period is not None:
        download_kwargs["period"] = config.period
    else:
        if config.start is not None:
            download_kwargs["start"] = config.start
        if config.end is not None:
            download_kwargs["end"] = config.end

    raw = yf.download(list(config.symbols), **download_kwargs)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for symbols={list(config.symbols)}")

    try:
        field_frame = raw.xs(config.field, axis=1, level=0)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"yfinance response does not contain '{config.field}' field for symbols={list(config.symbols)}"
        ) from exc

    field_frame = field_frame.dropna(how="all")
    if field_frame.empty:
        raise ValueError(f"yfinance returned only null '{config.field}' rows for symbols={list(config.symbols)}")

    missing = [symbol for symbol in config.symbols if symbol not in field_frame.columns]
    if missing:
        raise ValueError(f"yfinance returned no data for symbols={missing}")

    result = field_frame[list(config.symbols)].copy()
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    if (result <= 0).any().any():
        raise ValueError("yfinance returned non-positive prices")
    return result


__all__ = [
    "YFinanceConfig",
    "YFinanceField",
    "YFinanceIngestor",
    "YFinanceInterval",
    "fetch_yfinance_prices",
]
