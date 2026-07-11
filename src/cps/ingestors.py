"""Multi-asset price ingestors.

This module contains pluggable price-data ingestors that produce a
long-form ``pd.DataFrame`` indexed by date with one column per asset.
The current ingestor pulls daily OHLCV data from Yahoo! Finance via the
optional ``yfinance`` package.

Install the optional extra with::

    pip install 'crypto-portfolio-system[ingestors]'

The ingestor is intentionally side-effect free beyond returning a
DataFrame; callers decide whether to write to disk, feed the pipeline,
or stream elsewhere.

Contract
--------
The returned frame is indexed by a sorted, deduplicated ``DatetimeIndex``
with one column per requested symbol. All values are strictly positive
(negatives would corrupt log-return computation downstream) and partial
failures -- a single symbol returning an empty series -- are surfaced as
a ``ValueError`` that lists every missing ticker, so the caller can
abort or fall back as appropriate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

YFinanceInterval = Literal["1d", "1h", "1m", "5m", "15m", "30m", "60m", "1wk", "1mo"]
YFinanceField = Literal["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass(frozen=True)
class YFinanceIngestorConfig:
    """Configuration for the yfinance-backed ingestor.

    Attributes:
        symbols: Tuple of Yahoo! tickers (e.g. ``("BTC-USD", "ETH-USD")``).
            At least one is required.
        start: ISO-8601 start date (``"YYYY-MM-DD"``). Mutually
            compatible with ``period`` -- when both are supplied,
            ``start`` wins.
        end: ISO-8601 end date (``"YYYY-MM-DD"``). Defaults to "today".
        period: Yahoo! period string (e.g. ``"1mo"``, ``"6mo"``,
            ``"1y"``, ``"max"``). Used when ``start`` / ``end`` are not
            supplied.
        interval: Yahoo! candle interval. Defaults to ``"1d"``.
        field: Which OHLCV column to keep. Defaults to ``"Close"``.
            ``"Adj Close"`` is recommended for total-return analyses
            because it accounts for splits and dividends.
        auto_adjust: Whether to let Yahoo! auto-adjust prices. Defaults
            to ``False`` so the ``field`` selection is honoured
            verbatim.
    """

    symbols: tuple[str, ...]
    start: str | None = None
    end: str | None = None
    period: str | None = None
    interval: YFinanceInterval = "1d"
    field: YFinanceField = "Close"
    auto_adjust: bool = False


def _require_yfinance() -> None:
    """Lazy guard for the optional ``yfinance`` dependency.

    Raises:
        RuntimeError: With a message instructing the caller to install
            the ``[ingestors]`` extra.
    """
    try:
        import yfinance  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise RuntimeError(
            "The yfinance ingestor requires the 'yfinance' package. "
            "Install the optional extra with: pip install 'crypto-portfolio-system[ingestors]'"
        ) from exc


def fetch_yfinance_prices(config: YFinanceIngestorConfig) -> pd.DataFrame:
    """Fetch price history for one or more symbols from Yahoo! Finance.

    The returned frame is indexed by timestamp (UTC-naive, sorted, deduplicated)
    with one column per symbol. Symbols that fail to fetch or produce empty
    series raise a ``ValueError`` listing all failures so the caller can
    decide whether to abort or fall back.

    Args:
        config: :class:`YFinanceIngestorConfig` describing the query.

    Returns:
        ``pd.DataFrame`` indexed by ``pd.Timestamp`` with one column
        per symbol, all values strictly positive.

    Raises:
        ValueError: When no symbols are supplied, when neither ``start``
            / ``end`` nor ``period`` is supplied, when the yfinance
            response is empty or contains only nulls for ``field``, when
            one or more symbols are missing from the response, or when
            any returned value is non-positive.

    Examples:
        >>> config = YFinanceIngestorConfig(
        ...     symbols=("BTC-USD", "ETH-USD"),
        ...     period="1mo",
        ... )
        >>> prices = fetch_yfinance_prices(config)
        >>> prices.head()
                          BTC-USD   ETH-USD
        Date
        2024-01-01   42210.5    2201.3
        ...
    """
    _require_yfinance()
    if not config.symbols:
        raise ValueError("At least one symbol is required")
    if config.start is None and config.end is None and config.period is None:
        raise ValueError("One of 'start', 'end', or 'period' must be supplied")

    import yfinance as yf

    # Build the kwargs dict dynamically so the yfinance call receives
    # exactly one of the ``period`` / ``start`` / ``end`` triples.
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
        # ``yfinance`` returns a wide frame with a MultiIndex on the
        # columns axis: ``(field, ticker)``. Slice out the requested
        # field across every ticker.
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
        # Aggregate every missing ticker into a single error so the
        # caller sees the complete failure picture at once.
        raise ValueError(f"yfinance returned no data for symbols={missing}")

    result = field_frame[list(config.symbols)].copy()
    # ``to_datetime`` without ``utc=True`` keeps the index tz-naive to
    # match the CSV-loaded frames produced by ``cps.data.load_price_data``.
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    if (result <= 0).any().any():
        # Non-positive prices corrupt log-return computation downstream;
        # fail fast rather than produce silently wrong returns.
        raise ValueError("yfinance returned non-positive prices")
    return result


def fetch_yfinance_symbols(
    symbols: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: YFinanceInterval = "1d",
    field: YFinanceField = "Close",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper around :func:`fetch_yfinance_prices`.

    Args:
        symbols: Sequence of Yahoo! tickers.
        start: Optional start date.
        end: Optional end date.
        period: Optional period string.
        interval: Candle interval. Defaults to ``"1d"``.
        field: OHLCV field to keep. Defaults to ``"Close"``.
        auto_adjust: Whether to let Yahoo! auto-adjust. Defaults to
            ``False``.

    Returns:
        ``pd.DataFrame`` indexed by date with one column per symbol.
    """
    config = YFinanceIngestorConfig(
        symbols=tuple(symbols),
        start=start,
        end=end,
        period=period,
        interval=interval,
        field=field,
        auto_adjust=auto_adjust,
    )
    return fetch_yfinance_prices(config)
