"""Price ingestion, validation, cleaning, and return-series construction.

Used by every ingestor that produces a long-form price frame
(``cps.infrastructure.ingestors.csv_ingestor``,
``...yfinance_ingestor``, ``...synthetic_ingestor``,
``...ccxt_ingestor``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataValidationConfig:
    """Tuning knobs for :func:`clean_price_data`.

    Attributes:
        max_missing_days: Maximum number of NaN values a column may
            have and still be kept. Defaults to ``10``.
        min_assets: Minimum number of asset columns the input must
            have before and after filtering. Defaults to ``4``.
    """

    max_missing_days: int = 10
    min_assets: int = 4


def load_price_data(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    """Parse a CSV file into a date-indexed price frame.

    Args:
        csv_path: Path to the CSV file.
        date_col: Name of the date column.

    Returns:
        ``pd.DataFrame`` indexed by ``pd.Timestamp`` with one column
        per asset, sorted ascending by date.

    Raises:
        ValueError: If the file is empty, ``date_col`` is missing, any
            column is duplicated, or any non-date column contains
            non-numeric values.
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in {csv_path}")
    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.sort_values(date_col).set_index(date_col)
    if df.empty:
        raise ValueError("Price data is empty")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate asset columns detected")
    if not np.issubdtype(df.to_numpy().dtype, np.number):
        raise ValueError("Price matrix contains non-numeric values")
    return df.astype(float)


def clean_price_data(prices: pd.DataFrame, config: DataValidationConfig) -> pd.DataFrame:
    """Apply the four-stage validation pipeline to a raw price frame.

    Stages: NaN column filter -> time interpolation -> positivity
    check -> minimum asset count.

    Args:
        prices: Raw price ``pd.DataFrame``.
        config: Validation configuration.

    Returns:
        A cleaned ``pd.DataFrame``.

    Raises:
        ValueError: If any stage fails.
    """
    if prices.shape[1] < config.min_assets:
        raise ValueError("Insufficient number of assets before filtering")
    missing_counts = prices.isna().sum(axis=0)
    keep_cols = missing_counts[missing_counts <= config.max_missing_days].index
    filtered = prices[keep_cols].copy()
    if filtered.empty:
        raise ValueError("No assets remaining after missing-value filtering")
    filtered = filtered.interpolate(method="time").ffill().bfill()
    if filtered.isna().any().any():
        raise ValueError("NaN values remain after interpolation")
    if (filtered <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if filtered.shape[1] < config.min_assets:
        raise ValueError("Insufficient number of assets after filtering")
    return filtered


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns from a cleaned price frame.

    Applies ``r_t = log(P_t / P_{t-1})`` and drops any residual ``NaN``
    or ``inf`` rows.

    Args:
        prices: Cleaned price frame.

    Returns:
        ``pd.DataFrame`` of log-returns.

    Raises:
        ValueError: When no log-returns could be computed.
    """
    lr = np.log(prices / prices.shift(1)).dropna(how="all")
    if lr.empty:
        raise ValueError("No log-returns could be computed")
    return lr.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def market_proxy(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional benchmark return series."""
    return returns.mean(axis=1)


__all__ = [
    "DataValidationConfig",
    "clean_price_data",
    "load_price_data",
    "log_returns",
    "market_proxy",
]