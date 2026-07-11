"""Price ingestion, validation, cleaning, and return-series construction.

This module is the boundary between raw CSV price data on disk and the
pure-pandas / pure-NumPy data structures consumed by the rest of the
package. It exposes four entry points used by the CLI and the API:

* :func:`load_price_data` -- parse a CSV file into a
  ``pd.DataFrame`` indexed by date.
* :func:`clean_price_data` -- apply the validation pipeline (missing-data
  filtering, time interpolation, positivity checks, minimum asset count).
* :func:`log_returns` -- convert cleaned prices into log-returns and drop
  any residual rows containing ``NaN`` / ``inf``.
* :func:`market_proxy` -- compute the equal-weight cross-sectional mean
  used as the benchmark for ``mes_95`` and the per-trade market return.

Validation pipeline
-------------------
``clean_price_data`` runs four checks in order; the first failure raises:

1. **Asset-count floor**: at least ``cfg.min_assets`` columns must be
   present *before* filtering, otherwise there is nothing to clean.
2. **Missing-data filter**: columns with more than ``cfg.max_missing_days``
   NaN values are dropped. The threshold is inclusive.
3. **Interpolation**: time-indexed linear interpolation with
   forward/backward fill handles any remaining gaps.
4. **Positivity and asset-count floor again**: all remaining values must
   be strictly positive (log-returns are undefined otherwise) and the
   filtered frame must still satisfy ``min_assets``.

A second ``min_assets`` check guards against pathological inputs that
collapse to a single column after filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataValidationConfig:
    """Tuning knobs for :func:`clean_price_data`.

    Attributes:
        max_missing_days: Maximum number of NaN values a column may have
            and still be kept. Defaults to ``10``.
        min_assets: Minimum number of asset columns the input must have
            before *and* after the filtering pipeline. Defaults to ``4``.
    """

    max_missing_days: int = 10
    min_assets: int = 4


def load_price_data(csv_path: str, date_col: str = "date") -> pd.DataFrame:
    """Parse a CSV file into a date-indexed price frame.

    The CSV must have a column whose name matches ``date_col`` containing
    ISO-8601 (or otherwise ``pd.to_datetime``-parseable) date strings. All
    remaining columns are coerced to ``float``. The result is sorted by
    date with the date column promoted to the index.

    Args:
        csv_path: Path to the CSV file.
        date_col: Name of the date column. Defaults to ``"date"``.

    Returns:
        ``pd.DataFrame`` indexed by ``pd.Timestamp`` with one column per
        asset, sorted ascending by date.

    Raises:
        ValueError: If the file is empty, if ``date_col`` is missing, if
            any column is duplicated, or if any non-date column contains
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


def clean_price_data(prices: pd.DataFrame, cfg: DataValidationConfig) -> pd.DataFrame:
    """Apply the four-stage validation pipeline to a raw price frame.

    See the module docstring for a description of each stage.

    Args:
        prices: Raw price ``pd.DataFrame`` (typically the output of
            :func:`load_price_data`).
        cfg: Validation configuration.

    Returns:
        A cleaned ``pd.DataFrame`` ready to be passed to
        :func:`log_returns`.

    Raises:
        ValueError: If any stage of the validation pipeline fails. See
            the module docstring for the exhaustive list.
    """
    if prices.shape[1] < cfg.min_assets:
        raise ValueError("Insufficient number of assets before filtering")
    missing_counts = prices.isna().sum(axis=0)
    # ``<`` rather than ``<=`` so the threshold is interpreted as an
    # *inclusive* upper bound on tolerated NaNs per column.
    keep_cols = missing_counts[missing_counts <= cfg.max_missing_days].index
    filtered = prices[keep_cols].copy()
    if filtered.empty:
        raise ValueError("No assets remaining after missing-value filtering")
    # ``method="time"`` requires a sorted datetime index so we pass the
    # original index through; ``ffill``/``bfill`` close any remaining
    # leading/trailing gaps.
    filtered = filtered.interpolate(method="time").ffill().bfill()
    if filtered.isna().any().any():
        raise ValueError("NaN values remain after interpolation")
    if (filtered <= 0).any().any():
        raise ValueError("Prices must be strictly positive")
    if filtered.shape[1] < cfg.min_assets:
        raise ValueError("Insufficient number of assets after filtering")
    return filtered


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns from a cleaned price frame.

    Applies the standard transformation ``r_t = log(P_t / P_{t-1})`` and
    drops the first row (which would otherwise be ``NaN``), followed by a
    pass that strips any residual ``NaN`` or ``inf`` rows that may have
    been introduced by zero-division guards upstream.

    Args:
        prices: Cleaned price frame.

    Returns:
        A ``pd.DataFrame`` of log-returns indexed by date, with no NaN or
        inf values.

    Raises:
        ValueError: If no log-returns could be computed (the input is
            empty or has fewer than two rows).
    """
    lr = np.log(prices / prices.shift(1)).dropna(how="all")
    if lr.empty:
        raise ValueError("No log-returns could be computed")
    # ``replace([inf, -inf], nan).dropna`` is the safety net for the rare
    # case where a downstream caller feeds a frame with zero prices that
    # somehow survived ``clean_price_data``.
    return lr.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def market_proxy(returns: pd.DataFrame) -> pd.Series:
    """Compute the equal-weight cross-sectional benchmark return series.

    The benchmark is the simple mean across asset columns at every
    timestamp -- a deliberately naïve choice that matches the convention
    used by most crypto index providers (and avoids any look-ahead bias
    that would come from a market-cap-weighted scheme).

    Args:
        returns: Log-return ``pd.DataFrame``.

    Returns:
        ``pd.Series`` of equal-weight mean returns indexed by the same
        timestamps as ``returns``.
    """
    return returns.mean(axis=1)
