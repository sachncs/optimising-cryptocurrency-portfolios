"""Shared value objects exchanged between pipeline modules.

The dataclasses here are frozen and use only immutable container types
(``tuple``, ``MappingProxyType``). Mutating any instance after
construction raises ``AttributeError``.

The three dataclasses:

* :class:`PortfolioResult` -- a single per-rebalance trade record
  written to ``trades.json`` and surfaced through the REST API.
* :class:`EvaluationSummary` -- a per (strategy, horizon) aggregate
  produced by :func:`cps.application.portfolio_service` and serialised
  to ``summary.json``.
* :class:`RunArtifacts` -- the top-level return container of the
  application services. Bundles every output of one pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioResult:
    """A single per-rebalance trade record.

    Attributes:
        strategy: Strategy variant that produced the trade.
        horizon_days: Holding period in calendar days.
        rebalance_date: Start of the holding window.
        exit_date: End of the holding window.
        selected_assets: Tuple of asset tickers selected by consensus
            clustering, ordered by cluster draw.
        weights: Read-only mapping ``asset -> weight``.
        turnover: ``sum(|weights|)`` before clipping.
        gross_return: Compounded simple return before costs.
        net_return: ``gross_return`` adjusted for transaction costs and
            slippage.
    """

    strategy: str
    horizon_days: int
    rebalance_date: pd.Timestamp
    exit_date: pd.Timestamp
    selected_assets: tuple[str, ...]
    weights: Mapping[str, float]
    turnover: float
    gross_return: float
    net_return: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Per (strategy, horizon) aggregate of the per-trade metrics."""

    strategy: str
    horizon_days: int
    average_trade: float
    win_rate: float
    profit_factor: float
    var_95: float
    mes_95: float
    omega_0: float
    trade_count: int


@dataclass(frozen=True)
class RunArtifacts:
    """Top-level return container of the application services.

    Attributes:
        returns: Cleaned log-returns time series, indexed by date.
        market_returns: Equal-weight cross-sectional mean of ``returns``.
        trades: Tuple of per-rebalance :class:`PortfolioResult`.
        summary: Tuple of per (strategy, horizon)
            :class:`EvaluationSummary`.
        similarity_matrices: Read-only mapping ``scenario_key ->
            np.ndarray``.
    """

    returns: pd.DataFrame
    market_returns: pd.Series
    trades: tuple[PortfolioResult, ...]
    summary: tuple[EvaluationSummary, ...]
    similarity_matrices: Mapping[str, np.ndarray]


def freeze_trades(trades: list[PortfolioResult]) -> tuple[PortfolioResult, ...]:
    """Convert a list of trades into an immutable tuple."""
    return tuple(trades)


def freeze_summary(summary: list[EvaluationSummary]) -> tuple[EvaluationSummary, ...]:
    """Convert a list of summaries into an immutable tuple."""
    return tuple(summary)


def freeze_similarity_matrices(
    matrices: dict[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    """Convert a dict of similarity matrices into a read-only mapping view."""
    return MappingProxyType(dict[str, np.ndarray](matrices))


__all__ = [
    "EvaluationSummary",
    "PortfolioResult",
    "RunArtifacts",
    "freeze_similarity_matrices",
    "freeze_summary",
    "freeze_trades",
]