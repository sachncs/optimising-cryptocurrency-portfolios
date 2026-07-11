"""Per-trade performance and downside-risk metrics.

Stateless and dependency-free apart from NumPy and pandas.

Conventions:

* "trade return" = *simple* return over a single holding period.
* empty inputs return ``0.0`` for ratio-style metrics (profit_factor
  and omega_ratio return ``+inf`` when there are gains but no losses).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .types import EvaluationSummary


def average_trade(trades: np.ndarray) -> float:
    """Compute the arithmetic mean of trade returns."""
    return float(np.mean(trades)) if trades.size else 0.0


def win_rate(trades: np.ndarray) -> float:
    """Compute the fraction of trades with strictly positive return."""
    return float(np.mean(trades > 0)) if trades.size else 0.0


def profit_factor(trades: np.ndarray) -> float:
    """Compute ``sum gains / sum |losses|``."""
    gains = float(np.sum(trades[trades > 0]))
    losses = float(np.sum(np.abs(trades[trades < 0])))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def var_quantile(trades: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the Value-at-Risk at confidence ``alpha``."""
    if trades.size == 0:
        return 0.0
    return float(np.quantile(trades, 1 - alpha))


def mes(trades: np.ndarray, market: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the Marginal Expected Shortfall conditioning on the benchmark tail."""
    if trades.size == 0 or market.size == 0:
        return 0.0
    threshold = np.quantile(market, 1 - alpha)
    tail = trades[market <= threshold]
    return float(np.mean(tail)) if tail.size else 0.0


def omega_ratio(trades: np.ndarray, threshold: float = 0.0) -> float:
    """Compute the Omega ratio with a user-supplied threshold."""
    if trades.size == 0:
        return 0.0
    gains = np.sum(np.maximum(trades - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - trades, 0.0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize_strategy(
    strategy: str,
    horizon: int,
    trade_returns: list[float],
    market_returns: list[float],
) -> EvaluationSummary:
    """Aggregate the per-trade metrics into a single summary row."""
    t = np.asarray(trade_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    return EvaluationSummary(
        strategy=strategy,
        horizon_days=horizon,
        average_trade=average_trade(t),
        win_rate=win_rate(t),
        profit_factor=profit_factor(t),
        var_95=var_quantile(t, 0.95),
        mes_95=mes(t, m, 0.95),
        omega_0=omega_ratio(t, 0.0),
        trade_count=int(t.size),
    )


def summaries_to_frame(summaries: list[EvaluationSummary]) -> pd.DataFrame:
    """Convert a list of summaries into a tabular ``pd.DataFrame``."""
    return pd.DataFrame([asdict(summary) for summary in summaries])


__all__ = [
    "average_trade",
    "mes",
    "omega_ratio",
    "profit_factor",
    "summaries_to_frame",
    "summarize_strategy",
    "var_quantile",
    "win_rate",
]