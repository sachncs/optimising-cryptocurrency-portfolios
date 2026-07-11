"""Per-trade performance and downside-risk metrics.

This module is intentionally stateless and dependency-free apart from NumPy
and pandas. Every function accepts a 1-D ``np.ndarray`` of trade returns
and returns a scalar; the higher-level :func:`summarize_strategy` and
:func:`summaries_to_frame` glue these primitives together to produce the
per-(strategy, horizon) summary rows written to ``summary.csv``.

Conventions:

* A "trade return" is a *simple* return (not log return) over a single
  holding period -- the same convention used by
  :func:`cps.portfolio.compute_portfolio_simple_return`.
* Empty inputs return ``0.0`` for ratio-style metrics (with the exception
  of :func:`profit_factor` and :func:`omega_ratio`, which return ``+inf``
  when there are gains but no losses, and ``0.0`` otherwise).

Design notes
------------
The ``var_95`` and ``mes_95`` metrics form a simple downside-risk pair:
``var_95`` is the 5th-percentile cutoff of the trade distribution and
``mes_95`` is the conditional mean of trades whose *benchmark* return
falls below the 5th-percentile of the benchmark distribution. This is
the standard "marginal expected shortfall" formulation -- it conditions on
the market regime rather than on the strategy's own drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import EvaluationSummary


def average_trade(trades: np.ndarray) -> float:
    """Compute the arithmetic mean of trade returns.

    Args:
        trades: 1-D array of per-trade simple returns.

    Returns:
        Mean return, or ``0.0`` when ``trades`` is empty.
    """
    return float(np.mean(trades)) if trades.size else 0.0


def win_rate(trades: np.ndarray) -> float:
    """Compute the fraction of trades with strictly positive return.

    Args:
        trades: 1-D array of per-trade simple returns.

    Returns:
        Win rate in ``[0.0, 1.0]``, or ``0.0`` when ``trades`` is empty.
    """
    return float(np.mean(trades > 0)) if trades.size else 0.0


def profit_factor(trades: np.ndarray) -> float:
    """Compute the profit factor (``sum gains / sum |losses|``).

    Args:
        trades: 1-D array of per-trade simple returns.

    Returns:
        * ``+inf`` when there is at least one gain and zero losses.
        * ``0.0`` when there are no trades, or when there are losses but no
          gains (the standard convention for an unprofitable sample).
    """
    gains = float(np.sum(trades[trades > 0]))
    losses = float(np.sum(np.abs(trades[trades < 0])))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def var_quantile(trades: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the Value-at-Risk at confidence level ``alpha``.

    Args:
        trades: 1-D array of per-trade simple returns.
        alpha: Confidence level in ``(0, 1)``. Defaults to ``0.95``.

    Returns:
        The ``1 - alpha`` quantile of ``trades`` (a *low* number -- more
        negative is worse). ``0.0`` when ``trades`` is empty.
    """
    if trades.size == 0:
        return 0.0
    return float(np.quantile(trades, 1 - alpha))


def mes(trades: np.ndarray, market: np.ndarray, alpha: float = 0.95) -> float:
    """Compute the Marginal Expected Shortfall at confidence level ``alpha``.

    ``mes`` conditions on the *benchmark* return rather than on the strategy
    return: it averages the trade returns for those rebalances whose market
    return falls below the ``1 - alpha`` quantile of the market
    distribution. This captures "how does the strategy perform when the
    market is in its left tail?".

    Args:
        trades: 1-D array of per-trade simple returns aligned with
            ``market``.
        market: 1-D array of benchmark (equal-weight market proxy) returns
            aligned with ``trades``.
        alpha: Confidence level in ``(0, 1)``. Defaults to ``0.95``.

    Returns:
        Mean trade return conditional on the market being in its left tail,
        or ``0.0`` when either input is empty or the tail slice is empty.
    """
    if trades.size == 0 or market.size == 0:
        return 0.0
    threshold = np.quantile(market, 1 - alpha)
    tail = trades[market <= threshold]
    return float(np.mean(tail)) if tail.size else 0.0


def omega_ratio(trades: np.ndarray, threshold: float = 0.0) -> float:
    """Compute the Omega ratio with a user-supplied threshold.

    The Omega ratio is the probability-weighted gains above ``threshold``
    divided by the probability-weighted losses below ``threshold``. With
    the default threshold of ``0`` it reduces to ``sum(gains) /
    sum(|losses|)`` -- functionally similar to :func:`profit_factor` but
    admitting any threshold (e.g. the risk-free rate).

    Args:
        trades: 1-D array of per-trade simple returns.
        threshold: Reference return used to split gains from losses.
            Defaults to ``0.0``.

    Returns:
        * ``+inf`` when there is at least one gain above ``threshold`` and
          zero losses below it.
        * ``0.0`` when there are no trades or there are losses but no gains
          above ``threshold``.
    """
    if trades.size == 0:
        return 0.0
    gains = np.sum(np.maximum(trades - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - trades, 0.0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize_strategy(
    strategy: str, horizon: int, trade_returns: list[float], market_returns: list[float]
) -> EvaluationSummary:
    """Aggregate the per-trade metrics into a single summary row.

    Args:
        strategy: Strategy name (``"baseline"``, ``"s"``, ``"p"``, ``"p-s"``).
        horizon: Holding period in days.
        trade_returns: Per-trade simple returns for the strategy at the
            given horizon.
        market_returns: Benchmark returns aligned with ``trade_returns``.

    Returns:
        A populated :class:`cps.types.EvaluationSummary`.
    """
    # Materialise to numpy once -- the metric functions are all pure NumPy
    # and converting on each call would be wasteful on large samples.
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
    """Convert a list of summaries into a tabular ``pd.DataFrame``.

    The conversion relies on :py:meth:`dataclasses.asdict` semantics --
    ``__dict__`` is used directly to preserve field order. Any consumer
    that adds a new dataclass field to :class:`EvaluationSummary` will see
    the column appear automatically in the resulting frame.

    Args:
        summaries: List of :class:`EvaluationSummary` instances.

    Returns:
        A :class:`pd.DataFrame` with one row per summary.
    """
    return pd.DataFrame([s.__dict__ for s in summaries])
