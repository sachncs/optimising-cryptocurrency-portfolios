"""Shared data structures exchanged between pipeline modules.

This module defines the four dataclasses that flow through every layer of the
system:

* :class:`StrategySpec` -- a declarative flag pair used by the pipeline to
  decide whether a rebalance should incorporate the model's forward-looking
  prediction and/or a rolling time shift before the consensus Louvain pass.
* :class:`PortfolioResult` -- a single per-rebalance trade record written to
  ``trades.csv`` and surfaced through the REST API.
* :class:`EvaluationSummary` -- a per (strategy, horizon) aggregate of the
  metrics produced by :mod:`cps.metrics`; this is the row schema of
  ``summary.csv``.
* :class:`RunArtifacts` -- the top-level return container for
  :func:`cps.pipeline.run_pipeline`. It bundles the cleaned returns, the
  per-strategy/per-horizon summary, every trade, and the consensus
  similarity matrices computed during the run.

The structures are deliberately simple (``@dataclass`` instances of pandas
and NumPy primitives) so that they can be serialised to JSON / parquet /
``.npy`` without bespoke encoders. They are intentionally *not* frozen --
fields such as ``weights`` may be populated by helper functions before the
record is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    """A single strategy variant exercised at every rebalance.

    The four strategies produced by :func:`cps.pipeline.build_strategy_specs`
    arise from the two boolean flags:

    * ``use_prediction``: append the forecaster's forward-looking window to
      the rolling correlation matrix so clustering sees the predicted regime.
    * ``use_shifts``: stride the rolling-window end by the run index so each
      consensus Louvain pass conditions on a slightly different lookback.

    Attributes:
        name: Human-readable identifier (``"baseline"``, ``"s"``, ``"p"``,
            ``"p-s"``).
        use_prediction: When ``True``, the consensus similarity matrix
            reflects the predicted future window in addition to the realised
            one.
        use_shifts: When ``True``, the rolling-window end is shifted by the
            consensus-run index to introduce stochastic variation across
            passes.
    """

    name: str
    use_prediction: bool
    use_shifts: bool


@dataclass
class PortfolioResult:
    """A single per-rebalance trade record.

    Written as one row of ``trades.csv`` by the CLI and as one entry in the
    ``trades`` list returned by the REST API.

    Attributes:
        strategy: Name of the strategy variant that produced the trade
            (matches a :class:`StrategySpec.name`).
        horizon_days: Holding period in calendar days.
        rebalance_date: Calendar date of the rebalance (start of the holding
            window).
        exit_date: Calendar date at which the holding window ends and net
            return is realised.
        selected_assets: List of asset tickers selected by consensus
            clustering, ordered by cluster draw.
        weights: Mapping ``asset -> weight`` where weights lie on the
            long-only unit simplex (sum to 1, all ``>= 0``) after
            :func:`cps.risk.apply_weight_cap`.
        turnover: ``sum(|weights|)`` before clipping; equals 1.0 when the
            portfolio is fully invested and no position is shorted.
        gross_return: Compounded simple return of the portfolio over the
            holding window *before* execution costs.
        net_return: ``gross_return`` adjusted for transaction costs and
            slippage via :func:`cps.execution.apply_execution_costs`.
    """

    strategy: str
    horizon_days: int
    rebalance_date: pd.Timestamp
    exit_date: pd.Timestamp
    selected_assets: list[str]
    weights: dict[str, float]
    turnover: float
    gross_return: float
    net_return: float


@dataclass
class EvaluationSummary:
    """Per (strategy, horizon) aggregate of the per-trade metrics.

    One instance is produced by :func:`cps.metrics.summarize_strategy` for
    every strategy/horizon combination and serialised to ``summary.csv``.

    Attributes:
        strategy: Strategy name; matches a :class:`StrategySpec.name`.
        horizon_days: Holding period in calendar days.
        average_trade: Mean of realised trade returns (simple return, not
            log). ``0.0`` when there are no trades.
        win_rate: Fraction of trades with positive net return. ``0.0`` on
            an empty input.
        profit_factor: ``sum(gains) / sum(|losses|)``. ``+inf`` when there
            are no losing trades and at least one gain; ``0.0`` when there
            are no trades.
        var_95: 5th percentile of trade returns (Value-at-Risk at the 95%
            confidence level). Lower (more negative) is worse.
        mes_95: Mean trade return in the worst market-realised tail
            (Marginal Expected Shortfall at the 95% market quantile).
        omega_0: Omega ratio with threshold ``0`` -- gains above zero
            divided by losses below zero. ``+inf`` when there are no
            losing trades.
        trade_count: Number of trades that contributed to the summary.
    """

    strategy: str
    horizon_days: int
    average_trade: float
    win_rate: float
    profit_factor: float
    var_95: float
    mes_95: float
    omega_0: float
    trade_count: int


@dataclass
class RunArtifacts:
    """Top-level return container of :func:`cps.pipeline.run_pipeline`.

    Bundles every output of a single pipeline run so it can be written to
    disk by the CLI or surfaced through the REST API.

    Attributes:
        returns: Cleaned log-returns time series, indexed by date, with one
            column per surviving asset.
        market_returns: Equal-weight cross-sectional mean of ``returns``,
            used as the benchmark for ``mes_95`` and for the compounded
            market trade return in the pipeline.
        trades: List of per-rebalance :class:`PortfolioResult` records.
        summary: List of per (strategy, horizon) :class:`EvaluationSummary`
            records.
        similarity_matrices: Mapping ``scenario_key -> np.ndarray`` where
            ``scenario_key`` encodes the strategy, horizon, and rebalance
            index. Each matrix is the consensus co-occurrence similarity
            produced by :func:`cps.networking.consensus_similarity_matrix`.
    """

    returns: pd.DataFrame
    market_returns: pd.Series
    trades: list[PortfolioResult]
    summary: list[EvaluationSummary]
    similarity_matrices: dict[str, np.ndarray]
