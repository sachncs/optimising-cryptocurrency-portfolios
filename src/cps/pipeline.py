"""Pipeline orchestration.

This module is the single integration point that wires together every
other module in the package: data ingestion, return forecasting,
correlation-graph construction, consensus Louvain clustering, mean-variance
portfolio optimisation with Ledoit-Wolf covariance shrinkage, risk
validation, execution-cost adjustment, governance drift detection, and
metric summarisation.

Lifecycle
---------
For each ``horizon_days`` in ``config.horizons_days`` and for each
``strategy`` in :func:`build_strategy_specs`::

    for rebalance_index in eligible_indices:
        train  = returns[rebalance_index - train_window : rebalance_index]
        future = returns[rebalance_index : rebalance_index + horizon]

        1. forecast_matrix(train, horizon, ...)   if strategy.use_prediction
        2. consensus similarity over `consensus_runs` Louvain partitions
        3. threshold + connected components -> stable_clusters
        4. draw one asset per cluster -> selected_assets
        5. mean / Ledoit-Wolf covariance over selected_train_returns
        6. Sharpe-ratio ascent with risk-free rate -> weights
        7. apply_weight_cap + validate_trade_risk
        8. compute_portfolio_simple_return(future, weights)
        9. apply_execution_costs -> net_return
       10. record MSE into ForecastGovernance

Reference
---------
arXiv:2505.24831v2, "Consensus-Clustered Cryptocurrency Portfolio
Construction" -- the framework this module implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DataValidationConfig, clean_price_data, log_returns, market_proxy
from .execution import ExecutionCostConfig, apply_execution_costs, compute_total_cost_rate
from .forecast import GARCHDistribution, GARCHForecastConfig, GARCHMeanModel, forecast_matrix
from .governance import ForecastGovernance
from .lstm_model import LSTMTrainingConfig
from .metrics import summarize_strategy
from .networking import (
    build_weighted_graph_from_distance,
    consensus_similarity_matrix,
    correlation_distance_matrix,
    louvain_partition,
    stable_clusters_from_similarity,
)
from .observability import MetricsRegistry, StructuredLogger, Timer
from .portfolio import (
    compute_ledoit_wolf_constant_variance_covariance,
    compute_portfolio_simple_return,
    optimize_maximum_sharpe_ratio,
)
from .risk import RiskLimits, apply_weight_cap, validate_trade_risk
from .types import PortfolioResult, RunArtifacts, StrategySpec


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for portfolio construction and evaluation pipeline.

    Attributes:
        train_window_days: Number of days of returns used to fit the
            forecaster and estimate the covariance matrix. Defaults to
            ``180``.
        correlation_window_days: Width of the rolling window used to
            build each consensus similarity matrix. Defaults to ``60``.
        rebalance_step_days: Number of days between successive rebalances.
            Defaults to ``30``.
        horizons_days: Holding periods (in days) to evaluate. Defaults
            to ``[1, 3, 7, 14]``.
        consensus_runs: Number of independent Louvain partitions used to
            build the consensus similarity matrix. Defaults to ``20``.
        majority_threshold: Co-membership probability cutoff for
            declaring two assets stable neighbours. Defaults to ``0.5``.
        risk_free_rate_annual: Annualised risk-free rate used in the
            Sharpe-ratio objective. Defaults to ``0.045`` (4.5%).
        forecast_method: One of ``"naive"``, ``"arima"``, ``"garch"``,
            ``"lstm"``. Defaults to ``"arima"``.
        random_seed: Seed for the NumPy RNG used by the Louvain passes.
            Defaults to ``42``.
        weight_cap: Configured per-asset cap (see
            :func:`cps.risk.apply_weight_cap`). Defaults to ``0.35``.
        max_assets: Upper bound on selected assets. Defaults to ``25``.
        min_assets: Lower bound on selected assets. Defaults to ``2``.
        max_volatility_annual: Annualised volatility ceiling for the
            realised portfolio. Defaults to ``1.2`` (120%).
        transaction_cost_bps: One-way commission in basis points. Defaults
            to ``10``.
        slippage_bps: Expected price impact in basis points. Defaults to
            ``5``.
        lstm_lookback: Lookback window for the LSTM forecaster. Defaults
            to ``10``.
        lstm_hidden_size: LSTM hidden-state width. Defaults to ``16``.
        lstm_num_layers: Number of stacked LSTM layers. Defaults to ``1``.
        lstm_max_epochs: Maximum LSTM training epochs. Defaults to ``80``.
        garch_p: GARCH lag order. Defaults to ``1``.
        garch_o: GARCH asymmetry order. Defaults to ``1``.
        garch_q: ARCH lag order. Defaults to ``1``.
        garch_mean: GARCH mean model. Defaults to ``"Zero"``.
        garch_dist: GARCH innovation distribution. Defaults to ``"t"``.
        garch_auto_order: Whether to fit a small AIC candidate grid.
            Defaults to ``True``.
    """

    train_window_days: int = 180
    correlation_window_days: int = 60
    rebalance_step_days: int = 30
    horizons_days: list[int] = field(default_factory=lambda: [1, 3, 7, 14])
    consensus_runs: int = 20
    majority_threshold: float = 0.5
    risk_free_rate_annual: float = 0.045
    forecast_method: str = "arima"
    random_seed: int = 42
    weight_cap: float = 0.35
    max_assets: int = 25
    min_assets: int = 2
    max_volatility_annual: float = 1.2
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    lstm_lookback: int = 10
    lstm_hidden_size: int = 16
    lstm_num_layers: int = 1
    lstm_max_epochs: int = 80
    garch_p: int = 1
    garch_o: int = 1
    garch_q: int = 1
    garch_mean: GARCHMeanModel = "Zero"
    garch_dist: GARCHDistribution = "t"
    garch_auto_order: bool = True


def build_strategy_specs() -> list[StrategySpec]:
    """Return the four fixed strategy variants exercised at every rebalance.

    The two boolean flags in :class:`StrategySpec` produce the four
    combinations used to ablate the consensus Louvain pipeline:

    * ``baseline``: no forecast, no shift -- pure rolling correlation.
    * ``s``: shift only -- introduces stochastic variation across runs.
    * ``p``: prediction only -- clusters the predicted future window.
    * ``p-s``: both prediction and shift.
    """
    return [
        StrategySpec("baseline", use_prediction=False, use_shifts=False),
        StrategySpec("s", use_prediction=False, use_shifts=True),
        StrategySpec("p", use_prediction=True, use_shifts=False),
        StrategySpec("p-s", use_prediction=True, use_shifts=True),
    ]


def compute_daily_risk_free_rate(annual_rate: float) -> float:
    """Convert an annual risk-free rate to its daily compounded equivalent.

    Uses ``((1 + r) ** (1 / 365)) - 1`` rather than naive division so the
    daily rate compounds exactly to the supplied annual rate over a 365-day
    window.

    Args:
        annual_rate: Annualised risk-free rate as a decimal (e.g. ``0.045``
            for 4.5%).

    Returns:
        Daily rate as a decimal.
    """
    return float((1.0 + annual_rate) ** (1.0 / 365.0) - 1.0)


def build_consensus_partitions(
    train_segment: pd.DataFrame,
    strategy: StrategySpec,
    config: PipelineConfig,
    random_generator: np.random.Generator,
) -> tuple[list[list[set[str]]], np.ndarray]:
    """Compute the consensus similarity matrix for one strategy on one rebalance.

    Args:
        train_segment: Returns over the rolling training window. One
            column per asset.
        strategy: The :class:`StrategySpec` variant being evaluated.
        config: The pipeline configuration.
        random_generator: NumPy RNG used to seed each Louvain pass.

    Returns:
        A 2-tuple ``(partitions, similarity)`` where ``partitions`` is
        the list of Louvain partitions used to build the consensus (the
        caller may discard it) and ``similarity`` is the consensus
        co-occurrence matrix.

    Algorithm:
        Pseudocode::

            if strategy.use_prediction:
                prediction = forecast_matrix(segment, horizon, config.forecast_method)
            else:
                prediction = None

            partitions = []
            for run_index in range(config.consensus_runs):
                if strategy.use_shifts:
                    shift = run_index
                else:
                    shift = 0
                end = len(segment) - shift
                start = end - config.correlation_window_days
                if start < 0:
                    continue
                window = segment.iloc[start:end]
                if prediction is not None:
                    window = concat(window, prediction)
                distance = correlation_distance_matrix(window)
                graph = build_weighted_graph_from_distance(distance)
                partitions.append(louvain_partition(graph, seed=seed))
            similarity = consensus_similarity_matrix(partitions, segment.columns)
            return partitions, similarity
    """
    assets = list(train_segment.columns)
    partitions: list[list[set[str]]] = []
    forecast_steps = config.correlation_window_days if strategy.use_prediction else 0
    # Build typed forecast configs once so we can reuse them across the
    # consensus runs without re-instantiating dataclasses.
    garch_config = GARCHForecastConfig(
        p=config.garch_p,
        o=config.garch_o,
        q=config.garch_q,
        mean=config.garch_mean,
        dist=config.garch_dist,
        auto_order=config.garch_auto_order,
    )
    lstm_config = LSTMTrainingConfig(
        lookback=config.lstm_lookback,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        max_epochs=config.lstm_max_epochs,
        seed=config.random_seed,
    )
    prediction = (
        forecast_matrix(
            train_segment,
            forecast_steps,
            config.forecast_method,
            garch_config=garch_config,
            lstm_config=lstm_config,
        )
        if forecast_steps > 0
        else None
    )

    for run_index in range(config.consensus_runs):
        # ``use_shifts`` introduces a one-day stride per run so each
        # consensus pass conditions on a slightly different lookback --
        # the source of stochastic variation across Louvain partitions.
        shift = run_index if strategy.use_shifts else 0
        end_index = len(train_segment) - shift
        start_index = end_index - config.correlation_window_days
        if start_index < 0:
            continue
        window = train_segment.iloc[start_index:end_index].copy()
        if strategy.use_prediction and prediction is not None:
            window = pd.concat([window, prediction], axis=0, ignore_index=True)
        distance = correlation_distance_matrix(window)
        graph = build_weighted_graph_from_distance(distance)
        partition = louvain_partition(graph, seed=int(random_generator.integers(0, 1_000_000)))
        partitions.append(partition)

    similarity = consensus_similarity_matrix(partitions, assets)
    return partitions, similarity


def run_pipeline(
    prices: pd.DataFrame,
    config: PipelineConfig,
    logger: StructuredLogger | None = None,
    metrics_registry: MetricsRegistry | None = None,
) -> RunArtifacts:
    """Run the full consensus-clustered portfolio pipeline.

    Iterates over every horizon, every rebalance index, and every strategy
    variant, executing the lifecycle documented in the module docstring.

    Args:
        prices: Raw price ``pd.DataFrame`` indexed by date, one column
            per asset.
        config: Pipeline configuration.
        logger: Optional :class:`StructuredLogger` for emitting pipeline
            events. When ``None``, events are silently dropped.
        metrics_registry: Optional :class:`MetricsRegistry` for counters
            and timings. When ``None``, no metrics are recorded.

    Returns:
        A fully populated :class:`RunArtifacts` containing the cleaned
        returns, market proxy, every trade, every summary, and every
        consensus similarity matrix keyed by scenario.

    Raises:
        ValueError: Propagated from :func:`cps.data.clean_price_data`
            and :func:`cps.risk.validate_trade_risk` when input data or
            intermediate portfolios violate their respective invariants.

    Examples:
        >>> from cps import PipelineConfig, run_pipeline
        >>> import pandas as pd
        >>> prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
        >>> artifacts = run_pipeline(prices, PipelineConfig(forecast_method="naive"))
    """
    pipeline_timer = Timer()
    cleaned_prices = clean_price_data(prices, DataValidationConfig(min_assets=config.min_assets))
    returns = log_returns(cleaned_prices)
    market_returns = market_proxy(returns)
    random_generator = np.random.default_rng(config.random_seed)
    governance = ForecastGovernance()

    risk_limits = RiskLimits(
        max_assets=config.max_assets,
        min_assets=config.min_assets,
        max_weight_per_asset=config.weight_cap,
        max_volatility_annual=config.max_volatility_annual,
    )
    cost_config = ExecutionCostConfig(
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
    )

    all_trades: list[PortfolioResult] = []
    all_summaries = []
    similarity_matrices: dict[str, np.ndarray] = {}
    daily_risk_free_rate = compute_daily_risk_free_rate(config.risk_free_rate_annual)
    strategy_specs = build_strategy_specs()

    if logger is not None:
        logger.log_event("pipeline_started", {"rows": len(returns), "assets": returns.shape[1]})

    for horizon_days in config.horizons_days:
        returns_by_strategy: dict[str, list[float]] = {spec.name: [] for spec in strategy_specs}
        market_by_strategy: dict[str, list[float]] = {spec.name: [] for spec in strategy_specs}

        rebalance_index = config.train_window_days
        while rebalance_index + horizon_days <= len(returns):
            train_returns = returns.iloc[rebalance_index - config.train_window_days : rebalance_index]
            future_returns = returns.iloc[rebalance_index : rebalance_index + horizon_days]

            for strategy in strategy_specs:
                partitions, similarity = build_consensus_partitions(train_returns, strategy, config, random_generator)
                # ``partitions`` is no longer needed after the consensus
                # matrix is built -- release the reference eagerly to
                # keep memory bounded across rebalances.
                del partitions
                similarity_key = f"{strategy.name}_h{horizon_days}_t{rebalance_index}"
                similarity_matrices[similarity_key] = similarity

                stable_clusters = stable_clusters_from_similarity(
                    similarity, list(train_returns.columns), config.majority_threshold
                )
                selected_assets = [
                    cluster[int(random_generator.integers(0, len(cluster)))] for cluster in stable_clusters if cluster
                ]
                if not selected_assets:
                    continue

                if len(selected_assets) > config.max_assets:
                    selected_assets = selected_assets[: config.max_assets]
                if len(selected_assets) < config.min_assets:
                    continue

                selected_train_returns = train_returns[selected_assets]
                selected_future_returns = future_returns[selected_assets]
                expected_returns = selected_train_returns.mean(axis=0)
                covariance = compute_ledoit_wolf_constant_variance_covariance(selected_train_returns)
                weights = optimize_maximum_sharpe_ratio(expected_returns, covariance, daily_risk_free_rate)
                weights = apply_weight_cap(weights, config.weight_cap)
                validate_trade_risk(selected_assets, weights, covariance, risk_limits)

                # In-sample MSE between the realised returns and the
                # mean forecast. Used by ``ForecastGovernance`` to flag
                # drift when the model suddenly stops fitting.
                mse_value = float(((selected_train_returns - expected_returns) ** 2).mean().mean())
                governance.record_error(mse_value)

                gross_trade_return = compute_portfolio_simple_return(selected_future_returns, weights)
                turnover = float(np.abs(weights).sum())
                cost_rate = compute_total_cost_rate(cost_config, turnover)
                net_trade_return = apply_execution_costs(gross_trade_return, cost_rate)

                # The market "trade return" is the compounded simple
                # return of the equal-weight benchmark over the holding
                # window -- the same series the pipeline uses for MES.
                market_trade_return = float(((1.0 + future_returns.mean(axis=1)).prod()) - 1.0)
                returns_by_strategy[strategy.name].append(net_trade_return)
                market_by_strategy[strategy.name].append(market_trade_return)

                all_trades.append(
                    PortfolioResult(
                        strategy=strategy.name,
                        horizon_days=horizon_days,
                        rebalance_date=returns.index[rebalance_index],
                        exit_date=returns.index[rebalance_index + horizon_days - 1],
                        selected_assets=selected_assets,
                        weights=weights.to_dict(),
                        turnover=turnover,
                        gross_return=gross_trade_return,
                        net_return=net_trade_return,
                    )
                )
                if metrics_registry is not None:
                    metrics_registry.increment("trades_executed")
            rebalance_index += config.rebalance_step_days

        for strategy in strategy_specs:
            all_summaries.append(
                summarize_strategy(
                    strategy=strategy.name,
                    horizon=horizon_days,
                    trade_returns=returns_by_strategy[strategy.name],
                    market_returns=market_by_strategy[strategy.name],
                )
            )

    if governance.is_drift_detected() and logger is not None:
        logger.log_event("forecast_drift_detected", {"history_points": len(governance.mse_history)})

    if metrics_registry is not None:
        metrics_registry.record_timing_millis("pipeline_duration_millis", pipeline_timer.elapsed_millis())

    if logger is not None:
        logger.log_event("pipeline_completed", {"trades": len(all_trades), "summaries": len(all_summaries)})

    return RunArtifacts(
        returns=returns,
        market_returns=market_returns,
        trades=all_trades,
        summary=all_summaries,
        similarity_matrices=similarity_matrices,
    )
