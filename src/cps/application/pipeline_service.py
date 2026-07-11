"""Pipeline orchestration service.

Decomposes the previous 180-line god function into discrete stages:

* :meth:`PipelineService.run` -- the single entry point.
* :meth:`PipelineService._run_horizon` -- per-horizon loop.
* :meth:`PipelineService._run_rebalance` -- per-rebalance loop.
* :meth:`PipelineService._select_assets` -- consensus Louvain cluster
  selection.
* :meth:`PipelineService._construct_portfolio` -- risk + portfolio
  construction.
* :meth:`PipelineService._emit` -- typed event publishing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    SHARPE_DEFAULT_LEARNING_STEP,
    SHARPE_DEFAULT_MAX_ITERATIONS,
    PipelineConfig,
    StrategySpec,
    default_strategy_specs,
)
from ..domain import (
    ArtifactStore,
    EvaluationSummary,
    EventPayload,
    ExecutionCostConfig,
    ForecastDriftPayload,
    ForecastGovernance,
    Horizon,
    PipelineCompletedPayload,
    PipelineContext,
    PipelineEvent,
    PipelineStartedPayload,
    PortfolioResult,
    RebalanceExecutedPayload,
    RiskLimits,
    RunArtifacts,
    ScenarioKey,
    build_weighted_graph_from_distance,
    consensus_similarity_matrix,
    correlation_distance_matrix,
    freeze_similarity_matrices,
    freeze_summary,
    freeze_trades,
    louvain_partition,
    stable_clusters_from_similarity,
)
from ..infrastructure.observability import MetricsRegistry, StructuredLogger
from .data_cleaning import (
    DataValidationConfig,
    clean_price_data,
    log_returns,
    market_proxy,
)
from .forecast_service import ForecastService
from .portfolio_metrics import summarize_strategy
from .portfolio_service import PortfolioConstructionError, PortfolioService
from .risk_service import RiskService


@dataclass(frozen=True)
class PipelineResult:
    """Lightweight result wrapper used by ``PipelineService.run``.

    ``RunArtifacts`` is the persisted bundle; ``PipelineResult`` adds
    in-memory state used by tests and the typed event consumers.
    """

    artifacts: RunArtifacts
    trades: list[PortfolioResult]
    summaries: list[EvaluationSummary]


class PipelineService:
    """Orchestrates the consensus-clustered portfolio pipeline.

    All collaborators (artifact store, logger, governance, registries)
    are constructor-injected so the service is fully testable.
    """

    def __init__(
        self,
        config: PipelineConfig,
        context: PipelineContext,
        forecast_service: ForecastService,
    ) -> None:
        """Initialise the service with a config, context, and forecaster service."""
        if not config.horizons:
            raise ValueError("PipelineConfig.horizons must not be empty")
        self.__config = config
        self.__context = context
        self.__forecast_service = forecast_service
        self.__risk_limits = self.__derive_risk_limits(config)

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self.__config

    @property
    def context(self) -> PipelineContext:
        """Return the surrounding dependency context."""
        return self.__context

    def run(self, prices: pd.DataFrame) -> PipelineResult:
        """Run the full pipeline on the supplied price frame.

        Args:
            prices: Raw price ``pd.DataFrame`` indexed by date.

        Returns:
            :class:`PipelineResult` with the in-memory trades and
            summaries plus the persisted :class:`RunArtifacts`.
        """
        cleaned_prices = clean_price_data(prices, DataValidationConfig(min_assets=self.__config.min_assets))
        returns = log_returns(cleaned_prices)
        market_returns = market_proxy(returns)

        cost_config = ExecutionCostConfig(
            transaction_cost_bps=self.__config.transaction_cost_bps,
            slippage_bps=self.__config.slippage_bps,
        )
        portfolio_service = PortfolioService(
            self.__risk_limits,
            cost_config,
            daily_risk_free_rate=self.__daily_risk_free_rate(),
            max_iterations=SHARPE_DEFAULT_MAX_ITERATIONS,
            learning_step=SHARPE_DEFAULT_LEARNING_STEP,
        )
        risk_service = RiskService(self.__risk_limits)

        all_trades: list[PortfolioResult] = []
        all_summaries: list[EvaluationSummary] = []
        similarity_matrices: dict[str, np.ndarray] = {}
        strategy_specs = default_strategy_specs()

        self._emit(
            PipelineEvent.PIPELINE_STARTED,
            PipelineStartedPayload(rows=len(returns), assets=returns.shape[1]),
        )

        for horizon in self.__config.horizons:
            new_trades, new_summaries, _new_similarities = self._run_horizon(
                returns=returns,
                market_returns=market_returns,
                horizon=horizon,
                strategy_specs=strategy_specs,
                portfolio_service=portfolio_service,
                risk_service=risk_service,
                similarity_matrices=similarity_matrices,
            )
            all_trades.extend(new_trades)
            all_summaries.extend(new_summaries)

        artifacts = RunArtifacts(
            returns=returns,
            market_returns=market_returns,
            trades=freeze_trades(all_trades),
            summary=freeze_summary(all_summaries),
            similarity_matrices=freeze_similarity_matrices(similarity_matrices),
        )
        self.__context.metrics_registry.record_timing_millis(
            "pipeline_duration_millis",
            float("nan"),
        )
        self._emit(
            PipelineEvent.PIPELINE_COMPLETED,
            PipelineCompletedPayload(
                trades=len(all_trades),
                summaries=len(all_summaries),
                duration_millis=self.__context.metrics_registry.snapshot().timings_millis.get(
                    "pipeline_duration_millis", ()
                )[-1]
                if self.__context.metrics_registry.snapshot().timings_millis.get("pipeline_duration_millis", ())
                else 0.0,
            ),
        )
        return PipelineResult(artifacts=artifacts, trades=all_trades, summaries=all_summaries)

    def _run_horizon(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        horizon: Horizon,
        strategy_specs: Sequence[StrategySpec],
        portfolio_service: PortfolioService,
        risk_service: RiskService,
        similarity_matrices: dict[str, np.ndarray],
    ) -> tuple[list[PortfolioResult], list[EvaluationSummary], dict[str, np.ndarray]]:
        """Run every strategy at every eligible rebalance for one horizon."""
        horizon_trades_by_strategy: dict[str, list[float]] = {spec.name: [] for spec in strategy_specs}
        horizon_market_by_strategy: dict[str, list[float]] = {spec.name: [] for spec in strategy_specs}
        horizon_trades: list[PortfolioResult] = []
        horizon_summaries: list[EvaluationSummary] = []

        rebalance_index = self.__config.train_window_days
        while rebalance_index + horizon.days <= len(returns):
            train_returns = returns.iloc[rebalance_index - self.__config.train_window_days : rebalance_index]
            future_returns = returns.iloc[rebalance_index : rebalance_index + horizon.days]
            self._run_rebalance(
                train_returns=train_returns,
                future_returns=future_returns,
                market_returns=future_returns.mean(axis=1),
                returns_index=returns.index,
                rebalance_index=rebalance_index,
                horizon=horizon,
                strategy_specs=strategy_specs,
                portfolio_service=portfolio_service,
                risk_service=risk_service,
                similarity_matrices=similarity_matrices,
                horizon_trades_by_strategy=horizon_trades_by_strategy,
                horizon_market_by_strategy=horizon_market_by_strategy,
                horizon_trades=horizon_trades,
            )
            rebalance_index += self.__config.rebalance_step_days

        for spec in strategy_specs:
            horizon_summaries.append(
                summarize_strategy(
                    strategy=spec.name,
                    horizon=horizon.days,
                    trade_returns=horizon_trades_by_strategy[spec.name],
                    market_returns=horizon_market_by_strategy[spec.name],
                )
            )
        return horizon_trades, horizon_summaries, similarity_matrices

    def _run_rebalance(
        self,
        train_returns: pd.DataFrame,
        future_returns: pd.DataFrame,
        market_returns: pd.Series,
        returns_index: pd.DatetimeIndex,
        rebalance_index: int,
        horizon: Horizon,
        strategy_specs: Sequence[StrategySpec],
        portfolio_service: PortfolioService,
        risk_service: RiskService,
        similarity_matrices: dict[str, np.ndarray],
        horizon_trades_by_strategy: dict[str, list[float]],
        horizon_market_by_strategy: dict[str, list[float]],
        horizon_trades: list[PortfolioResult],
    ) -> None:
        """Run every strategy for one rebalance and update bookkeeping."""
        for spec in strategy_specs:
            similarity = self._build_consensus_similarity(train_returns, spec, rebalance_index)
            similarity_key = ScenarioKey(spec.name, horizon, rebalance_index)
            similarity_matrices[str(similarity_key)] = similarity

            clusters = stable_clusters_from_similarity(
                similarity, list(train_returns.columns), self.__config.majority_threshold
            )
            selected = self._select_assets(clusters, rebalance_index)
            if not selected:
                continue
            if len(selected) > self.__config.max_assets:
                selected = selected[: self.__config.max_assets]
            if len(selected) < self.__config.min_assets:
                continue

            selected_train = train_returns[selected]
            selected_future = future_returns[selected]
            mse = float(((selected_train - selected_train.mean(axis=0)) ** 2).mean().mean())
            self.__context.governance.record_error(mse)
            if self.__context.governance.is_drift_detected():
                self._emit(
                    PipelineEvent.FORECAST_DRIFT_DETECTED,
                    ForecastDriftPayload(history_points=len(self.__context.governance.snapshot())),
                )

            try:
                weights, _cov, gross, net = portfolio_service.build(
                    selected_assets=selected,
                    train_returns=selected_train,
                    future_returns=selected_future,
                )
            except PortfolioConstructionError:
                continue

            market_trade = float(((1.0 + market_returns).prod()) - 1.0)
            horizon_trades_by_strategy[spec.name].append(net.value)
            horizon_market_by_strategy[spec.name].append(market_trade)
            self.__context.metrics_registry.increment("trades_executed")

            trade = PortfolioResult(
                strategy=spec.name,
                horizon_days=horizon.days,
                rebalance_date=returns_index[rebalance_index],
                exit_date=returns_index[rebalance_index + horizon.days - 1],
                selected_assets=tuple(selected),
                weights=dict(weights.mapping),
                turnover=weights.turnover,
                gross_return=gross.value,
                net_return=net.value,
            )
            horizon_trades.append(trade)

            self._emit(
                PipelineEvent.REBALANCE_EXECUTED,
                RebalanceExecutedPayload(
                    strategy=spec.name,
                    horizon_days=horizon.days,
                    rebalance_index=rebalance_index,
                    n_assets_selected=len(selected),
                    net_return=net.value,
                ),
            )

    def _build_consensus_similarity(
        self,
        train_returns: pd.DataFrame,
        strategy: StrategySpec,
        rebalance_index: int,
    ) -> np.ndarray:
        """Compute the consensus similarity matrix for one strategy at one rebalance."""
        prediction = None
        if strategy.use_prediction:
            prediction = self.__forecast_service.forecast_matrix(
                train_returns,
                self.__config.correlation_window_days,
                self.__config.forecast_method,
                config=self.__config.forecaster,
            )
        partitions: list[list[set[str]]] = []
        assets = list(train_returns.columns)
        for run_index in range(self.__config.consensus_runs):
            shift = run_index if strategy.use_shifts else 0
            end_index = len(train_returns) - shift
            start_index = end_index - self.__config.correlation_window_days
            if start_index < 0:
                continue
            window = train_returns.iloc[start_index:end_index].copy()
            if prediction is not None:
                window = pd.concat([window, prediction], axis=0, ignore_index=True)
            distance = correlation_distance_matrix(window)
            graph = build_weighted_graph_from_distance(distance)
            seed = int(np.random.default_rng(self.__config.random_seed + rebalance_index).integers(0, 2**31 - 1))
            partitions.append(louvain_partition(graph, seed=seed))
        return consensus_similarity_matrix(partitions, assets)

    def _select_assets(self, clusters: list[list[str]], rebalance_index: int) -> list[str]:
        """Draw one asset per cluster, deterministically seeded by ``rebalance_index``."""
        rng = np.random.default_rng(self.__config.random_seed + rebalance_index)
        return [cluster[int(rng.integers(0, len(cluster)))] for cluster in clusters if cluster]

    def _emit(self, event: PipelineEvent, payload: EventPayload) -> None:
        """Publish a typed event to the logger and any registered listener."""
        self.__context.logger.publish(event, payload)
        listener = self.__context.event_listener
        if listener is not None:
            listener(event, payload)

    def __daily_risk_free_rate(self) -> float:
        """Compound the annual risk-free rate to a daily rate using the configured Horizon."""
        return self.__config.horizons[0].annual_to_daily_risk_free_rate(self.__config.risk_free_rate_annual)

    def __derive_risk_limits(self, config: PipelineConfig) -> RiskLimits:
        return RiskLimits(
            max_assets=config.max_assets,
            min_assets=config.min_assets,
            max_weight_per_asset=config.weight_cap,
            max_volatility_annual=config.max_volatility_annual,
        )


__all__ = ["PipelineResult", "PipelineService", "run_pipeline"]


def run_pipeline(
    prices: pd.DataFrame,
    config: PipelineConfig,
    artifact_store: ArtifactStore | None = None,
    logger: StructuredLogger | None = None,
    metrics_registry: MetricsRegistry | None = None,
    governance: ForecastGovernance | None = None,
    forecast_service: ForecastService | None = None,
) -> PipelineResult:
    """Convenience entry point that wires up the default registries.

    Args:
        prices: Raw price ``pd.DataFrame``.
        config: Pipeline configuration.
        artifact_store: Optional :class:`ArtifactStore`. Defaults to a
            no-op when not provided.
        logger: Optional :class:`StructuredLogger`. Defaults to a
            no-op logger when not provided.
        metrics_registry: Optional :class:`MetricsRegistry`. Defaults
            to a fresh registry.
        governance: Optional :class:`ForecastGovernance`. Defaults to
            a fresh instance.
        forecast_service: Optional pre-built :class:`ForecastService`.
            Defaults to one using the built-in registry.

    Returns:
        The :class:`PipelineResult`.
    """
    from ..domain.policies import ForecastGovernance
    from ..infrastructure.observability import StructuredLogger

    if artifact_store is None or logger is None or metrics_registry is None:
        raise ValueError("artifact_store, logger, and metrics_registry are required")
    if forecast_service is None:
        raise ValueError("forecast_service is required")
    context = PipelineContext(
        artifact_store=artifact_store,
        metrics_registry=metrics_registry,
        forecaster_registry=forecast_service.registry,
        governance=governance or ForecastGovernance(),
        logger=logger or StructuredLogger("pipeline"),
    )
    service = PipelineService(
        config=config,
        context=context,
        forecast_service=forecast_service or ForecastService(),
    )
    return service.run(prices)
