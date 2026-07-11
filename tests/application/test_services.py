"""Tests for the application services."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cps.application import (
    ArtifactService,
    ForecastService,
    PipelineResult,
    PipelineService,
    PortfolioConstructionError,
    PortfolioService,
    RiskService,
    run_pipeline,
)
from cps.config import (
    Horizon,
    PipelineConfig,
)
from cps.domain import (
    ForecastGovernance,
    PipelineContext,
    PipelineEvent,
    RiskLimits,
    Weights,
)
from cps.domain.execution import ExecutionCostConfig
from cps.infrastructure.ingestors import SyntheticIngestor
from cps.infrastructure.observability import MetricsRegistry, StructuredLogger
from cps.infrastructure.stores import FileArtifactStore


class TestForecastService:
    def test_uses_default_registry(self):
        service = ForecastService()
        assert "naive" in service.available()

    def test_custom_registry(self):
        from cps.infrastructure.forecasters import ForecasterRegistry

        class Stub:
            name = "stub"

            def forecast(self, returns, steps, *, config=None):
                return returns.iloc[-steps:]

        registry = ForecasterRegistry()
        registry.register(Stub())
        service = ForecastService(registry=registry)
        assert service.available() == ("stub",)
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        out = service.forecast_matrix(frame, 2, "stub")
        assert out.equals(frame.iloc[-2:])


class TestPortfolioService:
    def test_build_returns_weights_and_returns(self):
        train = pd.DataFrame(
            {"a": [0.01, -0.02, 0.03, 0.04], "b": [0.02, -0.01, 0.0, 0.05]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        future = pd.DataFrame(
            {"a": [0.01, 0.02], "b": [0.03, -0.01]},
            index=pd.date_range("2024-01-05", periods=2, freq="D"),
        )
        limits = RiskLimits(max_assets=5, min_assets=2, max_weight_per_asset=0.6, max_volatility_annual=10.0)
        cost = ExecutionCostConfig(transaction_cost_bps=5.0, slippage_bps=5.0)
        service = PortfolioService(limits, cost, daily_risk_free_rate=0.0, max_iterations=100, learning_step=0.05)
        weights, _cov, gross, net = service.build(["a", "b"], train, future)
        assert isinstance(weights, Weights)
        assert abs(sum(weights.mapping.values()) - 1.0) < 1e-8
        assert gross.value != 0.0
        # Net < gross because the cost model is multiplicative.
        assert net.value < gross.value

    def test_raises_when_below_min_assets(self):
        train = pd.DataFrame({"a": [0.01]}, index=pd.date_range("2024-01-01", periods=1, freq="D"))
        future = train.copy()
        limits = RiskLimits(min_assets=2, max_assets=5, max_weight_per_asset=0.6, max_volatility_annual=10.0)
        cost = ExecutionCostConfig()
        service = PortfolioService(limits, cost, daily_risk_free_rate=0.0, max_iterations=100, learning_step=0.05)
        with pytest.raises(PortfolioConstructionError):
            service.build(["a"], train, future)


class TestRiskService:
    def test_effective_weight_cap(self):
        service = RiskService(RiskLimits())
        assert service.effective_weight_cap(0.05, 5) == pytest.approx(0.2)


class TestArtifactService:
    def test_delegates_to_store(self, tmp_path):
        store = FileArtifactStore(tmp_path)
        service = ArtifactService(store)
        # No runs yet -> FileNotFoundError propagates.
        with pytest.raises(FileNotFoundError):
            service.read_metrics("nope")


class TestPipelineService:
    def _build_context(self, tmp_path: Path) -> tuple[PipelineContext, MetricsRegistry, FileArtifactStore, list]:
        store = FileArtifactStore(tmp_path)
        metrics = MetricsRegistry()
        governance = ForecastGovernance()
        logger = StructuredLogger("pipeline_test")
        captured: list = []

        def listener(event, payload):
            captured.append((event, payload))

        context = PipelineContext(
            artifact_store=store,
            metrics_registry=metrics,
            forecaster_registry=None,
            governance=governance,
            logger=logger,
            event_listener=listener,
        )
        return context, metrics, store, captured

    def test_run_emits_started_and_completed_events(self, tmp_path):
        context, _metrics, _store, captured = self._build_context(tmp_path)
        config = PipelineConfig(
            train_window_days=5,
            correlation_window_days=3,
            rebalance_step_days=2,
            horizons=(Horizon(1),),
            consensus_runs=2,
            max_volatility_annual=5.0,
        )
        service = PipelineService(config, context, ForecastService())
        prices = SyntheticIngestor(days=20, assets=4, seed=3).fetch()
        result = service.run(prices)
        assert isinstance(result, PipelineResult)
        events = [event for event, _ in captured]
        assert PipelineEvent.PIPELINE_STARTED in events
        assert PipelineEvent.PIPELINE_COMPLETED in events

    def test_run_rebalance_event_carries_strategy(self, tmp_path):
        context, _, _, captured = self._build_context(tmp_path)
        config = PipelineConfig(
            train_window_days=5,
            correlation_window_days=3,
            rebalance_step_days=2,
            horizons=(Horizon(1),),
            consensus_runs=2,
            max_volatility_annual=5.0,
        )
        service = PipelineService(config, context, ForecastService())
        prices = SyntheticIngestor(days=20, assets=4, seed=3).fetch()
        service.run(prices)
        # The default four strategies each emit at least one
        # REBALANCE_EXECUTED event (in the smoke test the synthetic
        # data is rich enough to clear the risk filter).
        rebalance_events = [
            payload for event, payload in captured if event == PipelineEvent.REBALANCE_EXECUTED
        ]
        assert rebalance_events
        seen_strategies = {payload.strategy for payload in rebalance_events}
        assert seen_strategies.issubset({"baseline", "s", "p", "p-s"})

    def test_drift_event_emitted_on_spike(self, tmp_path):
        context, _, _, captured = self._build_context(tmp_path)
        # Inject a history into the governance so drift is triggered.
        for _ in range(10):
            context.governance.record_error(0.1)
        context.governance.record_error(0.5)

        config = PipelineConfig(
            train_window_days=5,
            correlation_window_days=3,
            rebalance_step_days=2,
            horizons=(Horizon(1),),
            consensus_runs=2,
            max_volatility_annual=5.0,
        )
        service = PipelineService(config, context, ForecastService())
        prices = SyntheticIngestor(days=20, assets=4, seed=3).fetch()
        service.run(prices)
        drift_events = [p for e, p in captured if e == PipelineEvent.FORECAST_DRIFT_DETECTED]
        assert drift_events

    def test_run_pipeline_convenience_requires_store_logger_metrics(self, tmp_path):
        prices = SyntheticIngestor(days=20, assets=4, seed=3).fetch()
        config = PipelineConfig(
            train_window_days=5,
            correlation_window_days=3,
            rebalance_step_days=2,
            horizons=(Horizon(1),),
            consensus_runs=2,
            max_volatility_annual=5.0,
        )
        with pytest.raises(ValueError):
            run_pipeline(prices, config)
