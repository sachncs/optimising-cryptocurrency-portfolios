"""Domain layer: pure business types and rules.

Contains:

* :mod:`cps.domain.primitives` -- value objects (``Weights``, ``Horizon``,
  ``GrossReturn``, ``NetReturn``, ``ScenarioKey``, ``CovarianceMatrix``)
* :mod:`cps.domain.events` -- typed pipeline events
* :mod:`cps.domain.protocols` -- structural interfaces
  (``Forecaster``, ``Ingestor``, ``ArtifactStore``, ``ExchangeFactory``)
* :mod:`cps.domain.policies` -- risk limits and drift governance

The domain layer is intentionally pure: no I/O, no network, no
side effects. Application services depend on it; infrastructure
adapters implement its Protocols.
"""

from __future__ import annotations

from .events import (
    EventPayload,
    ForecastDriftPayload,
    PipelineCompletedPayload,
    PipelineEvent,
    PipelineStartedPayload,
    RebalanceExecutedPayload,
)
from .execution import ExecutionCostConfig, compute_total_cost_rate, apply_execution_costs
from .metrics_snapshot import MetricsSnapshot
from .policies import (
    ForecastGovernance,
    MIN_HISTORY_FOR_DRIFT,
    RiskLimits,
    apply_weight_cap,
    compute_effective_weight_cap,
)
from .primitives import (
    CovarianceMatrix,
    GrossReturn,
    Horizon,
    NetReturn,
    ScenarioKey,
    Weights,
    freeze_similarity_matrices,
    freeze_summary,
    freeze_trades,
)
from .protocols import (
    ArtifactStore,
    EventListener,
    ExchangeFactory,
    Forecaster,
    ForecasterConfig,
    Ingestor,
    IngestorRequest,
    PipelineContext,
    RunPaths,
    SleepCallable,
)

__all__ = [
    "ArtifactStore",
    "CovarianceMatrix",
    "EventListener",
    "EventPayload",
    "ExecutionCostConfig",
    "ExchangeFactory",
    "Forecaster",
    "ForecasterConfig",
    "ForecastDriftPayload",
    "ForecastGovernance",
    "GrossReturn",
    "Horizon",
    "Ingestor",
    "IngestorRequest",
    "MIN_HISTORY_FOR_DRIFT",
    "MetricsSnapshot",
    "NetReturn",
    "PipelineCompletedPayload",
    "PipelineContext",
    "PipelineEvent",
    "PipelineStartedPayload",
    "RebalanceExecutedPayload",
    "RiskLimits",
    "RunPaths",
    "ScenarioKey",
    "SleepCallable",
    "Weights",
    "apply_execution_costs",
    "apply_weight_cap",
    "compute_effective_weight_cap",
    "compute_total_cost_rate",
    "freeze_similarity_matrices",
    "freeze_summary",
    "freeze_trades",
]