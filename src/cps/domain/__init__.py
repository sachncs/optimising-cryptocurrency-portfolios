"""Domain layer: pure business types and rules.

Contains:

* :mod:`cps.domain.primitives` -- value objects (``Weights``, ``Horizon``,
  ``GrossReturn``, ``NetReturn``, ``ScenarioKey``, ``CovarianceMatrix``)
* :mod:`cps.domain.artifacts` -- ``PortfolioResult``, ``EvaluationSummary``,
  ``RunArtifacts`` (and the freeze helpers)
* :mod:`cps.domain.events` -- typed pipeline events
* :mod:`cps.domain.protocols` -- structural interfaces
  (``Forecaster``, ``Ingestor``, ``ArtifactStore``, ``ExchangeFactory``)
* :mod:`cps.domain.policies` -- risk limits and drift governance
* :mod:`cps.domain.execution` -- execution-cost configuration
* :mod:`cps.domain.networking` -- correlation graphs and consensus Louvain
* :mod:`cps.domain.portfolio_math` -- Ledoit-Wolf shrinkage, simplex
  projection, Sharpe ascent

The domain layer is intentionally pure: no I/O, no network, no
side effects. Application services depend on it; infrastructure
adapters implement its Protocols.
"""

from __future__ import annotations

from .artifacts import (
    EvaluationSummary,
    PortfolioResult,
    RunArtifacts,
    freeze_similarity_matrices,
    freeze_summary,
    freeze_trades,
)
from .events import (
    EventPayload,
    ForecastDriftPayload,
    PipelineCompletedPayload,
    PipelineEvent,
    PipelineStartedPayload,
    RebalanceExecutedPayload,
)
from .execution import ExecutionCostConfig, apply_execution_costs, compute_total_cost_rate
from .metrics_snapshot import MetricsSnapshot
from .networking import (
    build_weighted_graph_from_distance,
    consensus_similarity_matrix,
    correlation_distance_matrix,
    louvain_partition,
    stable_clusters_from_similarity,
)
from .policies import (
    MIN_HISTORY_FOR_DRIFT,
    ForecastGovernance,
    RiskLimits,
    apply_weight_cap,
    compute_effective_weight_cap,
)
from .portfolio_math import (
    compute_ledoit_wolf_constant_variance_covariance,
    compute_portfolio_simple_return,
    optimize_maximum_sharpe_ratio,
    project_weights_to_simplex,
)
from .primitives import (
    CovarianceMatrix,
    GrossReturn,
    Horizon,
    NetReturn,
    ScenarioKey,
    Weights,
)
from .protocols import (
    ArtifactStore,
    EventListener,
    ExchangeFactory,
    Forecaster,
    Ingestor,
    IngestorRequest,
    PipelineContext,
    RunPaths,
    SleepCallable,
)

__all__ = [
    "MIN_HISTORY_FOR_DRIFT",
    "ArtifactStore",
    "CovarianceMatrix",
    "EvaluationSummary",
    "EventListener",
    "EventPayload",
    "ExchangeFactory",
    "ExecutionCostConfig",
    "ForecastDriftPayload",
    "ForecastGovernance",
    "Forecaster",
    "GrossReturn",
    "Horizon",
    "Ingestor",
    "IngestorRequest",
    "MetricsSnapshot",
    "NetReturn",
    "PipelineCompletedPayload",
    "PipelineContext",
    "PipelineEvent",
    "PipelineStartedPayload",
    "PortfolioResult",
    "RebalanceExecutedPayload",
    "RiskLimits",
    "RunArtifacts",
    "RunPaths",
    "ScenarioKey",
    "SleepCallable",
    "Weights",
    "apply_execution_costs",
    "apply_weight_cap",
    "build_weighted_graph_from_distance",
    "compute_effective_weight_cap",
    "compute_ledoit_wolf_constant_variance_covariance",
    "compute_portfolio_simple_return",
    "compute_total_cost_rate",
    "consensus_similarity_matrix",
    "correlation_distance_matrix",
    "freeze_similarity_matrices",
    "freeze_summary",
    "freeze_trades",
    "louvain_partition",
    "optimize_maximum_sharpe_ratio",
    "project_weights_to_simplex",
    "stable_clusters_from_similarity",
]
