"""Typed pipeline events.

Every event the pipeline emits is modelled as an instance of
:class:`PipelineEvent` carrying a typed payload dataclass. Listeners
match on the event kind and read strongly-typed attributes, eliminating
the ``dict[str, Any]`` payload black hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PipelineEvent(str, Enum):
    """Enumeration of every event the pipeline emits.

    Inheriting from ``str`` keeps the string identifiers used by the
    existing log format; ``Enum`` gives the listener a typed switch.
    """

    PIPELINE_STARTED = "pipeline_started"
    REBALANCE_EXECUTED = "rebalance_executed"
    FORECAST_DRIFT_DETECTED = "forecast_drift_detected"
    PIPELINE_COMPLETED = "pipeline_completed"


@dataclass(frozen=True)
class PipelineStartedPayload:
    """Payload for :attr:`PipelineEvent.PIPELINE_STARTED`."""

    rows: int
    assets: int


@dataclass(frozen=True)
class RebalanceExecutedPayload:
    """Payload for :attr:`PipelineEvent.REBALANCE_EXECUTED`."""

    strategy: str
    horizon_days: int
    rebalance_index: int
    n_assets_selected: int
    net_return: float


@dataclass(frozen=True)
class ForecastDriftPayload:
    """Payload for :attr:`PipelineEvent.FORECAST_DRIFT_DETECTED`."""

    history_points: int


@dataclass(frozen=True)
class PipelineCompletedPayload:
    """Payload for :attr:`PipelineEvent.PIPELINE_COMPLETED`."""

    trades: int
    summaries: int
    duration_millis: float


EventPayload = PipelineStartedPayload | RebalanceExecutedPayload | ForecastDriftPayload | PipelineCompletedPayload
