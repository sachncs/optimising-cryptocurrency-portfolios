"""Domain Protocols.

Defines the structural interfaces that every multi-implementation
component must satisfy. Using ``Protocol`` (rather than abstract base
classes) lets us define duck-typed contracts that any conforming
implementation can fulfil without inheriting from a shared base.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunPaths:
    """Absolute paths of every artifact produced for one run."""

    trades_json: Path
    summary_json: Path
    log_returns_csv: Path
    metrics_json: Path
    events_jsonl: Path
    similarity_dir: Path


@runtime_checkable
class Forecaster(Protocol):
    """Forecasts future returns for one or more assets.

    Concrete implementations live in
    :mod:`cps.infrastructure.forecasters`. The registry returned by
    :func:`cps.infrastructure.forecasters.default_registry` resolves
    a forecaster by name.
    """

    name: ClassVar[str]

    def forecast(
        self,
        returns: pd.DataFrame,
        steps: int,
        *,
        config: "ForecasterConfig | None" = None,
    ) -> pd.DataFrame:
        """Project ``steps`` forward for every column of ``returns``.

        Args:
            returns: ``pd.DataFrame`` of historical returns, one column
                per asset.
            steps: Number of forward steps to project.
            config: Optional forecaster-specific configuration override.

        Returns:
            ``pd.DataFrame`` of shape ``(steps, n_assets)``.
        """
        ...


@dataclass(frozen=True)
class ForecasterConfig:
    """Composite configuration for the forecaster registry.

    Groups the per-algorithm overrides so ``PipelineConfig`` no longer
    needs flat ``lstm_*`` / ``garch_*`` fields.
    """

    garch: "GARCHForecastConfig | None" = None
    lstm: "LSTMTrainingConfig | None" = None


@runtime_checkable
class Ingestor(Protocol):
    """Fetches a wide price frame from an upstream source.

    Concrete implementations live in :mod:`cps.infrastructure.ingestors`.
    """

    name: ClassVar[str]

    def fetch(self) -> pd.DataFrame:
        """Return the fetched price frame."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Persists a :class:`cps.types.RunArtifacts` bundle to durable storage.

    The default implementation is
    :class:`cps.infrastructure.stores.file_artifact_store.FileArtifactStore`.
    In-memory and S3 variants are trivial to add by implementing this
    Protocol.
    """

    def write_run(
        self,
        run_id: str,
        artifacts: "RunArtifacts",  # noqa: F821
        *,
        metrics: Mapping[str, object],
        events: Sequence[Mapping[str, object]],
    ) -> RunPaths:
        """Persist the canonical artifact bundle for one run."""
        ...

    def read_trades(self, run_id: str) -> list[dict[str, object]]:
        """Read the trades JSON for ``run_id``."""
        ...

    def read_summary(self, run_id: str) -> list[dict[str, object]]:
        """Read the summary JSON for ``run_id``."""
        ...

    def read_metrics(self, run_id: str) -> dict[str, object]:
        """Read the metrics JSON for ``run_id``."""
        ...

    def read_log_returns_text(self, run_id: str) -> str:
        """Return the raw CSV text of the log-returns file for ``run_id``."""
        ...

    def run_dir(self, run_id: str) -> Path:
        """Return the directory containing ``run_id``'s artifacts."""
        ...

    def write_upload(self, run_id: str, content: str) -> Path:
        """Persist an uploaded CSV string and return its on-disk path."""
        ...


@runtime_checkable
class ExchangeFactory(Protocol):
    """Constructs a ccxt-like exchange from an identifier.

    Replaces the loose ``Callable[[str], Any]`` previously used for
    ``CCXTPollerConfig.exchange_factory``.
    """

    def __call__(self, exchange_id: str) -> Any:
        """Return an instantiated exchange for ``exchange_id``."""
        ...


@runtime_checkable
class SleepCallable(Protocol):
    """Sleep-like callable used to inject deterministic delays."""

    def __call__(self, seconds: float) -> None:
        """Block (or record) for ``seconds``."""
        ...


@dataclass(frozen=True)
class IngestorRequest:
    """Bundles the inputs the ingestor registry resolves against.

    Used by the CLI / API to construct the right ingestor without the
    surface-level conditional dispatch the previous code used.
    """

    source: str
    csv_path: str | None = None
    yfinance_config: "YFinanceIngestorConfig | None" = None
    ccxt_config: "CCXTPollerConfig | None" = None
    synthetic_seed: int = 7
    synthetic_assets: int = 12
    synthetic_days: int = 500
    date_col: str = "date"


@dataclass(frozen=True)
class PipelineContext:
    """Inputs the :class:`cps.application.pipeline_service.PipelineService`
    consumes beyond the price frame and configuration.

    Centralises all dependencies that the previous god function
    constructed implicitly so callers can swap any of them via
    constructor injection.
    """

    artifact_store: ArtifactStore
    metrics_registry: "MetricsRegistry"  # noqa: F821
    forecaster_registry: "ForecasterRegistry"  # noqa: F821
    governance: "ForecastGovernance"  # noqa: F821
    logger: "StructuredLogger"  # noqa: F821
    event_listener: "EventListener | None" = None
    extra: Mapping[str, object] = field(default_factory=dict)


EventListener = "Callable[[PipelineEvent, EventPayload], None]"