"""FastAPI routes for the stateless REST surface.

Endpoints:

* ``POST /api/v1/runs`` -- submit a run, returns ``run_id`` and artifact paths.
* ``GET  /api/v1/runs/{run_id}`` -- run metadata.
* ``GET  /api/v1/runs/{run_id}/summary`` -- per (strategy, horizon) summary.
* ``GET  /api/v1/runs/{run_id}/trades?limit=`` -- trade records (paginated).
* ``GET  /api/v1/runs/{run_id}/metrics`` -- counters and timings.
* ``GET  /api/v1/runs/{run_id}/log-returns?max_rows=`` -- cleaned log-returns frame.
* ``GET  /api/v1/health`` -- liveness probe.

The application is stateless; every call to :func:`create_app` returns
an independent ASGI app whose only state is the on-disk base directory.

Annotations are *not* stringised: Pydantic models are defined inside
:func:`create_app` so FastAPI can resolve their annotations eagerly.
"""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ...application import (
    ArtifactService,
    ForecastService,
    build_run_id,
    ensure_idempotent_run,
    mark_run_complete,
    run_pipeline,
)
from ...config import PipelineConfig
from ...domain import (
    ArtifactStore,
    EventListener,
    EventPayload,
    PipelineEvent,
)
from ...domain.policies import ForecastGovernance
from ...infrastructure.observability import MetricsRegistry, StructuredLogger
from ...infrastructure.stores import FileArtifactStore


def require_fastapi() -> None:
    """Lazy import guard for the optional ``fastapi`` dependency."""
    from ...infrastructure.resilience import require_optional

    require_optional("fastapi", "api")


def _parse_inline_prices(prices_payload: Sequence[Sequence[float | str]], date_col: str) -> pd.DataFrame:
    """Parse an inline ``[[header...], [row...], ...]`` prices payload."""
    if not prices_payload:
        raise ValueError("'prices' payload is empty")
    header = list(prices_payload[0])
    rows = [list(row) for row in prices_payload[1:]]
    if not rows:
        raise ValueError("'prices' payload has no data rows")
    frame = pd.DataFrame(rows, columns=header)
    if date_col not in frame.columns:
        raise ValueError(f"Date column '{date_col}' not present in 'prices' payload")
    frame[date_col] = pd.to_datetime(frame[date_col], utc=False)
    frame = frame.sort_values(date_col).set_index(date_col)
    return frame.astype(float)


def _resolve_prices(payload: dict[str, Any], artifact_store: ArtifactStore, run_id: str) -> pd.DataFrame:
    """Resolve the price frame from the mutually-exclusive input shapes."""
    prices_payload = payload.get("prices")
    if prices_payload is not None:
        return _parse_inline_prices(prices_payload, payload.get("date_col", "date"))
    csv_content = payload.get("prices_csv_content")
    if csv_content:
        csv_path = artifact_store.write_upload(run_id, csv_content)
        from ...application import load_price_data

        return load_price_data(str(csv_path), date_col=payload.get("date_col", "date"))
    csv_path_value = payload.get("prices_csv_path")
    if csv_path_value:
        from ...application import load_price_data

        return load_price_data(csv_path_value, date_col=payload.get("date_col", "date"))
    raise ValueError("Request must include 'prices', 'prices_csv_content', or 'prices_csv_path'.")


def _config_from_payload(payload: dict[str, Any]) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a request payload."""
    import dataclasses

    config_payload = payload.get("config", {}) or {}
    known_fields = {field.name for field in dataclasses.fields(PipelineConfig)}
    filtered = {key: value for key, value in config_payload.items() if key in known_fields}
    return PipelineConfig(**filtered)


def _capture_events(logger: StructuredLogger, sink: list[dict[str, Any]]) -> EventListener:
    """Return an :class:`EventListener` that appends events to ``sink``."""

    def listener(event: PipelineEvent, payload: EventPayload) -> None:
        sink.append({"event": event.value, **asdict(payload)})

    logger.add_listener(listener)
    return listener


def create_app(base_dir: str | Path = "./cps_data") -> Any:
    """Build a FastAPI app bound to ``base_dir`` for artifact storage.

    Args:
        base_dir: Root directory for uploads and run artifacts.

    Returns:
        The configured :class:`fastapi.FastAPI` application.

    Raises:
        RuntimeError: When ``fastapi`` is not installed.
    """
    require_fastapi()
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field

    class RunRequest(BaseModel):
        """JSON body for ``POST /api/v1/runs``."""

        config: dict[str, Any] = Field(description="PipelineConfig overrides")
        prices: list[list[float | str]] | None = Field(
            default=None,
            description="Inline prices as [header_row, data_row_1, ...].",
        )
        prices_csv_content: str | None = Field(default=None)
        prices_csv_path: str | None = Field(default=None)
        date_col: str = Field(default="date")

    class RunSummaryResponse(BaseModel):
        """Response body for ``POST /api/v1/runs``."""

        run_id: str
        config: dict[str, Any]
        artifact_paths: dict[str, str]
        trades_count: int
        summary_count: int

    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    artifact_store = FileArtifactStore(base_path)
    artifact_service = ArtifactService(artifact_store)

    app = FastAPI(
        title="Crypto Portfolio System API",
        version="0.2.0",
        description="Stateless REST interface for running and reading consensus-clustered crypto portfolios.",
    )

    @app.exception_handler(ValueError)
    def value_error_handler(_request: Any, exc: ValueError) -> Any:
        """Translate ``ValueError`` raised anywhere in the request to HTTP 400."""
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    def key_error_handler(_request: Any, exc: KeyError) -> Any:
        """Translate ``KeyError`` (e.g. unknown forecast method) to HTTP 400."""
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "base_dir": str(base_path)}

    @app.post("/api/v1/runs", response_model=RunSummaryResponse)
    def create_run(body: RunRequest) -> RunSummaryResponse:
        """Submit a portfolio run and persist its artifacts."""
        config = _config_from_payload(body.model_dump())
        run_id = build_run_id(config)
        ensure_idempotent_run(str(base_path / "run_markers"), run_id)
        prices = _resolve_prices(body.model_dump(), artifact_store, run_id)

        run_dir = base_path / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = StructuredLogger("cps_api", run_dir / "events.jsonl")
        metrics_registry = MetricsRegistry()
        captured_events: list[dict[str, Any]] = []
        listener = _capture_events(logger, captured_events)

        try:
            result = run_pipeline(
                prices,
                config,
                artifact_store=artifact_store,
                logger=logger,
                metrics_registry=metrics_registry,
                governance=ForecastGovernance(),
                forecast_service=ForecastService(),
            )
        finally:
            logger.remove_listener(listener)

        artifact_paths = artifact_store.write_run(
            run_id,
            result.artifacts,
            metrics=asdict(metrics_registry.snapshot()),
            events=captured_events,
        )
        mark_run_complete(base_path / "run_markers" / f"{run_id}.done")

        return RunSummaryResponse(
            run_id=run_id,
            config=asdict(config),
            artifact_paths={
                field: str(getattr(artifact_paths, field))
                for field in (
                    "trades_json",
                    "summary_json",
                    "log_returns_csv",
                    "metrics_json",
                    "events_jsonl",
                    "similarity_dir",
                )
            },
            trades_count=len(result.trades),
            summary_count=len(result.summaries),
        )

    def run_or_404(run_id: str) -> None:
        """Raise 404 when the run directory does not exist."""
        try:
            artifact_store.run_dir(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found") from exc

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Return run metadata."""
        run_or_404(run_id)
        trades = artifact_service.read_trades(run_id)
        summary = artifact_service.read_summary(run_id)
        return {
            "run_id": run_id,
            "trades_count": len(trades),
            "summary_count": len(summary),
            "artifacts": {
                "trades_json": "trades.json",
                "summary_json": "summary.json",
                "log_returns_csv": "log_returns.csv",
                "metrics_json": "metrics.json",
                "events_jsonl": "events.jsonl",
            },
        }

    @app.get("/api/v1/runs/{run_id}/summary")
    def get_summary(run_id: str) -> dict[str, Any]:
        """Return the per-(strategy, horizon) summary for a run."""
        run_or_404(run_id)
        return {"run_id": run_id, "summary": artifact_service.read_summary(run_id)}

    @app.get("/api/v1/runs/{run_id}/trades")
    def get_trades(run_id: str, limit: int = Query(default=100, ge=1, le=10_000)) -> dict[str, Any]:
        """Return a (possibly truncated) list of trades for a run."""
        run_or_404(run_id)
        trades = artifact_service.read_trades(run_id)
        return {"run_id": run_id, "trades": trades[:limit], "total": len(trades), "limit": limit}

    @app.get("/api/v1/runs/{run_id}/metrics")
    def get_metrics(run_id: str) -> dict[str, Any]:
        """Return the run's counters and timing samples."""
        run_or_404(run_id)
        return {"run_id": run_id, "metrics": artifact_service.read_metrics(run_id)}

    @app.get("/api/v1/runs/{run_id}/log-returns")
    def get_log_returns(run_id: str, max_rows: int = Query(default=1000, ge=1, le=100_000)) -> dict[str, Any]:
        """Return the leading rows of the cleaned log-returns frame."""
        run_or_404(run_id)
        frame = artifact_service.read_log_returns(run_id)
        truncated = frame.head(max_rows)
        data: list[dict[str, Any]] = []
        for index_value, row in truncated.iterrows():
            record: dict[str, Any] = {"date": index_value.isoformat()}
            record.update({column: float(row[column]) for column in truncated.columns})
            data.append(record)
        return {
            "run_id": run_id,
            "rows": truncated.shape[0],
            "total_rows": frame.shape[0],
            "columns": list(truncated.columns),
            "data": data,
        }

    return app


__all__ = ["create_app"]
