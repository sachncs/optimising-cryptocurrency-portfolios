"""Stateless FastAPI interface for the crypto portfolio system.

The API exposes a small surface for kicking off portfolio runs and reading
back their artifacts. The service is intentionally **stateless**:

* No process-level caches, connection pools, or background workers.
* Each request carries all inputs (configuration + price data) inline or
  via a reference to a previously uploaded CSV file.
* All artifacts (trades, summary, log returns, metrics, events) are written
  to the file system under a configurable base directory and read back on
  subsequent reads.

Install the optional extra with::

    pip install 'crypto-portfolio-system[api]'

Run the dev server with::

    uvicorn cps.api:create_app --factory --host 0.0.0.0 --port 8000

Or programmatically::

    from cps.api import create_app
    app = create_app(base_dir="/var/lib/cps/data")

The ``base_dir`` argument (default ``./cps_data``) controls where
uploads and run artifacts are stored on disk. Multiple replicas can
share the same ``base_dir`` over a network mount because the service
holds no in-process state.

Error model
-----------
The API surfaces ``ValueError`` raised anywhere in the request lifecycle
as HTTP ``400 Bad Request`` via the registered ``_value_error_handler``
exception handler. Other exceptions propagate as ``500 Internal Server
Error`` (default FastAPI behaviour).
"""

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data import load_price_data
from .metrics import summaries_to_frame
from .observability import MetricsRegistry, StructuredLogger
from .pipeline import PipelineConfig, run_pipeline
from .runner import build_run_id, ensure_idempotent_run, mark_run_complete


def _require_fastapi() -> None:
    """Lazy guard for the optional ``fastapi`` dependency.

    Raises:
        RuntimeError: With a message instructing the caller to install
            the ``[api]`` extra.
    """
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise RuntimeError(
            "The REST API requires the 'fastapi' package. "
            "Install the optional extra with: pip install 'crypto-portfolio-system[api]'"
        ) from exc


def _parse_inline_prices(prices_payload: list[list[float | str]], date_col: str) -> Any:
    """Parse an inline ``[[header...], [row...], ...]`` prices payload.

    Args:
        prices_payload: Header row followed by data rows. The header is
            used to name the columns; the date column is coerced to
            ``pd.Timestamp``.
        date_col: Name of the date column in the payload.

    Returns:
        ``pd.DataFrame`` indexed by date with one column per asset.

    Raises:
        ValueError: When the payload is empty, contains no data rows, or
            is missing the ``date_col``.
    """
    import pandas as pd

    if not prices_payload:
        raise ValueError("'prices' payload is empty")
    header = prices_payload[0]
    rows = prices_payload[1:]
    if not rows:
        raise ValueError("'prices' payload has no data rows")
    df = pd.DataFrame(rows, columns=header)
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not present in 'prices' payload")
    df[date_col] = pd.to_datetime(df[date_col], utc=False)
    df = df.sort_values(date_col).set_index(date_col)
    return df.astype(float)


def _write_inline_csv(content: str, base_dir: Path, run_id: str) -> Path:
    """Persist an inline CSV payload to disk under ``base_dir/uploads``.

    Args:
        content: Raw CSV text supplied by the caller.
        base_dir: Base directory configured for the FastAPI app.
        run_id: Tentative run id used to namespace the upload. The same
            ``run_id`` may not end up matching the actual run because
            the pipeline config is built from the request body -- but
            the upload file is identified by the time of submission, so
            ``"pending"`` is fine here.

    Returns:
        The :class:`pathlib.Path` of the written CSV.
    """
    upload_dir = base_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = upload_dir / f"{run_id}.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def _config_from_payload(payload: dict[str, Any]) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a request payload.

    Unknown keys in ``payload["config"]`` are silently dropped so a
    caller can pass an over-broad dict without breaking the contract.

    Args:
        payload: Parsed JSON body. ``payload["config"]`` (when present)
            supplies the dataclass kwargs.

    Returns:
        A populated :class:`PipelineConfig`.
    """
    # Use ``__import__("dataclasses").fields`` instead of a top-level
    # ``fields`` import so the module remains lightweight for callers
    # that never invoke the API.
    config_payload = payload.get("config", {}) or {}
    known_fields = set(field.name for field in __import__("dataclasses").fields(PipelineConfig))
    filtered = {key: value for key, value in config_payload.items() if key in known_fields}
    return PipelineConfig(**filtered)


def _resolve_prices(payload: dict[str, Any], base_dir: Path, run_id: str) -> Any:
    """Resolve the price frame from the mutually-exclusive input shapes.

    The API accepts exactly one of:

    * ``prices`` -- an inline ``[[header...], [row...], ...]`` payload.
    * ``prices_csv_content`` -- raw CSV text written to disk and
      loaded via :func:`cps.data.load_price_data`.
    * ``prices_csv_path`` -- a server-side path to an existing CSV.
      Treated as a reference; the file is re-read on every call.

    Args:
        payload: Parsed JSON body.
        base_dir: Base directory configured for the FastAPI app.
        run_id: Tentative identifier used to namespace uploaded CSVs.

    Returns:
        ``pd.DataFrame`` of prices.

    Raises:
        ValueError: When none of the three input shapes is supplied.
    """
    prices_payload = payload.get("prices")
    if prices_payload is not None:
        date_col = payload.get("date_col", "date")
        return _parse_inline_prices(prices_payload, date_col)
    csv_content = payload.get("prices_csv_content")
    if csv_content:
        # Persist the upload so subsequent GETs against the same
        # ``run_id`` can resolve the prices that produced the run.
        csv_path = _write_inline_csv(csv_content, base_dir, run_id)
        return load_price_data(str(csv_path), date_col=payload.get("date_col", "date"))
    csv_path_value = payload.get("prices_csv_path")
    if csv_path_value:
        # Read but do not copy: the caller is responsible for the file.
        return load_price_data(csv_path_value, date_col=payload.get("date_col", "date"))
    raise ValueError("Request must include 'prices', 'prices_csv_content', or 'prices_csv_path'.")


def _summaries_to_records(summaries: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of :class:`EvaluationSummary` to JSON-friendly dicts."""
    frame = summaries_to_frame(summaries)
    raw = frame.to_dict(orient="records")
    return [dict(record) for record in raw]


def _trades_to_records(trades: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of :class:`PortfolioResult` to JSON-friendly dicts.

    Timestamps are serialised as ISO-8601 strings; the ``weights`` dict is
    copied verbatim; ``selected_assets`` is materialised as a list.
    """
    records: list[dict[str, Any]] = []
    for trade in trades:
        records.append(
            {
                "strategy": trade.strategy,
                "horizon_days": trade.horizon_days,
                "rebalance_date": trade.rebalance_date.isoformat(),
                "exit_date": trade.exit_date.isoformat(),
                "selected_assets": list(trade.selected_assets),
                "weights": dict(trade.weights),
                "turnover": trade.turnover,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
            }
        )
    return records


def _write_run_artifacts(
    base_dir: Path,
    run_id: str,
    artifacts: Any,
    metrics_registry: MetricsRegistry,
    logger_events: list[dict[str, Any]],
) -> dict[str, str]:
    """Persist every run artifact to disk and return their filenames.

    Args:
        base_dir: Base directory configured for the FastAPI app.
        run_id: Identifier for this run.
        artifacts: :class:`RunArtifacts` returned by the pipeline.
        metrics_registry: In-memory metrics collected during the run.
        logger_events: Captured structured events from the logger.

    Returns:
        Mapping of logical artifact name (``"trades_json"``,
        ``"summary_json"``, ...) to the *relative* filename inside the
        per-run directory.
    """
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_records = _trades_to_records(artifacts.trades)
    summary_records = _summaries_to_records(artifacts.summary)

    (run_dir / "trades.json").write_text(json.dumps(trades_records, default=str), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary_records, default=str), encoding="utf-8")
    (run_dir / "log_returns.csv").write_text(artifacts.returns.to_csv(), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "counters": metrics_registry.counters,
                "timings_millis": metrics_registry.timings_millis,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, default=str) for event in logger_events) + "\n",
        encoding="utf-8",
    )

    similarity_path = run_dir / "similarity"
    similarity_path.mkdir(exist_ok=True)
    for key, matrix in artifacts.similarity_matrices.items():
        (similarity_path / f"{key}.npy").write_bytes(_numpy_save_bytes(matrix))

    return {
        "trades_json": "trades.json",
        "summary_json": "summary.json",
        "log_returns_csv": "log_returns.csv",
        "metrics_json": "metrics.json",
        "events_jsonl": "events.jsonl",
    }


def _numpy_save_bytes(matrix: Any) -> bytes:
    """Serialise a NumPy array to an in-memory ``.npy`` byte buffer.

    Args:
        matrix: 2-D ``np.ndarray`` (a similarity matrix).

    Returns:
        Bytes containing the NumPy ``.npy`` serialisation.
    """
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, matrix, allow_pickle=False)
    return buf.getvalue()


def _numpy_load(path: Path) -> Any:
    """Load a NumPy ``.npy`` file without pickle support.

    Args:
        path: Filesystem path to the ``.npy`` file.

    Returns:
        The deserialised ``np.ndarray``.
    """
    import numpy as np

    return np.load(path, allow_pickle=False)


def _read_json(path: Path) -> Any:
    """Read and JSON-parse a UTF-8 text file.

    Args:
        path: Filesystem path.

    Returns:
        The deserialised Python object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def create_app(base_dir: str | Path = "./cps_data") -> Any:
    """Build a FastAPI app bound to ``base_dir`` for artifact storage.

    The application is stateless: each call to this factory returns an
    independent ASGI app whose only state is the ``base_dir`` path on
    disk. Multiple replicas can share the same ``base_dir`` over a
    network mount because nothing is held in memory between requests.

    Args:
        base_dir: Root directory for uploads and run artifacts. Created
            if it does not already exist. Defaults to ``"./cps_data"``.

    Returns:
        A :class:`fastapi.FastAPI` application exposing the run /
        read-back endpoints listed in the module docstring.

    Raises:
        RuntimeError: When ``fastapi`` is not installed.

    Notes:
        The function intentionally defines its Pydantic models inside
        the factory scope so the module can be imported (and the file
        consumed by ``mypy``) without requiring ``fastapi`` to be
        installed. ``from __future__ import annotations`` is **not**
        used in this module -- Pydantic v2 + FastAPI cannot resolve
        string annotations for inline-defined models, and the previous
        behaviour manifested as silent query-vs-body parameter
        confusion.
    """
    _require_fastapi()
    import pandas as pd
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field

    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    class RunRequest(BaseModel):
        """JSON body for ``POST /api/v1/runs``.

        Attributes:
            config: PipelineConfig overrides. Required so the request
                always carries an explicit configuration; an empty dict
                selects the defaults.
            prices: Inline prices as ``[header_row, data_row_1, ...]``.
                Mutually exclusive with the CSV inputs.
            prices_csv_content: Raw CSV text. Mutually exclusive with
                ``prices`` and ``prices_csv_path``.
            prices_csv_path: Server-side path to a CSV. Mutually
                exclusive with the other two.
            date_col: Name of the date column in the prices / CSV
                payload. Defaults to ``"date"``.
        """

        config: dict[str, Any] = Field(description="PipelineConfig overrides")
        prices: list[list[float | str]] | None = Field(
            default=None,
            description="Inline prices as [header_row, data_row_1, ...].",
        )
        prices_csv_content: str | None = Field(
            default=None,
            description="Raw CSV text. Saved to disk and used as the price source.",
        )
        prices_csv_path: str | None = Field(
            default=None,
            description="Server-side path to a CSV file. Treated as a reference; the file is read on each call.",
        )
        date_col: str = Field(default="date", description="Name of the date column in prices/prices_csv.")

    class RunSummaryResponse(BaseModel):
        """Response body for ``POST /api/v1/runs``.

        Attributes:
            run_id: 16-hex-character identifier of the run.
            config: The fully-resolved pipeline configuration that was
                used.
            artifact_paths: Absolute filesystem paths of the artifacts
                produced by this run.
            trades_count: Number of trades recorded.
            summary_count: Number of per-(strategy, horizon) summaries
                recorded.
        """

        run_id: str
        config: dict[str, Any]
        artifact_paths: dict[str, str]
        trades_count: int
        summary_count: int

    app = FastAPI(
        title="Crypto Portfolio System API",
        version="0.1.0",
        description="Stateless REST interface for running and reading consensus-clustered crypto portfolios.",
    )

    @app.exception_handler(ValueError)
    def _value_error_handler(_request: Any, exc: ValueError) -> Any:
        """Translate ``ValueError`` raised anywhere in the request to HTTP 400.

        Args:
            _request: The ASGI request (unused).
            exc: The raised ``ValueError``.

        Returns:
            ``JSONResponse`` with status code ``400`` and a ``detail``
            payload containing the exception message.
        """
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        """Liveness probe.

        Returns:
            ``{"status": "ok", "base_dir": ...}``.
        """
        return {"status": "ok", "base_dir": str(base_path)}

    @app.post("/api/v1/runs", response_model=RunSummaryResponse)
    def create_run(body: RunRequest) -> RunSummaryResponse:
        """Submit a portfolio run and persist its artifacts.

        Args:
            body: The :class:`RunRequest` payload.

        Returns:
            :class:`RunSummaryResponse` with the run id and on-disk
            artifact paths.

        Raises:
            HTTPException: ``400`` when price resolution fails
                (delegated via :func:`_value_error_handler`).
        """
        config = _config_from_payload(body.model_dump())
        placeholder_run_id = "pending"
        prices = _resolve_prices(body.model_dump(), base_path, placeholder_run_id)

        run_id = build_run_id(config)
        ensure_idempotent_run(str(base_path / "run_markers"), run_id)

        output_dir = base_path / "runs" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = StructuredLogger("cps_api", str(output_dir / "events.jsonl"))
        captured_events: list[dict[str, Any]] = []
        original_log = logger.log_event

        def capturing_log(event: str, payload: dict[str, Any]) -> None:
            """Wrap ``logger.log_event`` to capture events into a list.

            The list is then serialised to ``events.jsonl`` after the run
            finishes so the API surface matches the CLI artifact set.
            """
            captured_events.append({"event": event, **payload})
            original_log(event, payload)

        # ``method-assign`` is the cleanest way to swap a single method
        # on an instance; we ignore the mypy warning because we *want*
        # to substitute the bound method here.
        logger.log_event = capturing_log  # type: ignore[method-assign]

        metrics_registry = MetricsRegistry()
        artifacts = run_pipeline(prices, config, logger, metrics_registry)

        artifact_paths = _write_run_artifacts(
            base_path,
            run_id,
            artifacts,
            metrics_registry,
            captured_events,
        )

        marker_path = base_path / "run_markers" / f"{run_id}.done"
        mark_run_complete(marker_path)

        return RunSummaryResponse(
            run_id=run_id,
            config=asdict(config),
            artifact_paths={name: str(base_path / "runs" / run_id / rel) for name, rel in artifact_paths.items()},
            trades_count=len(artifacts.trades),
            summary_count=len(artifacts.summary),
        )

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Return run metadata.

        Args:
            run_id: Identifier returned by :func:`create_run`.

        Returns:
            ``{"run_id", "trades_count", "summary_count", "artifacts"}``.

        Raises:
            HTTPException: ``404`` when no directory exists for the run.
        """
        run_dir = base_path / "runs" / run_id
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        trades = _read_json(run_dir / "trades.json")
        summary = _read_json(run_dir / "summary.json")
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
        """Return the per-(strategy, horizon) summary for a run.

        Args:
            run_id: Identifier returned by :func:`create_run`.

        Returns:
            ``{"run_id", "summary": [...records...]}``.

        Raises:
            HTTPException: ``404`` when the run does not exist.
        """
        run_dir = base_path / "runs" / run_id
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return {"run_id": run_id, "summary": _read_json(summary_path)}

    @app.get("/api/v1/runs/{run_id}/trades")
    def get_trades(run_id: str, limit: int = Query(default=100, ge=1, le=10_000)) -> dict[str, Any]:
        """Return a (possibly truncated) list of trades for a run.

        Args:
            run_id: Identifier returned by :func:`create_run`.
            limit: Maximum number of records to return. Defaults to
                ``100``; capped at ``10_000``.

        Returns:
            ``{"run_id", "trades": [...], "total", "limit"}``.

        Raises:
            HTTPException: ``404`` when the run does not exist.
        """
        run_dir = base_path / "runs" / run_id
        trades_path = run_dir / "trades.json"
        if not trades_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        trades = _read_json(trades_path)
        truncated = trades[:limit]
        return {"run_id": run_id, "trades": truncated, "total": len(trades), "limit": limit}

    @app.get("/api/v1/runs/{run_id}/metrics")
    def get_metrics(run_id: str) -> dict[str, Any]:
        """Return the run's counters and timing samples.

        Args:
            run_id: Identifier returned by :func:`create_run`.

        Returns:
            ``{"run_id", "metrics": {"counters", "timings_millis"}}``.

        Raises:
            HTTPException: ``404`` when the run does not exist.
        """
        run_dir = base_path / "runs" / run_id
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return {"run_id": run_id, "metrics": _read_json(metrics_path)}

    @app.get("/api/v1/runs/{run_id}/log-returns")
    def get_log_returns(run_id: str, max_rows: int = Query(default=1000, ge=1, le=100_000)) -> dict[str, Any]:
        """Return the leading rows of the cleaned log-returns frame.

        Args:
            run_id: Identifier returned by :func:`create_run`.
            max_rows: Maximum number of rows to include. Defaults to
                ``1000``; capped at ``100_000``.

        Returns:
            ``{"run_id", "rows", "total_rows", "columns", "data"}``
            where ``data`` is a list of ``{"date": iso, ...}`` records.

        Raises:
            HTTPException: ``404`` when the run does not exist.
        """
        run_dir = base_path / "runs" / run_id
        csv_path = run_dir / "log_returns.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
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
