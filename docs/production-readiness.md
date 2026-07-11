# Production Readiness

This implementation includes production controls across reliability,
risk, observability, data sources, and HTTP integration.

## Reliability
- Retry with bounded exponential backoff for critical operations via
  `cps.infrastructure.resilience.execute_with_retry`.
- Idempotent run markers via
  `cps.application.run_management.ensure_idempotent_run` and
  `mark_run_complete`.

## Risk and Execution
- Portfolio constraints in `cps.domain.RiskLimits`:
  - minimum and maximum asset count
  - effective per-asset cap enforcement
  - annualized volatility ceiling
- Cost model in `cps.domain.ExecutionCostConfig`:
  - transaction cost (bps)
  - slippage (bps)
  - net return after costs

## Forecasting
- Naive, ARIMA, GARCH (`arch`-backed, with AIC order selection), and a
  shared multi-asset LSTM (`torch`-backed), all behind the
  `Forecaster` Protocol. Configurable per run via `PipelineConfig` and
  the CLI.

## Data Sources
- Synthetic generator for smoke tests.
- CSV loader for backtests.
- yfinance multi-asset ingestor (`cps.infrastructure.ingestors.yfinance`).
- ccxt real-time OHLCV poller (`cps.infrastructure.ingestors.ccxt`,
  `cps-realtime` console script).

## Governance
- Forecast MSE tracking and latching drift detection in
  `cps.domain.ForecastGovernance`.

## Observability
- Structured JSON event logging in
  `cps.infrastructure.observability.StructuredLogger`.
- In-process counters and latency metrics via
  `cps.infrastructure.observability.MetricsRegistry`.
- CLI emits:
  - `events.jsonl`
  - `metrics.json`

## HTTP API
- Stateless FastAPI surface (`cps.interface.api.create_app`).
- No in-process caches or background workers.
- All artifacts persisted to the file system under `base_dir`.
- Scales horizontally behind any stateless WSGI/ASGI runner (uvicorn,
  gunicorn, mod_wsgi).

## Layering

The package follows a layered architecture: domain (pure business
types) → application (orchestration) → infrastructure (adapters) →
interface (CLI / API). New ingestors, forecasters, or artifact stores
plug in behind Protocols and never reach across layers. See
`docs/architecture.md` for the full module map.

## CI Quality Gate
- GitHub Actions workflow runs tests on Python 3.10, 3.11, and 3.12.
- `ruff check` and `mypy --strict` are part of the make check target.
