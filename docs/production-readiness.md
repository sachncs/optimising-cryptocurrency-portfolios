# Production Readiness

This implementation includes production controls across reliability, risk, observability, data sources, and HTTP integration.

## Reliability
- Retry with bounded exponential backoff for critical operations via `resilience.execute_with_retry`.
- Idempotent run markers via `runner.ensure_idempotent_run` and `runner.mark_run_complete`.

## Risk and Execution
- Portfolio constraints in `risk.RiskLimits`:
  - minimum and maximum asset count
  - effective per-asset cap enforcement
  - annualized volatility ceiling
- Cost model in `execution`:
  - transaction cost (bps)
  - slippage (bps)
  - net return after costs

## Forecasting
- Naive, ARIMA, GARCH (`arch`-backed, with AIC order selection), and a shared
  multi-asset LSTM (`torch`-backed). Configurable per run via `PipelineConfig`
  and the CLI.

## Data Sources
- Synthetic generator for smoke tests.
- CSV loader for backtests.
- yfinance multi-asset ingestor (`cps.ingestors`).
- ccxt real-time OHLCV poller (`cps.realtime`, `cps-realtime` console script).

## Governance
- Forecast MSE tracking and drift detection in `governance.ForecastGovernance`.

## Observability
- Structured JSON event logging in `observability.StructuredLogger`.
- In-process counters and latency metrics via `observability.MetricsRegistry`.
- CLI emits:
  - `events.jsonl`
  - `metrics.json`

## HTTP API
- Stateless FastAPI surface (`cps.api`).
- No in-process caches or background workers.
- All artifacts persisted to the file system under `base_dir`.
- Scales horizontally behind any stateless WSGI/ASGI runner (uvicorn, gunicorn,
  mod_wsgi).

## CI Quality Gate
- GitHub Actions workflow runs tests on Python 3.10, 3.11, and 3.12.
- Coverage gate enforced at 90% minimum.