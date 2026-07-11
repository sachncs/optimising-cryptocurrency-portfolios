# API Reference

## Core Entry Points
- `cps.application.run_pipeline(prices, config, *, artifact_store, logger, metrics_registry, governance, forecast_service)`
- `cps.interface.cli.main()`
- `cps.interface.cli.realtime_main(argv)`
- `cps.interface.api.create_app(base_dir)`

## Key Configuration
`PipelineConfig` fields (with defaults shown):

| Field | Default | Notes |
|-------|---------|-------|
| `train_window_days` | `180` | Training-window length in calendar days. |
| `correlation_window_days` | `60` | Rolling correlation window used inside consensus clustering. |
| `rebalance_step_days` | `30` | Days between rebalances. |
| `horizons` | `(Horizon(1), Horizon(3), Horizon(7), Horizon(14))` | Tuple of `Horizon` value objects. |
| `consensus_runs` | `20` | Independent Louvain partitions per rebalance. |
| `majority_threshold` | `0.5` | Co-membership cutoff for declaring two assets stable. |
| `risk_free_rate_annual` | `0.045` | Annualised risk-free rate. |
| `forecast_method` | `"arima"` | One of `naive`, `arima`, `garch`, `lstm`. |
| `random_seed` | `42` | NumPy RNG seed for Louvain passes. |
| `weight_cap` | `0.35` | Per-asset cap. |
| `max_assets`, `min_assets` | `25`, `2` | Operational risk limits. |
| `max_volatility_annual` | `1.2` | Annualised portfolio-volatility ceiling. |
| `transaction_cost_bps`, `slippage_bps` | `10.0`, `5.0` | Execution cost bps. |
| `forecaster` | `ForecasterConfig()` | Composite GARCH + LSTM overrides. |

`PipelineConfig.with_overrides(seed=...)` accepts the `seed` /
`rf_annual` aliases for the canonical `random_seed` /
`risk_free_rate_annual` names.

## Output Artifacts
`cps.domain.RunArtifacts` includes:

- `returns`: cleaned log-returns time series.
- `market_returns`: equal-weight market proxy series.
- `trades`: per-rebalance trade records (tuple of `PortfolioResult`).
- `summary`: per-strategy and per-horizon aggregated metrics (tuple of
  `EvaluationSummary`).
- `similarity_matrices`: consensus co-membership matrices keyed by
  `ScenarioKey`.

## Application services (`cps.application`)
- `PipelineService` / `PipelineResult` — orchestrator.
- `PortfolioService` (and `PortfolioConstructionError`) — construction.
- `ForecastService` — forecaster dispatch.
- `RiskService` — operational limits facade.
- `ArtifactService` — typed read-back of `ArtifactStore`.

## Forecaster registry (`cps.infrastructure.forecasters`)
- `default_registry()` returns a registry pre-populated with
  `naive`, `arima`, `garch`, `lstm`.
- Add a custom forecaster by implementing the `Forecaster` Protocol
  from `cps.domain.protocols`.

## Ingestors (`cps.infrastructure.ingestors`)
- `SyntheticIngestor`, `CsvIngestor`.
- `YFinanceIngestor` / `YFinanceConfig` / `YFinanceField` /
  `YFinanceInterval` / `fetch_yfinance_prices` — behind the
  `[ingestors]` extra.
- `CCXTPoller` / `CCXTIngestorConfig` / `default_exchange_factory` /
  `default_sleep` / `resolve_exchange_factory` / `pivot_to_price_frame`
  — behind the `[realtime]` extra. Console script: `cps-realtime`.

## Real-time Poller

```python
from cps.infrastructure.ingestors import CCXTPoller, CCXTIngestorConfig

config = CCXTIngestorConfig(
    exchange_id="binance",
    symbols=("BTC/USDT", "ETH/USDT"),
    output_csv="prices.csv",
    timeframe="1m",
    interval_seconds=60.0,
    max_iterations=5,
)
CCXTPoller(config).run()
```

## REST API

`cps.interface.api.create_app(base_dir)` builds a FastAPI app:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Liveness probe + base dir. |
| `POST` | `/api/v1/runs` | Submit a run; returns `run_id` and artifact paths. |
| `GET` | `/api/v1/runs/{run_id}` | Run metadata. |
| `GET` | `/api/v1/runs/{run_id}/summary` | Per-strategy metrics. |
| `GET` | `/api/v1/runs/{run_id}/trades?limit=` | Trade records (paginated). |
| `GET` | `/api/v1/runs/{run_id}/metrics` | Counters + timings. |
| `GET` | `/api/v1/runs/{run_id}/log-returns?max_rows=` | Head of log-returns frame. |

The service is stateless; all artifacts live under `base_dir`. Requires
the `[api]` extra (`fastapi`, `uvicorn`).

## Domain primitives (`cps.domain`)
- `Weights` (+ `Weights.equal_weight`, `Weights.from_series`).
- `Horizon` (+ `annual_to_daily_risk_free_rate`).
- `GrossReturn`, `NetReturn` (the cost model).
- `CovarianceMatrix` (+ `from_dataframe`).
- `ScenarioKey`.
- `PortfolioResult`, `EvaluationSummary`, `RunArtifacts`.
