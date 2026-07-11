# API Reference

## Core Entry Points
- `cps.pipeline.run_pipeline(prices, config)`
- `cps.cli.main()`
- `cps.cli.realtime_main(argv)`
- `cps.api.create_app(base_dir)`

## Key Configuration
`PipelineConfig` fields (with defaults shown):
- `train_window_days` (180)
- `correlation_window_days` (60)
- `rebalance_step_days` (30)
- `horizons_days` ([1, 3, 7, 14])
- `consensus_runs` (20)
- `majority_threshold` (0.5)
- `risk_free_rate_annual` (0.045)
- `forecast_method` (`arima`; one of `naive | arima | garch | lstm`)
- `random_seed` (42)
- `weight_cap` (0.35)
- `max_assets` (25)
- `min_assets` (2)
- `max_volatility_annual` (1.2)
- `transaction_cost_bps` (10.0)
- `slippage_bps` (5.0)
- `lstm_lookback` / `lstm_hidden_size` / `lstm_num_layers` / `lstm_max_epochs`
- `garch_p` / `garch_o` / `garch_q` / `garch_mean` / `garch_dist` /
  `garch_auto_order`

## Output Artifacts
`RunArtifacts` includes:
- `returns`: cleaned log-returns time series.
- `market_returns`: equal-weight market proxy series.
- `trades`: per-rebalance trade records.
- `summary`: per-strategy and per-horizon aggregated metrics.
- `similarity_matrices`: consensus co-membership matrices by scenario.

## Ingestors
- `cps.ingestors.YFinanceIngestorConfig`
- `cps.ingestors.fetch_yfinance_prices(config)`
- `cps.ingestors.fetch_yfinance_symbols(symbols, *, start, end, period,
  interval, field, auto_adjust)`
- Requires the `[ingestors]` extra (`yfinance`).

## Real-time Poller
- `cps.realtime.CCXTPollerConfig`
- `cps.realtime.poll_once(config)`
- `cps.realtime.run_polling_loop(config, max_iterations)`
- `cps.realtime.pivot_to_price_frame(csv_path, *, date_col, value_col)`
- Requires the `[realtime]` extra (`ccxt`).
- New console script: `cps-realtime`.

## REST API
`create_app(base_dir)` builds a FastAPI app:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/health` | Liveness probe + base dir. |
| `POST` | `/api/v1/runs` | Submit a run; returns `run_id` and artifact paths. |
| `GET` | `/api/v1/runs/{run_id}` | Run metadata. |
| `GET` | `/api/v1/runs/{run_id}/summary` | Per-strategy metrics. |
| `GET` | `/api/v1/runs/{run_id}/trades?limit=` | Trade records (paginated). |
| `GET` | `/api/v1/runs/{run_id}/metrics` | Counters + timings. |
| `GET` | `/api/v1/runs/{run_id}/log-returns?max_rows=` | Head of log-returns frame. |

The service is stateless; all artifacts live under `base_dir`. Requires the
`[api]` extra (`fastapi`, `uvicorn`).