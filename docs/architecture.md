# Architecture

## Design Goals
- Keep modules cohesive and single-purpose.
- Preserve loose coupling via explicit typed interfaces.
- Allow strategy, forecasting, ingestor, and optimization components to evolve
  independently.
- Keep the public surface small: a CLI, a Python API, and a stateless HTTP
  surface all share the same core modules.

## Module Boundaries
- `data.py`: ingestion, validation, cleaning, and return-series transformation.
- `ingestors.py`: optional market-data ingestors. Currently provides a
  yfinance-backed multi-asset fetcher behind the `[ingestors]` extra.
- `realtime.py`: ccxt-backed real-time OHLCV polling behind the `[realtime]`
  extra. Writes a long-form CSV consumable by `data.load_price_data`.
- `forecast.py`: return forecasting (naive, ARIMA, GARCH) with failure-safe
  behavior. Owns `GARCHForecastConfig`.
- `lstm_model.py`: shared multi-asset LSTM forecaster behind the
  `[forecast-lstm]` extra.
- `networking.py`: correlation-distance graph construction and consensus
  clustering.
- `portfolio.py`: covariance regularization, optimization, and trade return
  computation.
- `metrics.py`: performance and downside-risk metrics plus tabular summaries.
- `pipeline.py`: orchestration and integration across modules.
- `cli.py`: runtime configuration, two console scripts (`crypto-portfolio`,
  `cps-realtime`), and artifact export.
- `api.py`: stateless FastAPI surface. Each `create_app(base_dir)` factory
  call returns an independent ASGI app whose only state is the on-disk base
  directory.

## Request Lifecycle

```
client → CLI / API / Python API
            │
            ▼
     data ingest (CSV / synthetic / yfinance / ccxt)
            │
            ▼
   pipeline.run_pipeline(prices, config)
            │
            ├── forecast_matrix (naive / ARIMA / GARCH / LSTM)
            ├── consensus clustering + Louvain
            ├── Sharpe optimization + Ledoit-Wolf shrinkage
            ├── risk + execution cost adjustment
            └── governance drift tracking
            │
            ▼
     RunArtifacts ──► disk (CLI) ──► file system (API)
```

The API layer is stateless: every request carries its inputs inline (or
references an on-disk CSV) and the response describes the on-disk artifact
locations. The CLI follows the same pattern by writing outputs to
`--output-dir` and `--run-dir` on disk.

## Scalability Considerations
- Forecasting is isolated, enabling model swaps without pipeline rewrites.
- Strategy specification is centralized in `build_strategy_specs()` for extension.
- All data contracts use pandas structures and dataclasses for predictable composition.
- The ingestor surface (`ingestors.py`, `realtime.py`) is decoupled from the
  pipeline: any source that produces the wide price frame can plug in.
- The API is horizontally scalable because it stores no in-process state.