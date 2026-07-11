# Architecture

## Design Goals
- Keep modules cohesive and single-purpose.
- Preserve loose coupling via explicit typed interfaces.
- Allow strategy, forecasting, ingestor, and optimization components to evolve
  independently.
- Keep the public surface small: a CLI, a Python API, and a stateless HTTP
  surface all share the same core modules.

## Layered Architecture

```
cps/
├── config/        configuration types + central settings constants
├── domain/        pure business types and rules (no IO)
├── application/   orchestration services
├── infrastructure/ adapters to libraries and the file system
└── interface/     CLI + REST API
```

### `cps.domain` — pure business types and rules

| Module | Responsibility |
|--------|----------------|
| `domain/primitives.py` | ``Weights``, ``Horizon``, ``GrossReturn``, ``NetReturn``, ``ScenarioKey``, ``CovarianceMatrix`` value objects with construction-time invariants. |
| `domain/artifacts.py` | ``PortfolioResult``, ``EvaluationSummary``, ``RunArtifacts`` + the freeze helpers used by the application services. |
| `domain/events.py` | Typed pipeline events and their payload dataclasses. |
| `domain/protocols.py` | Structural interfaces (``Forecaster``, ``Ingestor``, ``ArtifactStore``, ``ExchangeFactory``, ``SleepCallable``, ``PipelineContext``, ``EventListener``). |
| `domain/policies.py` | ``RiskLimits``, ``apply_weight_cap``, ``ForecastGovernance`` (drift detection). |
| `domain/execution.py` | ``ExecutionCostConfig`` + the cost-rate / net-return math. |
| `domain/networking.py` | Correlation-distance matrix, weighted graph construction, Louvain partition, consensus similarity aggregation, stable clusters. |
| `domain/portfolio_math.py` | Ledoit-Wolf shrinkage, simplex projection, Sharpe-ratio ascent, simple-return compounding. |

The domain layer is intentionally pure: no I/O, no network, no side
effects. Application services depend on it; infrastructure adapters
implement its Protocols.

### `cps.application` — orchestration services

| Module | Responsibility |
|--------|----------------|
| `application/pipeline_service.py` | Orchestrates every stage of one pipeline run. ``PipelineService.run`` is the single entry point; ``run_pipeline`` is the convenience wrapper. |
| `application/portfolio_service.py` | Wraps Ledoit-Wolf + Sharpe ascent + risk-cap + cost model into a single ``build`` call. |
| `application/forecast_service.py` | Dispatches ``forecast_matrix(method=...)`` through the forecaster registry. |
| `application/risk_service.py` | Thin wrapper over ``RiskLimits`` so callers can pass the limits handle around. |
| `application/artifact_service.py` | Typed read-back facade over the on-disk ``ArtifactStore``. |
| `application/data_cleaning.py` | ``load_price_data``, ``clean_price_data``, ``log_returns``, ``market_proxy``. |
| `application/portfolio_metrics.py` | Per-trade and per-strategy evaluation metrics. |
| `application/run_management.py` | ``build_run_id``, ``ensure_idempotent_run``, ``mark_run_complete``. |

### `cps.infrastructure` — adapters

| Module | Responsibility |
|--------|----------------|
| `infrastructure/forecasters/` | ``NaiveForecaster``, ``ArimaForecaster``, ``GarchForecaster`` (arch), ``LstmForecaster`` + ``LstmForecasterFactory`` (torch) and the registry. |
| `infrastructure/ingestors/` | ``SyntheticIngestor``, ``CsvIngestor``, ``YFinanceIngestor``, ``CCXTPoller``. |
| `infrastructure/observability/` | ``MetricsRegistry``, ``StructuredLogger``, ``Timer``. |
| `infrastructure/resilience/` | ``RetryPolicy`` + ``execute_with_retry``, ``require_optional``. |
| `infrastructure/stores/` | ``FileArtifactStore`` (on-disk layout), ``LongFormCsvStore`` (ccxt poller). |

### `cps.config` — configuration

| Module | Responsibility |
|--------|----------------|
| `config/pipeline_config.py` | ``PipelineConfig``, ``StrategySpec``, ``ForecasterConfig``, ``GARCHForecastConfig``, ``LSTMTrainingConfig``, ``default_strategy_specs``, ``Horizon`` re-export. |
| `config/settings.py` | ``ANNUAL_TRADING_DAYS``, ``BPS_DENOMINATOR``, ``SHARPE_*``, ``LEDOIT_WOLF_*``, ``GARCH_AUTO_ORDER_CANDIDATES``, ``CCXT_*``. |

### `cps.interface` — entry points

| Module | Responsibility |
|--------|----------------|
| `interface/cli/` | Console scripts ``crypto-portfolio`` and ``cps-realtime``. |
| `interface/api/` | Stateless FastAPI factory ``create_app(base_dir)``. |

## Request Lifecycle

```
client → CLI / API / Python API
            │
            ▼
   ingestors (synthetic / csv / yfinance / ccxt)
            │
            ▼
  pipeline.run_pipeline(prices, config)
            │
            ├── forecast_matrix (naive / ARIMA / GARCH / LSTM)
            ├── correlation graph + consensus Louvain clustering
            ├── Sharpe optimization + Ledoit-Wolf shrinkage
            ├── risk + execution cost adjustment
            └── governance drift tracking
            │
            ▼
       RunArtifacts ──► artifact store ──► disk + JSON / NDJSON
```

Each layer only knows about the layer below it; infrastructure
adapters implement domain Protocols, application services glue the
domain together, and the interface exposes ``run_pipeline`` /
``create_app`` to the outside world.

The API layer is stateless: every request carries its inputs inline (or
references an on-disk CSV) and the response describes the on-disk artifact
locations. The CLI follows the same pattern by writing outputs to
``--output-dir`` and ``--run-dir`` on disk.

## Scalability Considerations
- Forecasting is isolated behind the ``Forecaster`` Protocol, enabling model
  swaps without pipeline rewrites.
- Strategy specification lives in ``default_strategy_specs()`` for extension.
- All data contracts use pandas structures and dataclasses for predictable composition.
- The ingestor surface (``cps.infrastructure.ingestors``) is decoupled from the
  pipeline: any source that produces the wide price frame can plug in.
- The API is horizontally scalable because it stores no in-process state.
