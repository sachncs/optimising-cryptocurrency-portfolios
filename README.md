<div align="center">

# Crypto Portfolio System

**Consensus-clustered cryptocurrency portfolio construction.**

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/sachncs/optimising-cryptocurrency-portfolios/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/optimising-cryptocurrency-portfolios/ci.yml?branch=master" alt="CI"></a>
  <a href="https://github.com/sachncs/optimising-cryptocurrency-portfolios/stargazers"><img src="https://img.shields.io/github/stars/sachncs/optimising-cryptocurrency-portfolios" alt="Stars"></a>
</p>

</div>

---

**Crypto Portfolio System** is a production-hardened implementation of the framework in
[arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2) for cryptocurrency portfolio
construction through consensus clustering.

It ingests price data, forecasts returns, builds rolling correlation networks, extracts
stable asset clusters via consensus Louvain community detection, then performs Sharpe-ratio
portfolio optimization with covariance regularization, risk limits, and execution costs.

## Features

- **Return Forecasting** — Naive, ARIMA, GARCH (with AIC order selection), and a shared multi-asset LSTM
- **Correlation Networks** — Rolling correlation matrices with Louvain community detection
- **Consensus Clustering** — Stable cluster extraction across multiple runs
- **Portfolio Optimization** — Sharpe-ratio maximization with covariance regularization
- **Risk Management** — Asset count limits, per-asset caps, volatility ceilings
- **Execution Modeling** — Transaction costs and slippage applied to net returns
- **Production Controls** — Retry logic, idempotent runs, structured logging
- **Governance** — Forecast drift detection and MSE tracking
- **Data Sources** — Synthetic generator, CSV loader, yfinance multi-asset ingestor, and a ccxt real-time poller
- **REST API** — Stateless FastAPI surface for running and reading artifacts via HTTP

## Installation

```bash
git clone https://github.com/sachncs/optimising-cryptocurrency-portfolios.git
cd optimising-cryptocurrency-portfolios
pip install -e ".[dev]"
```

The forecasting, real-time, ingestor, and API features live behind optional
extras to keep the default install small:

```bash
pip install -e ".[all]"                # everything
pip install -e ".[forecast-garch]"     # arch-backed GARCH forecasting
pip install -e ".[forecast-lstm]"      # torch-backed LSTM forecasting
pip install -e ".[ingestors]"          # yfinance price ingestor
pip install -e ".[realtime]"           # ccxt real-time poller
pip install -e ".[api]"                # FastAPI REST interface
```

## Usage

### Synthetic Data

```bash
crypto-portfolio --output-dir outputs --run-dir runs
```

### CSV Input

```bash
crypto-portfolio --prices-csv /path/to/prices.csv --date-col date --output-dir outputs --run-dir runs
```

### yfinance Ingestor

```bash
pip install -e ".[ingestors]"
crypto-portfolio \
  --source yfinance \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --period 6mo \
  --ingest-output-csv prices.csv \
  --output-dir outputs --run-dir runs
```

### Real-time ccxt Poller

```bash
pip install -e ".[realtime]"
cps-realtime \
  --exchange binance \
  --symbols BTC/USDT,ETH/USDT \
  --output-csv prices.csv \
  --timeframe 1m --interval-seconds 60 --max-iterations 5
```

The poller writes a long-form OHLCV CSV. Convert it to the wide price frame
the pipeline expects with:

```python
from cps.infrastructure.ingestors import pivot_to_price_frame
prices = pivot_to_price_frame("prices.csv")
```

### REST API

```bash
pip install -e ".[api]"
uvicorn cps.interface.api:create_app --factory --host 0.0.0.0 --port 8000
```

Submit a run:

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{
        "config": {"forecast_method": "arima", "horizons": [1, 3, 7]},
        "prices_csv_content": "date,btc,eth\n2024-01-01,42000,2200\n..."
      }'
```

Read artifacts back via `/api/v1/runs/{run_id}/summary`, `/trades`,
`/metrics`, and `/log-returns`.

### All Options

```bash
crypto-portfolio \
  --train-window-days 180 \
  --corr-window-days 60 \
  --rebalance-step-days 30 \
  --horizons 1,3,7,14 \
  --consensus-runs 20 \
  --majority-threshold 0.5 \
  --rf-annual 0.045 \
  --forecast-method arima \
  --weight-cap 0.35 \
  --max-assets 25 \
  --min-assets 2 \
  --max-volatility-annual 1.2 \
  --transaction-cost-bps 10 \
  --slippage-bps 5 \
  --seed 42
```

### Python API

```python
from cps import (
    PipelineConfig,
    ForecastService,
    StructuredLogger,
    MetricsRegistry,
    ForecastGovernance,
    FileArtifactStore,
    run_pipeline,
)

store = FileArtifactStore("outputs")
logger = StructuredLogger("crypto_portfolio", "outputs/events.jsonl")
metrics = MetricsRegistry()
governance = ForecastGovernance()
config = PipelineConfig(forecast_method="arima", random_seed=42)

artifacts = run_pipeline(
    prices,
    config,
    artifact_store=store,
    logger=logger,
    metrics_registry=metrics,
    governance=governance,
    forecast_service=ForecastService(),
)
```

## Outputs

| File | Description |
|------|-------------|
| `trades.csv` | Per-rebalance trade records with gross and net returns |
| `summary.csv` | Strategy-level aggregated metrics |
| `log_returns.csv` | Cleaned log-returns time series |
| `events.jsonl` | Structured runtime events |
| `metrics.json` | Counters and timing metrics |

## Project Structure

The package follows a layered architecture (see [docs/architecture.md](docs/architecture.md) for the full module map):

```
optimising-cryptocurrency-portfolios/
├── src/cps/
│   ├── config/        # PipelineConfig, ForecasterConfig, central settings
│   ├── domain/        # Pure value objects, events, protocols, policies
│   ├── application/   # PipelineService, PortfolioService, ForecastService, ...
│   ├── infrastructure/
│   │   ├── forecasters/   # naive, ARIMA, GARCH, LSTM
│   │   ├── ingestors/     # synthetic, csv, yfinance, ccxt poller
│   │   ├── observability/ # StructuredLogger, MetricsRegistry, Timer
│   │   ├── resilience/    # RetryPolicy, require_optional
│   │   └── stores/        # FileArtifactStore, LongFormCsvStore
│   └── interface/
│       ├── cli/           # crypto-portfolio, cps-realtime scripts
│       └── api/           # create_app, FastAPI routes
├── tests/             # Test suite (mirrors the layered layout)
│   ├── application/
│   ├── config/
│   ├── domain/
│   ├── infrastructure/
│   └── interface/
├── docs/              # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── deployment.md
│   ├── faq.md
│   ├── getting-started.md
│   └── production-readiness.md
└── pyproject.toml     # Project configuration
```

## Development

```bash
# Install with dev dependencies and pre-commit hooks
make dev

# Run all checks (lint, typecheck, test)
make check

# Or run individually
make test        # Run tests
make test-cov    # Run tests with coverage
make lint        # Run linting
make lint-fix    # Auto-fix linting issues
make format      # Format code
make typecheck   # Run type checking
make help        # Show all commands
```

## Tech Stack

- **Python 3.10+**
- **NumPy** — Numerical computation
- **pandas** — Data manipulation
- **NetworkX** — Graph algorithms
- **statsmodels** — Statistical models (ARIMA)
- **arch** — GARCH volatility models (`[forecast-garch]`)
- **PyTorch** — Multi-asset LSTM (`[forecast-lstm]`)
- **yfinance** — Yahoo! Finance market data (`[ingestors]`)
- **ccxt** — Real-time OHLCV polling (`[realtime]`)
- **FastAPI / Uvicorn** — Stateless REST surface (`[api]`)
- **pytest** — Testing framework
- **ruff** — Linting and formatting
- **mypy** — Static type checking
- **pre-commit** — Git hooks

## Roadmap

- [x] Additional forecasting methods (GARCH, LSTM)
- [x] Real-time data ingestion (ccxt polling)
- [x] REST API interface (FastAPI, stateless)
- [x] Multi-asset class support (yfinance ingestor)
- [x] Docker containerization
- [ ] Web dashboard
- [ ] Streaming ingestion backplane (WebSocket / message broker)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

[MIT](LICENSE)