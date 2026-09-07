<p align="center">
  <h1 align="center">Crypto Portfolio System</h1>
  <p align="center">Consensus-clustered cryptocurrency portfolio construction.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/optimising-cryptocurrency-portfolios/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/optimising-cryptocurrency-portfolios/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/crypto-portfolio-system/"><img src="https://img.shields.io/pypi/v/crypto-portfolio-system" alt="PyPI"></a>
    <a href="https://github.com/sachncs/optimising-cryptocurrency-portfolios/stargazers"><img src="https://img.shields.io/github/stars/sachncs/optimising-cryptocurrency-portfolios" alt="Stars"></a>
  </p>
</p>

**Crypto Portfolio System** is a production-hardened implementation of the framework in
[arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2) for cryptocurrency portfolio
construction through consensus clustering.

It ingests price data, forecasts returns, builds rolling correlation networks, extracts
stable asset clusters via consensus Louvain community detection, then performs Sharpe-ratio
portfolio optimization with covariance regularization, risk limits, and execution costs.

---

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

---

## Installation

### From PyPI

```bash
pip install crypto-portfolio-system
```

### From source

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

---

## Quick Start

### CLI

```bash
# 1. Synthetic data
crypto-portfolio --output-dir outputs --run-dir runs

# 2. CSV input
crypto-portfolio --prices-csv /path/to/prices.csv --date-col date --output-dir outputs --run-dir runs

# 3. yfinance ingestor (requires pip install -e ".[ingestors]")
pip install -e ".[ingestors]"
crypto-portfolio \
  --source yfinance \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --period 6mo \
  --ingest-output-csv prices.csv \
  --output-dir outputs --run-dir runs

# 4. Real-time ccxt poller (requires pip install -e ".[realtime]")
pip install -e ".[realtime]"
cps-realtime \
  --exchange binance \
  --symbols BTC/USDT,ETH/USDT \
  --output-csv prices.csv \
  --timeframe 1m --interval-seconds 60 --max-iterations 5

# 5. REST API (requires pip install -e ".[api]")
pip install -e ".[api]"
uvicorn cps.interface.api:create_app --factory --host 0.0.0.0 --port 8000
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

---

## Configuration

### CLI flags

| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--train-window-days` | — | `180` | Rolling training window size |
| `--corr-window-days` | — | `60` | Rolling correlation window |
| `--rebalance-step-days` | — | `30` | Days between rebalances |
| `--horizons` | — | `1,3,7,14` | Forecast horizons |
| `--consensus-runs` | — | `20` | Number of consensus Louvain runs |
| `--majority-threshold` | — | `0.5` | Co-occurrence threshold for stable clusters |
| `--rf-annual` | — | `0.045` | Annual risk-free rate |
| `--forecast-method` | — | `arima` | `naive`, `arima`, `garch`, `lstm` |
| `--weight-cap` | — | `0.35` | Maximum per-asset portfolio weight |
| `--max-assets` | — | `25` | Hard ceiling on the portfolio size |
| `--min-assets` | — | `2` | Minimum portfolio size |
| `--max-volatility-annual` | — | `1.2` | Annualised volatility cap |
| `--transaction-cost-bps` | — | `10` | Transaction cost in basis points |
| `--slippage-bps` | — | `5` | Slippage in basis points |
| `--seed` | — | `42` | Random seed for reproducibility |

### ccxt poller (`cps-realtime`)

| Flag | Default | Description |
|------|---------|-------------|
| `--exchange` | _required_ | ccxt exchange id (e.g. `binance`) |
| `--symbols` | _required_ | Comma-separated `BASE/QUOTE` symbols |
| `--output-csv` | _required_ | Path for long-form OHLCV CSV |
| `--timeframe` | `1m` | ccxt timeframe string |
| `--interval-seconds` | `60` | Poll interval |
| `--max-iterations` | `5` | Number of polls to perform (`0` = infinite) |

The poller writes a long-form OHLCV CSV. Convert it to the wide price frame
the pipeline expects with:

```python
from cps.infrastructure.ingestors import pivot_to_price_frame
prices = pivot_to_price_frame("prices.csv")
```

### REST API

```bash
# Submit a run
curl -X POST http://localhost:8000/api/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{
        "config": {"forecast_method": "arima", "horizons": [1, 3, 7]},
        "prices_csv_content": "date,btc,eth\n2024-01-01,42000,2200\n..."
      }'

# Read artifacts back
curl http://localhost:8000/api/v1/runs/{run_id}/summary
curl http://localhost:8000/api/v1/runs/{run_id}/trades
curl http://localhost:8000/api/v1/runs/{run_id}/metrics
curl http://localhost:8000/api/v1/runs/{run_id}/log-returns
```

---

## Outputs

| File | Description |
|------|-------------|
| `trades.csv` | Per-rebalance trade records with gross and net returns |
| `summary.csv` | Strategy-level aggregated metrics |
| `log_returns.csv` | Cleaned log-returns time series |
| `events.jsonl` | Structured runtime events |
| `metrics.json` | Counters and timing metrics |

---

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `PipelineConfig` | class | Forecast + portfolio configuration object |
| `run_pipeline` | function | Top-level orchestration entry point |
| `ForecastService` | class | Pluggable forecasting backend registry |
| `StructuredLogger` | class | JSONL structured logger |
| `MetricsRegistry` | class | Counters and timing metrics |
| `ForecastGovernance` | class | Drift detection + MSE tracking over forecast runs |
| `FileArtifactStore` | class | Persistent file-based artifact store |
| `LongFormCsvStore` | class | Stores long-form CSVs (e.g. OHLCV polls) |
| `create_app` | function | FastAPI factory for the REST surface |

---

## Examples

```bash
# 1. Default synthetic-data end-to-end run.
crypto-portfolio --output-dir outputs --run-dir runs

# 2. CSV-driven run with a long-window ARIMA forecast.
crypto-portfolio \
  --prices-csv data/daily_prices.csv --date-col date \
  --forecast-method arima --horizons 1,3,7,14 \
  --train-window-days 240 --corr-window-days 90 \
  --output-dir outputs --run-dir runs

# 3. yfinance multi-asset ingestor + pipeline.
pip install -e ".[ingestors]"
crypto-portfolio \
  --source yfinance --symbols BTC-USD,ETH-USD,SOL-USD --period 6mo \
  --ingest-output-csv prices.csv \
  --output-dir outputs --run-dir runs

# 4. Real-time ccxt poller (long-running).
pip install -e ".[realtime]"
cps-realtime \
  --exchange binance --symbols BTC/USDT,ETH/USDT \
  --output-csv prices.csv --timeframe 1m --interval-seconds 60

# 5. REST API server.
pip install -e ".[api]"
uvicorn cps.interface.api:create_app --factory --host 0.0.0.0 --port 8000
```

---

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

---

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
make lint-fix    # Auto-fix lint issues
make format      # Format code
make typecheck   # Run type checking
make help        # Show all commands
```

---

## Testing

```bash
pytest
pytest --cov=cps
```

---

## Build

```bash
python -m build
```

---

## Release

See [docs/deployment.md](docs/deployment.md) — version is bumped in `pyproject.toml`,
the changelog updated, a `vX.Y.Z` tag is cut, and the PyPI publishing workflow
publishes the source and wheel distributions.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Numerical | [NumPy](https://numpy.org/) |
| Data | [pandas](https://pandas.pydata.org/) |
| Graphs | [NetworkX](https://networkx.org/) (Louvain community detection) |
| Statistics | [statsmodels](https://www.statsmodels.org/) (ARIMA) |
| Volatility | [arch](https://bashtage.github.io/arch/) (GARCH — `[forecast-garch]`) |
| Deep Learning | [PyTorch](https://pytorch.org/) (LSTM — `[forecast-lstm]`) |
| Market Data | [yfinance](https://github.com/ranaroussi/yfinance) (`[ingestors]`), [ccxt](https://github.com/ccxt/ccxt) (`[realtime]`) |
| REST API | [FastAPI](https://fastapi.tiangolo.com/) / [Uvicorn](https://www.uvicorn.org/) (`[api]`) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Lint/Format | [ruff](https://docs.astral.sh/ruff/) |
| Type Check | [mypy](https://mypy-lang.org/) |
| Hooks | [pre-commit](https://pre-commit.com/) |

---

## Roadmap

- [x] Additional forecasting methods (GARCH, LSTM)
- [x] Real-time data ingestion (ccxt polling)
- [x] REST API interface (FastAPI, stateless)
- [x] Multi-asset class support (yfinance ingestor)
- [x] Docker containerization
- [ ] Web dashboard
- [ ] Streaming ingestion backplane (WebSocket / message broker)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

[MIT](LICENSE) © 2026 Sachin
