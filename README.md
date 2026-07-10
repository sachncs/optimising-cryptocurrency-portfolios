# Crypto Portfolio System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/sachncs/optimising-cryptocurrency-portfolios/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/optimising-cryptocurrency-portfolios/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Production-hardened implementation of the framework in [arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2) for cryptocurrency portfolio construction through consensus clustering.

## Features

- **Return Forecasting** - Naive and ARIMA-based return prediction
- **Correlation Networks** - Rolling correlation matrices with Louvain community detection
- **Consensus Clustering** - Stable cluster extraction across multiple runs
- **Portfolio Optimization** - Sharpe-ratio maximization with covariance regularization
- **Risk Management** - Asset count limits, per-asset caps, volatility ceilings
- **Execution Modeling** - Transaction costs and slippage applied to net returns
- **Production Controls** - Retry logic, idempotent runs, structured logging
- **Governance** - Forecast drift detection and MSE tracking

## Installation

```bash
git clone https://github.com/sachncs/optimising-cryptocurrency-portfolios.git
cd optimising-cryptocurrency-portfolios
pip install -e ".[dev]"
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
from cps import PipelineConfig, run_pipeline

config = PipelineConfig(forecast_method="arima", random_seed=42)
artifacts = run_pipeline(prices, config)
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

```
optimising-cryptocurrency-portfolios/
├── src/cps/           # Core package
│   ├── cli.py         # CLI entrypoint
│   ├── pipeline.py    # Orchestration
│   ├── data.py        # Data ingestion
│   ├── forecast.py    # Return forecasting
│   ├── networking.py  # Correlation networks
│   ├── portfolio.py   # Portfolio optimization
│   ├── risk.py        # Risk constraints
│   ├── execution.py   # Cost modeling
│   ├── metrics.py     # Performance metrics
│   ├── governance.py  # Forecast governance
│   ├── observability.py # Logging and metrics
│   ├── resilience.py  # Retry logic
│   ├── runner.py      # Run management
│   └── types.py       # Data types
├── tests/             # Test suite
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
- **NumPy** - Numerical computation
- **pandas** - Data manipulation
- **NetworkX** - Graph algorithms
- **statsmodels** - Statistical models (ARIMA)
- **pytest** - Testing framework
- **ruff** - Linting and formatting
- **mypy** - Static type checking
- **pre-commit** - Git hooks

## Roadmap

- [ ] Additional forecasting methods (GARCH, LSTM)
- [ ] Real-time data ingestion
- [ ] Web dashboard
- [ ] Docker containerization
- [ ] REST API interface
- [ ] Multi-asset class support

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

[MIT](LICENSE)
