# Getting Started

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Installation

### From Source

```bash
git clone https://github.com/sachncs/optimising-cryptocurrency-portfolios.git
cd optimising-cryptocurrency-portfolios
pip install -e ".[dev]"
```

## Quick Start

### Run with Synthetic Data

The fastest way to see the system in action:

```bash
crypto-portfolio --output-dir outputs --run-dir runs
```

This generates synthetic cryptocurrency price data and runs the full pipeline.

### Run with Your Data

Prepare a CSV file with dates and asset prices:

```csv
date,bitcoin,ethereum,solana
2024-01-01,42000,2200,100
2024-01-02,42500,2250,105
...
```

Run the pipeline:

```bash
crypto-portfolio --prices-csv prices.csv --date-col date --output-dir outputs --run-dir runs
```

## Understanding the Output

After running, check the `outputs/` directory:

- **trades.csv** - Every portfolio rebalance with selected assets, weights, and returns
- **summary.csv** - Aggregated performance metrics by strategy and horizon
- **log_returns.csv** - Time series of log returns for all assets
- **events.jsonl** - Structured log of pipeline events
- **metrics.json** - Performance counters and timing data

## Next Steps

- Read the [Architecture Guide](architecture.md) for design details
- See the [API Reference](api.md) for programmatic usage
- Review [Production Readiness](production-readiness.md) for operational controls
