# Frequently Asked Questions

## General

### What is this project?

A production-hardened system for constructing cryptocurrency portfolios using consensus clustering, based on the research paper [arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2).

### What cryptocurrencies are supported?

Any cryptocurrency with historical price data. The system is asset-agnostic and works with any set of price series. The yfinance ingestor supports Yahoo!-listed symbols (`BTC-USD`, `ETH-USD`, ...), and the ccxt poller supports any exchange that ccxt supports.

### Is this financial advice?

No. This is a research and educational tool. Always consult a qualified financial advisor before making investment decisions.

## Data

### What format should my data be in?

CSV files with:
- A date column (configurable name)
- One column per asset with price values

### How much data do I need?

- Minimum: enough for the training window (default 180 days)
- Recommended: at least 1 year of daily data

### Can I use real-time data?

Yes. The `cps-realtime` console script polls OHLCV candles via ccxt and
writes them to a long-form CSV. Install the realtime extra and run:

```bash
pip install -e ".[realtime]"
cps-realtime --exchange binance --symbols BTC/USDT,ETH/USDT \
  --output-csv prices.csv --timeframe 1m --interval-seconds 60 --max-iterations 5
```

You can then pivot the CSV into the wide price frame the pipeline expects:

```python
from cps.realtime import pivot_to_price_frame
prices = pivot_to_price_frame("prices.csv")
```

### How do I fetch data from Yahoo! Finance?

```bash
pip install -e ".[ingestors]"
crypto-portfolio \
  --source yfinance \
  --symbols BTC-USD,ETH-USD,SOL-USD \
  --period 6mo \
  --ingest-output-csv prices.csv \
  --output-dir outputs --run-dir runs
```

## Technical

### Why are there multiple consensus runs?

Multiple runs with different random seeds produce more stable cluster assignments. The majority threshold determines how consistently assets must cluster together.

### What is the difference between naive, ARIMA, GARCH, and LSTM forecasting?

- **Naive**: Uses the most recent return as the forecast.
- **ARIMA**: Uses autoregressive integrated moving average models for more sophisticated predictions.
- **GARCH**: Models mean and variance jointly with a GARCH(p,o,q) process; order is selected by AIC over a small grid when `auto_order` is enabled. Requires `arch` (`[forecast-garch]`).
- **LSTM**: Trains a single shared multi-asset LSTM over the entire panel; suitable when you want non-linear cross-asset interactions. Requires `torch` (`[forecast-lstm]`).

### What do the horizon parameters mean?

Horizons (e.g., 1,3,7,14) represent the number of days forward for return forecasting and portfolio evaluation.

### How are transaction costs applied?

Transaction costs and slippage are deducted from gross returns to produce net returns in the trade records.

### Is the REST API stateful?

No. `cps.api.create_app(base_dir)` builds a FastAPI app whose only state is
the on-disk base directory. Each request carries its inputs inline and all
artifacts are written to / read from that directory. You can run multiple
replicas behind a shared network mount.

## Troubleshooting

### Tests fail with import errors

Ensure you're running from the project root with PYTHONPATH set:

```bash
PYTHONPATH=src pytest -q
```

### Coverage is below 90%

Add tests for uncovered code paths. Check the coverage report for specific lines:

```bash
python -m coverage report -m
```

### `forecast-method garch` complains about a missing `arch` package

Install the GARCH extra: `pip install -e ".[forecast-garch]"`.

### `forecast-method lstm` complains about a missing `torch` package

Install the LSTM extra: `pip install -e ".[forecast-lstm]"`.

### `--source yfinance` complains about a missing `yfinance` package

Install the ingestors extra: `pip install -e ".[ingestors]"`.