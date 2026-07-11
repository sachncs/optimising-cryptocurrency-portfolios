"""Cryptocurrency portfolio optimisation through consensus clustering.

This package implements the consensus-clustered cryptocurrency portfolio
construction framework described in `arXiv:2505.24831v2
<https://arxiv.org/abs/2505.24831v2>`_. The end-to-end pipeline:

1. Ingests price data (CSV, yfinance, ccxt real-time, or synthetic).
2. Forecasts returns (naive, ARIMA, GARCH, or a shared multi-asset LSTM).
3. Builds rolling correlation networks from the returns window.
4. Runs Louvain community detection multiple times and aggregates the
   partitions into a consensus similarity matrix.
5. Extracts stable clusters above a majority co-membership threshold.
6. Optimises long-only portfolio weights per cluster draw using
   Ledoit-Wolf-regularised covariance and Sharpe-ratio ascent.
7. Validates risk limits, applies transaction costs, and writes
   ``trades.csv``, ``summary.csv``, ``log_returns.csv``, ``events.jsonl``,
   and ``metrics.json`` to disk.
8. Tracks forecast drift via :class:`cps.governance.ForecastGovernance`.

Public surface
--------------
The package exposes two top-level symbols::

    from cps import PipelineConfig, run_pipeline

``PipelineConfig`` is the dataclass that controls every pipeline stage;
``run_pipeline`` is the single entry point that takes a price frame and
a config and returns a :class:`cps.types.RunArtifacts` container with
all of the run's outputs.

Optional extras
---------------
Forecasting (``arch``, ``torch``), real-time ingestion (``ccxt``),
yfinance pull (``yfinance``), and the REST API (``fastapi``,
``uvicorn``) are gated behind optional extras so the default install
stays small. See ``pyproject.toml`` for the full list.

Examples:
    >>> import pandas as pd
    >>> from cps import PipelineConfig, run_pipeline
    >>> prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
    >>> artifacts = run_pipeline(prices, PipelineConfig(forecast_method="naive"))
    >>> artifacts.trades[0].net_return  # doctest: +SKIP
    0.0123
"""

from .pipeline import PipelineConfig, run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
