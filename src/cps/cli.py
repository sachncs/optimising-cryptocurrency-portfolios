"""Console-script entry points for the crypto-portfolio-system.

This module owns the two user-facing entry points registered in
``pyproject.toml``:

* ``crypto-portfolio`` -- :func:`main`, the canonical backtest runner.
  Reads configuration from ``sys.argv``, sources price data from one of
  ``synthetic``, ``csv``, or ``yfinance``, runs the pipeline, and writes
  ``trades.csv``, ``summary.csv``, ``log_returns.csv``, ``events.jsonl``,
  and ``metrics.json`` to ``--output-dir``.
* ``cps-realtime`` -- :func:`realtime_main`, a polling daemon that
  appends OHLCV candles from a single ccxt exchange to a long-form CSV.
  See :mod:`cps.realtime` for the underlying primitives.

Exit codes
----------
Both entry points follow the convention:

* ``0`` -- success.
* ``1`` -- the pipeline or poller raised an exception (errors are
  logged via :class:`cps.observability.StructuredLogger`).
* ``2`` -- argument validation failed (``argparse.error``).

Environment variables
---------------------
The ``cli`` module itself reads no environment variables directly;
configuration is via ``sys.argv``. The REST API and the Docker image
read ``CPS_OUTPUT_DIR`` / ``CPS_RUN_DIR``; those are owned by
:mod:`cps.runner` and the Docker entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_price_data
from .metrics import summaries_to_frame
from .observability import MetricsRegistry, StructuredLogger
from .pipeline import PipelineConfig, run_pipeline
from .resilience import RetryConfig, execute_with_retry
from .runner import build_run_id, ensure_idempotent_run, mark_run_complete


def generate_synthetic_prices(days: int = 500, assets: int = 12, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic price frame for smoke tests and CI.

    The synthetic price paths come from a 3-factor latent model::

        factors_t   ~ N(0.0005, 0.02)        # shape (T, 3)
        exposures   ~ N(0, 1)                # shape (n_assets, 3)
        idio_t      ~ N(0, 0.015)            # shape (T, n_assets)
        returns_t   = factors_t @ exposures.T + idio_t
        prices_t    = 100 * exp(cumsum(returns_t))

    The 3 factors provide shared market / sector exposure; the idiosyncratic
    term keeps individual asset paths from being perfectly collinear.

    Args:
        days: Number of calendar days of prices. Defaults to ``500``.
        assets: Number of asset columns. Defaults to ``12``.
        seed: RNG seed for reproducibility. Defaults to ``7``.

    Returns:
        ``pd.DataFrame`` indexed by date with one column per asset, all
        values strictly positive.
    """
    random_generator = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    factors = random_generator.normal(0.0005, 0.02, size=(days, 3))
    exposures = random_generator.normal(0, 1, size=(assets, 3))
    idiosyncratic = random_generator.normal(0, 0.015, size=(days, assets))
    returns = factors @ exposures.T + idiosyncratic
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"asset_{asset_index:02d}" for asset_index in range(assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)


def parse_horizons(horizons_text: str) -> list[int]:
    """Parse a comma-separated list of horizons.

    Args:
        horizons_text: Comma-separated positive integers (e.g. ``"1,3,7"``).

    Returns:
        ``list[int]`` of horizons in the order they appear.

    Raises:
        ValueError: When the string is empty or contains non-positive
            integers.
    """
    horizons = [int(value.strip()) for value in horizons_text.split(",") if value.strip()]
    if not horizons:
        raise ValueError("At least one horizon value is required")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizon values must be positive integers")
    return horizons


def parse_arguments() -> argparse.Namespace:
    """Parse the ``crypto-portfolio`` command-line arguments.

    The parser is intentionally permissive about unknown extensions --
    callers can add new flags without breaking the existing surface. Run
    ``crypto-portfolio --help`` for the full list of options.
    """
    parser = argparse.ArgumentParser(description="Consensus-clustered crypto portfolio system")
    parser.add_argument("--prices-csv", type=str, default="", help="CSV with date column and asset price columns")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument(
        "--source",
        choices=["auto", "synthetic", "csv", "yfinance"],
        default="auto",
        help="Price data source. 'yfinance' requires the [ingestors] extra. "
        "'auto' (default) infers from other arguments: --symbols -> yfinance, "
        "--prices-csv -> csv, otherwise synthetic.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated yfinance symbols (e.g. 'BTC-USD,ETH-USD'). Required when --source=yfinance.",
    )
    parser.add_argument("--start", type=str, default="", help="yfinance start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="", help="yfinance end date (YYYY-MM-DD).")
    parser.add_argument(
        "--period",
        type=str,
        default="",
        help="yfinance period string (e.g. '1y', '6mo'). Used when --start/--end are not supplied.",
    )
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval (default: 1d).")
    parser.add_argument(
        "--field",
        choices=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        default="Close",
        help="yfinance OHLCV field used as the price series (default: Close).",
    )
    parser.add_argument(
        "--ingest-output-csv",
        type=str,
        default="",
        help="If set, the ingested yfinance frame is written to this CSV path before pipeline execution.",
    )
    parser.add_argument("--train-window-days", type=int, default=180)
    parser.add_argument("--corr-window-days", type=int, default=60)
    parser.add_argument("--rebalance-step-days", type=int, default=30)
    parser.add_argument("--horizons", type=str, default="1,3,7,14")
    parser.add_argument("--consensus-runs", type=int, default=20)
    parser.add_argument("--majority-threshold", type=float, default=0.5)
    parser.add_argument("--rf-annual", type=float, default=0.045)
    parser.add_argument(
        "--forecast-method",
        choices=["arima", "naive", "garch", "lstm"],
        default="arima",
        help="Return forecasting method. 'garch' requires [forecast-garch] extra; "
        "'lstm' requires [forecast-lstm] extra.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-cap", type=float, default=0.35)
    parser.add_argument("--max-assets", type=int, default=25)
    parser.add_argument("--min-assets", type=int, default=2)
    parser.add_argument("--max-volatility-annual", type=float, default=1.2)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--lstm-lookback", type=int, default=10)
    parser.add_argument("--lstm-hidden-size", type=int, default=16)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--lstm-max-epochs", type=int, default=80)
    parser.add_argument("--garch-p", type=int, default=1)
    parser.add_argument("--garch-o", type=int, default=1)
    parser.add_argument("--garch-q", type=int, default=1)
    parser.add_argument("--garch-mean", choices=["Zero", "Constant", "AR"], default="Zero")
    parser.add_argument("--garch-dist", choices=["normal", "t", "skewt"], default="t")
    parser.add_argument(
        "--garch-auto-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select best GARCH order by AIC over a small grid (default: true).",
    )
    return parser.parse_args()


def _resolve_source(arguments: argparse.Namespace) -> str:
    """Resolve the effective price-data source from ``--source`` and friends.

    When ``--source auto`` (the default) the source is inferred from
    which optional arguments are populated::

        --symbols BTC-USD,ETH-USD  -> "yfinance"
        --prices-csv prices.csv    -> "csv"
        otherwise                  -> "synthetic"

    Explicit values of ``--source`` are returned unchanged.

    Args:
        arguments: The parsed :class:`argparse.Namespace`.

    Returns:
        The effective source string -- one of ``"synthetic"``,
        ``"csv"``, or ``"yfinance"``.
    """
    if arguments.source != "auto":
        return str(arguments.source)
    if arguments.symbols:
        return "yfinance"
    if arguments.prices_csv:
        return "csv"
    return "synthetic"


def _load_prices_from_source(arguments: argparse.Namespace, retry_config: RetryConfig) -> pd.DataFrame:
    """Resolve, fetch, and optionally persist the price frame.

    Branches on the effective source:

    * ``yfinance`` -- delegates to :func:`cps.ingestors.fetch_yfinance_prices`
      and (when ``--ingest-output-csv`` is set) writes the fetched frame
      to disk before returning.
    * ``csv`` -- loads the file at ``--prices-csv``.
    * ``synthetic`` -- falls back to :func:`generate_synthetic_prices`.

    Args:
        arguments: The parsed :class:`argparse.Namespace`.
        retry_config: Retry policy applied to the ingestion call.

    Returns:
        ``pd.DataFrame`` of prices indexed by date.

    Raises:
        SystemExit: ``2`` when the source-specific arguments are missing
            or malformed.
    """
    source = _resolve_source(arguments)
    if source == "yfinance":
        from .ingestors import YFinanceIngestorConfig, fetch_yfinance_prices

        if not arguments.symbols:
            print("--symbols is required when --source=yfinance", file=sys.stderr)
            raise SystemExit(2)
        symbols = [s.strip() for s in arguments.symbols.split(",") if s.strip()]
        if not symbols:
            print("--symbols must contain at least one ticker", file=sys.stderr)
            raise SystemExit(2)
        # Build the typed config once so the underlying fetcher sees a
        # consistent payload regardless of which optional fields were
        # provided on the command line.
        config = YFinanceIngestorConfig(
            symbols=tuple(symbols),
            start=arguments.start or None,
            end=arguments.end or None,
            period=arguments.period or None,
            interval=arguments.interval,
            field=arguments.field,
        )
        prices = execute_with_retry(lambda: fetch_yfinance_prices(config), retry_config)
        if arguments.ingest_output_csv:
            # Optional persistence: write the raw fetched frame so the
            # caller can re-run the pipeline against the same data
            # without re-hitting yfinance.
            output_path = Path(arguments.ingest_output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame_to_write = prices.reset_index().rename(columns={prices.index.name or "index": arguments.date_col})
            frame_to_write.to_csv(output_path, index=False)
        return prices
    if source == "csv":
        if not arguments.prices_csv:
            print("--prices-csv is required when --source=csv", file=sys.stderr)
            raise SystemExit(2)
        return execute_with_retry(
            lambda: load_price_data(arguments.prices_csv, date_col=arguments.date_col), retry_config
        )
    return generate_synthetic_prices()


def main() -> None:
    """Entry point for the ``crypto-portfolio`` console script.

    Reads the CLI arguments, sources the price frame, runs the pipeline
    with bounded retries, and writes the canonical artifact set
    (``trades.csv``, ``summary.csv``, ``log_returns.csv``, ``events.jsonl``,
    ``metrics.json``) to ``--output-dir``. The completion marker is
    written to ``--run-dir``.

    Returns:
        ``None``. Prints a short summary to stdout.

    Raises:
        SystemExit: ``2`` for argument-validation failures.
        Exception: Any pipeline / IO error surfaces unchanged so the
            process exits non-zero for orchestration systems.
    """
    arguments = parse_arguments()
    output_directory = Path(arguments.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    logger = StructuredLogger("crypto_portfolio", str(output_directory / "events.jsonl"))
    metrics_registry = MetricsRegistry()

    config = PipelineConfig(
        train_window_days=arguments.train_window_days,
        correlation_window_days=arguments.corr_window_days,
        rebalance_step_days=arguments.rebalance_step_days,
        horizons_days=parse_horizons(arguments.horizons),
        consensus_runs=arguments.consensus_runs,
        majority_threshold=arguments.majority_threshold,
        risk_free_rate_annual=arguments.rf_annual,
        forecast_method=arguments.forecast_method,
        random_seed=arguments.seed,
        weight_cap=arguments.weight_cap,
        max_assets=arguments.max_assets,
        min_assets=arguments.min_assets,
        max_volatility_annual=arguments.max_volatility_annual,
        transaction_cost_bps=arguments.transaction_cost_bps,
        slippage_bps=arguments.slippage_bps,
        lstm_lookback=arguments.lstm_lookback,
        lstm_hidden_size=arguments.lstm_hidden_size,
        lstm_num_layers=arguments.lstm_num_layers,
        lstm_max_epochs=arguments.lstm_max_epochs,
        garch_p=arguments.garch_p,
        garch_o=arguments.garch_o,
        garch_q=arguments.garch_q,
        garch_mean=arguments.garch_mean,
        garch_dist=arguments.garch_dist,
        garch_auto_order=arguments.garch_auto_order,
    )

    run_id = build_run_id(config)
    marker = ensure_idempotent_run(arguments.run_dir, run_id)

    retry_config = RetryConfig(max_attempts=3, initial_backoff_seconds=0.05)
    prices = _load_prices_from_source(arguments, retry_config)

    artifacts = execute_with_retry(lambda: run_pipeline(prices, config, logger, metrics_registry), retry_config)

    trades_frame = pd.DataFrame(
        [
            {
                "strategy": trade.strategy,
                "horizon_days": trade.horizon_days,
                "rebalance_date": trade.rebalance_date,
                "exit_date": trade.exit_date,
                "selected_assets": ",".join(trade.selected_assets),
                "weights": trade.weights,
                "turnover": trade.turnover,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
            }
            for trade in artifacts.trades
        ]
    )
    summary_frame = summaries_to_frame(artifacts.summary)

    trades_path = output_directory / "trades.csv"
    summary_path = output_directory / "summary.csv"
    returns_path = output_directory / "log_returns.csv"
    metrics_path = output_directory / "metrics.json"

    trades_frame.to_csv(trades_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    artifacts.returns.to_csv(returns_path, index=True)
    metrics_path.write_text(
        pd.Series({"counters": metrics_registry.counters, "timings_millis": metrics_registry.timings_millis}).to_json(),
        encoding="utf-8",
    )

    mark_run_complete(marker)
    print(f"Run id: {run_id}")
    print(f"Wrote {len(trades_frame)} trades to {trades_path}")
    print(f"Wrote summary to {summary_path}")


def realtime_main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cps-realtime`` console script.

    Parses ``argv`` (or ``sys.argv[1:]`` when ``argv`` is ``None``),
    builds a :class:`cps.realtime.CCXTPollerConfig`, runs the bounded
    polling loop, and writes the resulting OHLCV rows to ``--output-csv``.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv[1:]`` when
            omitted -- the function is invoked from the console script
            entry point with no arguments.

    Returns:
        ``0`` on success. Exits via :func:`argparse.error` (which raises
        :class:`SystemExit` with code ``2``) when argument validation
        fails.
    """
    from .realtime import CCXTPollerConfig, RetryConfig, run_polling_loop

    parser = argparse.ArgumentParser(prog="cps-realtime", description="Real-time OHLCV poller via ccxt")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument(
        "--symbols", type=str, required=True, help="Comma-separated symbols (e.g. 'BTC/USDT,ETH/USDT')."
    )
    parser.add_argument("--output-csv", type=str, required=True, help="Destination CSV file.")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=1, help="Number of polling iterations to execute.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--initial-backoff", type=float, default=1.0)
    arguments = parser.parse_args(argv)

    symbols = tuple(s.strip() for s in arguments.symbols.split(",") if s.strip())
    if not symbols:
        parser.error("--symbols must contain at least one ticker")

    config = CCXTPollerConfig(
        exchange_id=arguments.exchange,
        symbols=symbols,
        output_csv=Path(arguments.output_csv),
        timeframe=arguments.timeframe,
        interval_seconds=arguments.interval_seconds,
        limit=arguments.limit,
        max_iterations=arguments.max_iterations,
        retry=RetryConfig(
            max_attempts=arguments.max_attempts,
            initial_backoff_seconds=arguments.initial_backoff,
        ),
    )
    completed = run_polling_loop(config, max_iterations=arguments.max_iterations)
    print(f"Completed {completed} polling iterations; wrote {config.output_csv}")
    return 0


if __name__ == "__main__":
    main()
