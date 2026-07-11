"""CLI entry point for ``crypto-portfolio``.

Two console scripts are registered in ``pyproject.toml``:

* ``crypto-portfolio`` -- :func:`main`, the canonical backtest runner.
* ``cps-realtime`` -- :func:`realtime_main`, a polling daemon.

Exit codes
----------
Both follow the convention:

* ``0`` -- success.
* ``1`` -- the pipeline or poller raised an exception.
* ``2`` -- argument validation failed (``argparse.error``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...domain import EventPayload, PipelineEvent

from ...application import (
    ForecastService,
    build_run_id,
    ensure_idempotent_run,
    mark_run_complete,
    run_pipeline,
)
from ...config import PipelineConfig
from ...config.pipeline_config import (
    ForecasterConfig,
    GARCHForecastConfig,
    Horizon,
    LSTMTrainingConfig,
)
from ...domain.policies import ForecastGovernance
from ...infrastructure.ingestors import (
    CCXTIngestorConfig,
    CCXTPoller,
    CsvIngestor,
    SyntheticIngestor,
    YFinanceConfig,
    YFinanceIngestor,
)
from ...infrastructure.observability import MetricsRegistry, StructuredLogger, Timer
from ...infrastructure.resilience import RetryPolicy, execute_with_retry
from ...infrastructure.stores import FileArtifactStore


@dataclass(frozen=True)
class CLIArgs:
    """Typed representation of the ``crypto-portfolio`` CLI arguments."""

    prices_csv: str
    date_col: str
    output_dir: str
    run_dir: str
    source: str
    symbols: str
    start: str
    end: str
    period: str
    interval: str
    field: str
    ingest_output_csv: str
    train_window_days: int
    correlation_window_days: int
    rebalance_step_days: int
    horizons: tuple[int, ...]
    consensus_runs: int
    majority_threshold: float
    rf_annual: float
    forecast_method: str
    seed: int
    weight_cap: float
    max_assets: int
    min_assets: int
    max_volatility_annual: float
    transaction_cost_bps: float
    slippage_bps: float
    garch_p: int
    garch_o: int
    garch_q: int
    garch_mean: str
    garch_dist: str
    garch_auto_order: bool
    lstm_lookback: int
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_max_epochs: int

    def to_pipeline_config(self) -> PipelineConfig:
        """Build a :class:`PipelineConfig` from this CLI args bundle."""
        forecaster = ForecasterConfig(
            garch=GARCHForecastConfig(
                p=self.garch_p,
                o=self.garch_o,
                q=self.garch_q,
                mean=self.garch_mean,  # type: ignore[arg-type]
                dist=self.garch_dist,  # type: ignore[arg-type]
                auto_order=self.garch_auto_order,
            ),
            lstm=LSTMTrainingConfig(
                lookback=self.lstm_lookback,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                max_epochs=self.lstm_max_epochs,
                seed=self.seed,
            ),
        )
        return PipelineConfig(
            train_window_days=self.train_window_days,
            correlation_window_days=self.correlation_window_days,
            rebalance_step_days=self.rebalance_step_days,
            horizons=tuple(Horizon(days) for days in self.horizons),
            consensus_runs=self.consensus_runs,
            majority_threshold=self.majority_threshold,
            risk_free_rate_annual=self.rf_annual,
            forecast_method=self.forecast_method,
            random_seed=self.seed,
            weight_cap=self.weight_cap,
            max_assets=self.max_assets,
            min_assets=self.min_assets,
            max_volatility_annual=self.max_volatility_annual,
            transaction_cost_bps=self.transaction_cost_bps,
            slippage_bps=self.slippage_bps,
            forecaster=forecaster,
        )


def _add_cli_arguments(parser: argparse.ArgumentParser, available_forecast_methods: Sequence[str]) -> None:
    """Register the canonical ``crypto-portfolio`` flag set on ``parser``."""
    parser.add_argument("--prices-csv", type=str, default="")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument(
        "--source",
        choices=["auto", "synthetic", "csv", "yfinance"],
        default="auto",
    )
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--period", type=str, default="")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument(
        "--field",
        choices=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
        default="Close",
    )
    parser.add_argument("--ingest-output-csv", type=str, default="")
    parser.add_argument("--train-window-days", type=int, default=180)
    parser.add_argument("--corr-window-days", type=int, default=60)
    parser.add_argument("--rebalance-step-days", type=int, default=30)
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,3,7,14",
        help="Comma-separated positive integers.",
    )
    parser.add_argument("--consensus-runs", type=int, default=20)
    parser.add_argument("--majority-threshold", type=float, default=0.5)
    parser.add_argument("--rf-annual", type=float, default=0.045)
    parser.add_argument("--forecast-method", type=str, default="arima")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-cap", type=float, default=0.35)
    parser.add_argument("--max-assets", type=int, default=25)
    parser.add_argument("--min-assets", type=int, default=2)
    parser.add_argument("--max-volatility-annual", type=float, default=1.2)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--garch-p", type=int, default=1)
    parser.add_argument("--garch-o", type=int, default=1)
    parser.add_argument("--garch-q", type=int, default=1)
    parser.add_argument("--garch-mean", choices=["Zero", "Constant", "AR"], default="Zero")
    parser.add_argument("--garch-dist", choices=["normal", "t", "skewt"], default="t")
    parser.add_argument(
        "--garch-auto-order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lstm-lookback", type=int, default=10)
    parser.add_argument("--lstm-hidden-size", type=int, default=16)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--lstm-max-epochs", type=int, default=80)


def _parse_horizons(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of horizons; raise ``ValueError`` on bad input."""
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one horizon value is required")
    if any(v <= 0 for v in values):
        raise ValueError("All horizon values must be positive integers")
    return tuple(values)


def parse_arguments(
    argv: Sequence[str] | None = None, *, available_forecast_methods: Sequence[str] = ("naive", "arima", "garch", "lstm")
) -> CLIArgs:
    """Parse the CLI args into a typed :class:`CLIArgs`."""
    parser = argparse.ArgumentParser(description="Consensus-clustered crypto portfolio system")
    _add_cli_arguments(parser, available_forecast_methods)
    namespace = parser.parse_args(argv)
    return CLIArgs(
        prices_csv=namespace.prices_csv,
        date_col=namespace.date_col,
        output_dir=namespace.output_dir,
        run_dir=namespace.run_dir,
        source=namespace.source,
        symbols=namespace.symbols,
        start=namespace.start,
        end=namespace.end,
        period=namespace.period,
        interval=namespace.interval,
        field=namespace.field,
        ingest_output_csv=namespace.ingest_output_csv,
        train_window_days=namespace.train_window_days,
        correlation_window_days=namespace.corr_window_days,
        rebalance_step_days=namespace.rebalance_step_days,
        horizons=_parse_horizons(namespace.horizons),
        consensus_runs=namespace.consensus_runs,
        majority_threshold=namespace.majority_threshold,
        rf_annual=namespace.rf_annual,
        forecast_method=namespace.forecast_method,
        seed=namespace.seed,
        weight_cap=namespace.weight_cap,
        max_assets=namespace.max_assets,
        min_assets=namespace.min_assets,
        max_volatility_annual=namespace.max_volatility_annual,
        transaction_cost_bps=namespace.transaction_cost_bps,
        slippage_bps=namespace.slippage_bps,
        garch_p=namespace.garch_p,
        garch_o=namespace.garch_o,
        garch_q=namespace.garch_q,
        garch_mean=namespace.garch_mean,
        garch_dist=namespace.garch_dist,
        garch_auto_order=namespace.garch_auto_order,
        lstm_lookback=namespace.lstm_lookback,
        lstm_hidden_size=namespace.lstm_hidden_size,
        lstm_num_layers=namespace.lstm_num_layers,
        lstm_max_epochs=namespace.lstm_max_epochs,
    )


def _resolve_source(args: CLIArgs) -> str:
    """Resolve the effective price-data source from the CLI flags."""
    if args.source != "auto":
        return args.source
    if args.symbols:
        return "yfinance"
    if args.prices_csv:
        return "csv"
    return "synthetic"


def _build_ingestor(args: CLIArgs) -> "YFinanceIngestor | CsvIngestor | SyntheticIngestor":
    """Construct the Ingestor Protocol implementation for the resolved source."""
    source = _resolve_source(args)
    if source == "yfinance":
        if not args.symbols:
            print("--symbols is required when --source=yfinance", file=sys.stderr)
            raise SystemExit(2)
        symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
        if not symbols:
            print("--symbols must contain at least one ticker", file=sys.stderr)
            raise SystemExit(2)
        config = YFinanceConfig(
            symbols=symbols,
            start=args.start or None,
            end=args.end or None,
            period=args.period or None,
            interval=args.interval,  # type: ignore[arg-type]
            field=args.field,  # type: ignore[arg-type]
        )
        return YFinanceIngestor(config)
    if source == "csv":
        if not args.prices_csv:
            print("--prices-csv is required when --source=csv", file=sys.stderr)
            raise SystemExit(2)
        return CsvIngestor(args.prices_csv, date_col=args.date_col)
    return SyntheticIngestor()


def _persist_ingested_frame(prices: pd.DataFrame, output_csv: str, date_col: str) -> None:
    """Optionally persist the ingested frame to disk."""
    if not output_csv:
        return
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = prices.reset_index().rename(columns={prices.index.name or "index": date_col})
    frame.to_csv(output_path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``crypto-portfolio`` console script."""
    args = parse_arguments(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = StructuredLogger("crypto_portfolio", output_dir / "events.jsonl")
    metrics_registry = MetricsRegistry()
    artifact_store = FileArtifactStore(output_dir)
    governance = ForecastGovernance()
    forecast_service = ForecastService()
    available = forecast_service.available()
    if args.forecast_method not in available:
        print(
            f"Unknown forecast method {args.forecast_method!r}; available: {', '.join(available)}.",
            file=sys.stderr,
        )
        return 2

    config = args.to_pipeline_config()
    run_id = build_run_id(config)
    ensure_idempotent_run(args.run_dir, run_id)

    retry_policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.05)
    ingestor = _build_ingestor(args)
    prices = execute_with_retry(ingestor.fetch, retry_policy)
    _persist_ingested_frame(prices, args.ingest_output_csv, args.date_col)

    captured_events: list = []
    capture_listener = _capture_events(captured_events)
    logger.add_listener(capture_listener)

    timer = Timer()
    result = run_pipeline(
        prices,
        config,
        artifact_store=artifact_store,
        logger=logger,
        metrics_registry=metrics_registry,
        governance=governance,
        forecast_service=forecast_service,
    )
    metrics_registry.record_timing_millis("pipeline_duration_millis", timer.elapsed_millis())

    artifact_store.write_run(  # noqa: F841 (artifact_paths canonical layout; persisted by write_run)
        run_id,
        result.artifacts,
        metrics=asdict_payload(metrics_registry.snapshot()),
        events=captured_events,
    )

    trades_frame = pd.DataFrame(
        [
            {
                "strategy": trade.strategy,
                "horizon_days": trade.horizon_days,
                "rebalance_date": trade.rebalance_date,
                "exit_date": trade.exit_date,
                "selected_assets": ",".join(trade.selected_assets),
                "weights": dict(trade.weights),
                "turnover": trade.turnover,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
            }
            for trade in result.trades
        ]
    )
    summary_frame = pd.DataFrame([s.__dict__ for s in result.summaries])

    trades_path = output_dir / "trades.csv"
    summary_path = output_dir / "summary.csv"
    returns_path = output_dir / "log_returns.csv"
    metrics_path = output_dir / "metrics.json"

    trades_frame.to_csv(trades_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    result.artifacts.returns.to_csv(returns_path, index=True)
    metrics_path.write_text(
        pd.Series(
            {
                "counters": dict(metrics_registry.snapshot().counters),
                "timings_millis": dict(metrics_registry.snapshot().timings_millis),
            }
        ).to_json(),
        encoding="utf-8",
    )

    mark_run_complete(Path(args.run_dir) / f"{run_id}.done")
    print(f"Run id: {run_id}")
    print(f"Wrote {len(trades_frame)} trades to {trades_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


def _capture_events(sink: list) -> Callable[[PipelineEvent, EventPayload], None]:
    """Build a listener that appends ``(event, payload)`` pairs to ``sink``."""

    def listener(event: PipelineEvent, payload: EventPayload) -> None:
        sink.append({"event": event.value, **payload.__dict__})

    return listener


def asdict_payload(snapshot: "MetricsSnapshot") -> dict[str, object]:  # type: ignore[name-defined]  # MetricsSnapshot re-exported from cps.infrastructure.observability.metrics
    """Convert a :class:`MetricsSnapshot` to a JSON-ready dict."""
    from dataclasses import asdict

    return dict[str, object](asdict(snapshot))


# ----------------------------------------------------------------------
# ``cps-realtime`` console script
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RealtimeCLIArgs:
    """Typed representation of the ``cps-realtime`` CLI arguments."""

    exchange: str
    symbols: str
    output_csv: str
    timeframe: str
    interval_seconds: float
    limit: int
    max_iterations: int
    max_attempts: int
    initial_backoff: float

    @property
    def symbols_tuple(self) -> tuple[str, ...]:
        """Parse the symbols string into a tuple of stripped tickers."""
        symbols = tuple(s.strip() for s in self.symbols.split(",") if s.strip())
        if not symbols:
            raise ValueError("--symbols must contain at least one ticker")
        return symbols


def parse_realtime_arguments(argv: Sequence[str] | None = None) -> RealtimeCLIArgs:
    """Parse the ``cps-realtime`` CLI arguments."""
    parser = argparse.ArgumentParser(prog="cps-realtime", description="Real-time OHLCV poller via ccxt")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--symbols", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--initial-backoff", type=float, default=1.0)
    namespace = parser.parse_args(argv)
    if not any(symbol.strip() for symbol in namespace.symbols.split(",")):
        parser.error("--symbols must contain at least one ticker")
    return RealtimeCLIArgs(
        exchange=namespace.exchange,
        symbols=namespace.symbols,
        output_csv=namespace.output_csv,
        timeframe=namespace.timeframe,
        interval_seconds=namespace.interval_seconds,
        limit=namespace.limit,
        max_iterations=namespace.max_iterations,
        max_attempts=namespace.max_attempts,
        initial_backoff=namespace.initial_backoff,
    )


def realtime_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``cps-realtime`` console script."""
    from ...infrastructure.resilience import RetryPolicy as RealtimeRetryPolicy

    args = parse_realtime_arguments(argv)
    config = CCXTIngestorConfig(
        exchange_id=args.exchange,
        symbols=args.symbols_tuple,
        output_csv=Path(args.output_csv),
        timeframe=args.timeframe,
        interval_seconds=args.interval_seconds,
        limit=args.limit,
        max_iterations=args.max_iterations,
        retry=RealtimeRetryPolicy(
            max_attempts=args.max_attempts,
            initial_backoff_seconds=args.initial_backoff,
        ),
    )
    poller = CCXTPoller(config)
    completed = poller.run(max_iterations=args.max_iterations)
    print(f"Completed {completed} polling iterations; wrote {config.output_csv}")
    return 0


__all__ = [
    "CLIArgs",
    "RealtimeCLIArgs",
    "main",
    "parse_arguments",
    "parse_realtime_arguments",
    "realtime_main",
]


if __name__ == "__main__":
    raise SystemExit(main())
