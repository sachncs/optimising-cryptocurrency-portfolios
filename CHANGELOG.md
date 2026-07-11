# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Documentation pass**: every module in `src/cps/` now has a Google-style docstring (purpose, responsibilities, design rationale, references); every public class/function/method documents Args, Returns, Raises, and (where useful) Examples; inline comments explain non-obvious algorithmic operators (consensus co-occurrence accumulation, GARCH AIC grid, Ledoit-Wolf shrinkage, Held–Wolfe–Crowder simplex projection, Sharpe gradient, weight-cap water-filling, idempotent run IDs, structured-logger handler reset, ccxt polling rate limit). No behaviour or API changes.

### Added
- **GARCH forecasting** via the `arch` package. New `GARCHForecastConfig` exposes `mean`, `p`, `o`, `q`, `dist`, `rescale`, and `auto_order` (AIC grid search across `(1,0,1)`, `(1,1,1)`, `(2,1,1)`, `(1,1,2)`, `(2,0,1)`). Configurable from the CLI via `--forecast-method garch`, `--garch-p/o/q`, `--garch-mean`, `--garch-dist`, `--garch-auto-order`. Requires the `[forecast-garch]` extra.
- **LSTM forecasting** via `torch` with a shared multi-asset LSTM. Existing module exposed at the CLI via `--forecast-method lstm` and configurable via `--lstm-lookback/hidden-size/num-layers/max-epochs`. Requires the `[forecast-lstm]` extra.
- **yfinance multi-asset ingestor** in `cps.ingestors` (`YFinanceIngestorConfig`, `fetch_yfinance_prices`, `fetch_yfinance_symbols`). CLI flags `--source yfinance --symbols BTC-USD,ETH-USD [--start/--end/--period] [--interval] [--field] [--ingest-output-csv]` materialize a CSV that the rest of the pipeline consumes unchanged. Requires the `[ingestors]` extra.
- **Stateless REST API** in `cps.api`. `create_app(base_dir)` builds a FastAPI app with `GET /api/v1/health`, `POST /api/v1/runs`, `GET /api/v1/runs/{run_id}`, `GET /api/v1/runs/{run_id}/summary`, `GET /api/v1/runs/{run_id}/trades?limit=`, `GET /api/v1/runs/{run_id}/metrics`, and `GET /api/v1/runs/{run_id}/log-returns?max_rows=`. All artifacts are written to the file system under `base_dir`. Requires the `[api]` extra.
- **ccxt real-time ingestor** in `cps.realtime` (`CCXTPollerConfig`, `poll_once`, `run_polling_loop`, `pivot_to_price_frame`). New console script `cps-realtime` runs a bounded polling loop and appends OHLCV candles to a long-form CSV. Requires the `[realtime]` extra.
- Optional dependencies grouped as extras: `forecast-garch`, `forecast-lstm`, `realtime`, `api`, `ingestors`, and `all`.
- Tests covering GARCH config wiring, LSTM dispatch, yfinance ingestor (with mocks), the FastAPI surface, and the ccxt poller.

### Changed
- `PipelineConfig` and `forecast_matrix` now thread `GARCHForecastConfig` and `LSTMTrainingConfig` through so CLI overrides reach the forecaster.
- `cps.cli` auto-detects the price source from `--symbols` / `--prices-csv` when `--source` is omitted, keeping existing CSV-driven flows working.
- `mypy` now ignores missing imports for `arch`, `torch`, `fastapi`, `ccxt`, and `yfinance` (their stubs vary between versions).

## [0.1.0] - 2026-07-11

### Changed
- Bump project version to `0.1.0` (consensus-clustered portfolio framework).
- Bump GitHub Actions `actions/checkout` from `4` to `7` (`47fcf8c`, 2026-07-10T00:14:05Z).
- Bump GitHub Actions `actions/setup-python` from `5` to `6` (`e11ad0b`, 2026-06-20T15:46:46Z).
- Bump GitHub Actions `actions/download-artifact` from `4` to `8` (`d35da87`, 2026-06-20T15:46:48Z).
- Bump GitHub Actions `actions/upload-artifact` from `4` to `7` (`d28ffd2`, 2026-06-20T15:46:56Z).
- Bump GitHub Actions `softprops/action-gh-release` from `2` to `3` (`db00d8b`, 2026-06-20T15:46:54Z).

## [0.0.2] - 2026-06-20

Commit: `5ed3bc6` (2026-06-20T21:16:19+05:30) by sachin <sachncs@gmail.com>.

### Added
- `.editorconfig` and `.gitattributes` for consistent file formatting and line endings.
- `.gitignore` covering Python, virtual environments, IDE, testing, coverage, logs, environment files, output directories, Jupyter, mypy, and ruff caches.
- `.pre-commit-config.yaml` with automated code-quality hooks.
- `.github/dependabot.yml` for weekly `pip` and `github-actions` dependency updates.
- `.github/workflows/ci.yml` with separate `lint`, `typecheck`, and `test` (Python 3.10/3.11/3.12) jobs and a 90% coverage gate.
- `.github/workflows/release.yml` for PyPI publishing via `pypa/gh-action-pypi-publish` and GitHub Release generation via `softprops/action-gh-release`.
- `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` with summary, related issue, changes, testing, and checklist sections.
- `CHANGELOG.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md`, `SECURITY.md`, and `LICENSE` (MIT).
- `Makefile` with `help`, `install`, `dev`, `test`, `test-cov`, `lint`, `lint-fix`, `format`, `typecheck`, `check`, and `clean` targets.
- `docs/deployment.md`, `docs/faq.md`, and `docs/getting-started.md`.
- `tests/test_governance.py` covering the forecast-governance drift checks.
- `pyproject.toml` configuration for ruff linting/formatting, mypy strict type checking, and coverage reporting.
- Expanded `README.md` with feature list, installation, usage, output, project structure, development, tech stack, roadmap, and contributing sections.

### Changed
- `.github/workflows/ci.yml` upgraded with `lint`, `typecheck`, and matrixed test stages plus a coverage gate (`5ed3bc6`).
- `README.md` expanded from initial scaffold to full feature and usage documentation (`5ed3bc6`).
- `pyproject.toml` extended with tool configuration for setuptools, pytest, coverage, mypy, and ruff (`5ed3bc6`).
- `src/cps/cli.py`, `src/cps/governance.py`, `src/cps/metrics.py`, `src/cps/networking.py`, `src/cps/pipeline.py`, `src/cps/portfolio.py`, `src/cps/resilience.py`, and `src/cps/risk.py` refined (`5ed3bc6`).
- `tests/test_cli_and_error_paths.py`, `tests/test_data.py`, and `tests/test_metrics_and_cli.py` adjusted (`5ed3bc6`).

## [0.0.1] - 2026-05-09

Commit: `f7aff47` (2026-05-09T01:55:20+05:30) by sachin <sachncs@gmail.com>.

### Added
- Initial release of the consensus-clustered cryptocurrency portfolio construction framework, based on [arXiv:2505.24831v2](https://arxiv.org/abs/2505.24831v2).
- `src/cps/__init__.py` exposing `PipelineConfig` and `run_pipeline`.
- `src/cps/cli.py` providing the `crypto-portfolio` console script entrypoint.
- `src/cps/pipeline.py` orchestrating ingestion, forecasting, networking, clustering, optimization, risk, execution, and governance.
- `src/cps/data.py` with CSV ingestion and synthetic price generation.
- `src/cps/forecast.py` implementing naive and ARIMA return forecasting.
- `src/cps/networking.py` computing rolling correlation matrices and Louvain community detection.
- `src/cps/portfolio.py` performing Sharpe-ratio portfolio optimization with covariance regularization.
- `src/cps/risk.py` applying asset-count limits, per-asset caps, and annualized volatility ceilings.
- `src/cps/execution.py` modeling transaction costs and slippage on net returns.
- `src/cps/metrics.py` computing downside-risk and profitability metrics.
- `src/cps/governance.py` performing forecast drift detection and MSE tracking.
- `src/cps/observability.py` emitting structured events and metrics.
- `src/cps/resilience.py` providing retry and bounded backoff for critical operations.
- `src/cps/runner.py` enabling idempotent runs with run markers.
- `src/cps/types.py` defining shared data types.
- Initial `tests/` suite: `test_cli_and_error_paths.py`, `test_data.py`, `test_forecast_network_portfolio.py`, `test_metrics_and_cli.py`, `test_pipeline.py`, `test_production_features.py`.
- Initial `docs/`: `api.md`, `architecture.md`, `production-readiness.md`.
- Initial `.github/workflows/ci.yml` scaffold.
- Initial `README.md` and `pyproject.toml`.
- Initial run marker `runs/a598936a8471d6f2.done` (later removed; superseded by `.gitignore` rule `runs/*.done`).