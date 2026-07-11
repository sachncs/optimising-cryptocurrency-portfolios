"""Interface layer tests: CLI and REST API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cps.interface import create_app
from cps.interface.cli import (
    parse_arguments,
    parse_realtime_arguments,
)
from cps.interface.cli.main import main, realtime_main


# ---------- CLI ----------


class TestParseArguments:
    def test_defaults(self):
        args = parse_arguments([])
        assert args.train_window_days == 180
        assert args.horizons == (1, 3, 7, 14)
        assert args.forecast_method == "arima"

    def test_rejects_invalid_horizons(self):
        with pytest.raises(ValueError):
            parse_arguments(["--horizons", ""])
        with pytest.raises(ValueError):
            parse_arguments(["--horizons", "0,1"])


class TestCLIArgsToPipelineConfig:
    def test_round_trip(self):
        args = parse_arguments(
            [
                "--forecast-method",
                "naive",
                "--horizons",
                "1,3",
                "--consensus-runs",
                "3",
                "--weight-cap",
                "0.4",
            ]
        )
        config = args.to_pipeline_config()
        assert config.forecast_method == "naive"
        assert [h.days for h in config.horizons] == [1, 3]
        assert config.consensus_runs == 3
        assert config.weight_cap == 0.4

    def test_rejects_zero_weight_cap(self):
        args = parse_arguments(["--weight-cap", "0"])
        with pytest.raises(ValueError):
            args.to_pipeline_config()


class TestMain:
    def test_main_with_synthetic_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        output_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "crypto-portfolio",
                "--output-dir",
                str(output_dir),
                "--run-dir",
                str(output_dir / "runs"),
                "--forecast-method",
                "naive",
                "--horizons",
                "1",
                "--consensus-runs",
                "3",
            ],
        )
        assert main() == 0
        assert (output_dir / "trades.csv").exists()
        assert (output_dir / "summary.csv").exists()
        assert (output_dir / "log_returns.csv").exists()

    def test_main_with_csv(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prices_csv: Path):
        output_dir = tmp_path / "out_csv"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "crypto-portfolio",
                "--prices-csv",
                str(prices_csv),
                "--date-col",
                "date",
                "--output-dir",
                str(output_dir),
                "--run-dir",
                str(output_dir / "runs"),
                "--train-window-days",
                "5",
                "--corr-window-days",
                "3",
                "--rebalance-step-days",
                "2",
                "--horizons",
                "1",
                "--forecast-method",
                "naive",
                "--consensus-runs",
                "2",
            ],
        )
        assert main() == 0
        assert (output_dir / "summary.csv").exists()

    def test_main_rejects_unknown_forecast_method(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
        output_dir = tmp_path / "out"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "crypto-portfolio",
                "--output-dir",
                str(output_dir),
                "--run-dir",
                str(output_dir / "runs"),
                "--forecast-method",
                "bogus",
            ],
        )
        assert main() == 2
        assert "Unknown forecast method" in capsys.readouterr().err


class TestRealtimeCLI:
    def test_realtime_cli_invokes_poller(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class _StubPoller:
            name = "ccxt"

            def __init__(self, config):
                self.config = config

            def run(self, max_iterations=None):
                return 1

        monkeypatch.setattr("cps.interface.cli.main.CCXTPoller", _StubPoller)
        argv = [
            "--exchange",
            "binance",
            "--symbols",
            "BTC/USDT",
            "--output-csv",
            str(tmp_path / "out.csv"),
            "--max-iterations",
            "1",
            "--interval-seconds",
            "0",
        ]
        assert realtime_main(argv) == 0

    def test_realtime_cli_rejects_empty_symbols(self):
        with pytest.raises(SystemExit):
            realtime_main(["--symbols", "", "--output-csv", "out.csv"])


# ---------- REST API ----------


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from cps.interface import create_app as _create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    app = _create_app(tmp_path / "cps_data")
    return TestClient(app)


class TestRESTHealth:
    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "base_dir" in payload


class TestRESTRunLifecycle:
    def test_create_run_with_inline_prices(self, client, price_rows, base_pipeline_config):
        body = {"config": base_pipeline_config, "prices": price_rows}
        response = client.post("/api/v1/runs", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert "run_id" in payload
        assert "trades_json" in payload["artifact_paths"]

    def test_create_run_rejects_missing_prices(self, client):
        response = client.post("/api/v1/runs", json={"config": {}})
        assert response.status_code == 400

    def test_create_run_unknown_forecast_method(self, client, price_rows, base_pipeline_config):
        body = {
            "config": base_pipeline_config | {"forecast_method": "bogus"},
            "prices": price_rows,
        }
        response = client.post("/api/v1/runs", json=body)
        assert response.status_code == 400

    def test_run_dir_404(self, client):
        response = client.get("/api/v1/runs/does-not-exist")
        assert response.status_code == 404

    def test_full_lifecycle_reads(self, client, price_rows, base_pipeline_config):
        body = {"config": base_pipeline_config, "prices": price_rows}
        run_id = client.post("/api/v1/runs", json=body).json()["run_id"]
        # Summary
        summary = client.get(f"/api/v1/runs/{run_id}/summary").json()
        assert isinstance(summary["summary"], list)
        # Trades (paginated)
        trades = client.get(f"/api/v1/runs/{run_id}/trades?limit=2").json()
        assert "trades" in trades and "total" in trades and "limit" in trades
        # Metrics
        metrics = client.get(f"/api/v1/runs/{run_id}/metrics").json()
        assert "counters" in metrics["metrics"]
        # Log-returns
        returns = client.get(f"/api/v1/runs/{run_id}/log-returns?max_rows=3").json()
        assert returns["total_rows"] >= 1

    def test_create_run_with_csv_content(self, client, prices_csv: Path):
        csv_text = prices_csv.read_text(encoding="utf-8")
        config = {
            "train_window_days": 4,
            "correlation_window_days": 2,
            "rebalance_step_days": 1,
            "consensus_runs": 2,
            "max_volatility_annual": 5.0,
        }
        body = {"config": config, "prices_csv_content": csv_text}
        response = client.post("/api/v1/runs", json=body)
        assert response.status_code == 200

    def test_create_run_with_csv_path(self, tmp_path, client, price_rows):
        csv_path = tmp_path / "prices.csv"
        with csv_path.open("w") as fh:
            for row in price_rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        body = {
            "config": {
                "train_window_days": 5,
                "correlation_window_days": 3,
                "rebalance_step_days": 2,
                "consensus_runs": 2,
                "max_volatility_annual": 5.0,
            },
            "prices_csv_path": str(csv_path),
        }
        response = client.post("/api/v1/runs", json=body)
        assert response.status_code == 200

    def test_invalid_inline_prices_payload(self, client):
        response = client.post("/api/v1/runs", json={"config": {}, "prices": []})
        assert response.status_code == 400

    def test_prices_with_missing_date_column(self, client):
        response = client.post(
            "/api/v1/runs", json={"config": {}, "prices": [["x", "y"], [1, 2]]}
        )
        assert response.status_code == 400


# ---------- Low-level coverage modules ----------


class TestDataModule:
    def test_load_price_data(self, prices_csv: Path):
        from cps.data import load_price_data

        frame = load_price_data(str(prices_csv), date_col="date")
        assert list(frame.columns) == ["a", "b", "c"]
        assert frame.index.name == "date"

    def test_clean_price_data_rejects_non_positive(self):
        from cps.data import DataValidationConfig, clean_price_data

        prices = pd.DataFrame(
            {
                "a": [1.0, 0.0, 1.0, 1.0],
                "b": [1.0, 1.0, 1.0, 1.0],
                "c": [1.0, 1.0, 1.0, 1.0],
                "d": [1.0, 1.0, 1.0, 1.0],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        with pytest.raises(ValueError):
            clean_price_data(prices, DataValidationConfig())

    def test_log_returns_and_market_proxy(self):
        from cps.data import log_returns, market_proxy

        prices = pd.DataFrame(
            {"a": [10.0, 11.0, 12.1, 13.31], "b": [20.0, 19.0, 20.9, 21.945]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        returns = log_returns(prices)
        market = market_proxy(returns)
        assert returns.shape[0] == 3
        assert market.shape[0] == 3


class TestMetricsModule:
    def test_metrics_zero_loss_and_empty_inputs(self):
        import numpy as np

        from cps.metrics import average_trade, omega_ratio, profit_factor, win_rate

        assert profit_factor(np.array([0.1, 0.2])) == float("inf")
        assert omega_ratio(np.array([0.1, 0.2]), threshold=0.0) == float("inf")
        assert average_trade(np.array([])) == 0.0
        assert win_rate(np.array([])) == 0.0

    def test_summarize_strategy(self):
        from cps.metrics import summarize_strategy

        summary = summarize_strategy("baseline", 1, [0.1, -0.1], [0.05, -0.05])
        assert summary.strategy == "baseline"
        assert summary.trade_count == 2


class TestNetworkingModule:
    def test_correlation_distance_diagonal_is_zero(self):
        from cps.networking import correlation_distance_matrix

        returns = pd.DataFrame(
            {"a": [0.01, 0.02, 0.01, 0.0], "b": [0.01, 0.021, 0.009, -0.001]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        distance = correlation_distance_matrix(returns)
        for column in distance.columns:
            assert distance.loc[column, column] == 0.0

    def test_consensus_similarity_default_identity(self):
        from cps.networking import consensus_similarity_matrix

        sim = consensus_similarity_matrix([], ["a", "b"])
        import numpy as np

        np.testing.assert_array_equal(sim, np.eye(2))


class TestRunnerModule:
    def test_run_id_is_deterministic(self):
        from cps.config import PipelineConfig
        from cps.runner import build_run_id

        config = PipelineConfig()
        first = build_run_id(config)
        second = build_run_id(config)
        assert first == second

    def test_run_id_changes_with_field(self):
        from cps.config import PipelineConfig
        from cps.runner import build_run_id

        first = build_run_id(PipelineConfig.with_overrides(seed=1))
        second = build_run_id(PipelineConfig.with_overrides(seed=2))
        assert first != second

    def test_ensure_idempotent_run_rejects_completed(self, tmp_path):
        from cps.runner import ensure_idempotent_run, mark_run_complete

        marker = ensure_idempotent_run(str(tmp_path), "run_1")
        mark_run_complete(marker)
        with pytest.raises(ValueError):
            ensure_idempotent_run(str(tmp_path), "run_1")


class TestPortfolioModule:
    def test_compute_ledoit_wolf_and_simplex(self):
        import numpy as np

        from cps.portfolio import (
            compute_ledoit_wolf_constant_variance_covariance,
            optimize_maximum_sharpe_ratio,
            project_weights_to_simplex,
        )

        returns = pd.DataFrame(
            {"a": [0.01, 0.02, -0.01, 0.015], "b": [0.0, 0.01, 0.02, -0.005]},
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )
        covariance = compute_ledoit_wolf_constant_variance_covariance(returns)
        weights = optimize_maximum_sharpe_ratio(
            returns.mean(), covariance, 0.0, max_iterations=200
        )
        projected = project_weights_to_simplex(np.array([0.6, 0.7]))
        assert abs(weights.values.sum() - 1.0) < 1e-6
        assert projected.sum() == pytest.approx(1.0)

    def test_compute_portfolio_simple_return(self):
        from cps.portfolio import compute_portfolio_simple_return

        weights = pd.Series({"a": 0.5, "b": 0.5})
        future = pd.DataFrame(
            {"a": [0.01, 0.01], "b": [0.02, -0.01]},
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        value = compute_portfolio_simple_return(future, weights)
        assert isinstance(value, float)


class TestExecutionModule:
    def test_execution_costs_reduce_return(self):
        from cps.domain.execution import (
            ExecutionCostConfig,
            apply_execution_costs,
            compute_total_cost_rate,
        )

        cost = compute_total_cost_rate(
            ExecutionCostConfig(transaction_cost_bps=10.0, slippage_bps=5.0),
            turnover=1.0,
        )
        net = apply_execution_costs(0.10, cost)
        assert cost > 0
        assert net < 0.10


class TestConfigSettings:
    def test_settings_present(self):
        from cps.config import settings

        assert hasattr(settings, "BPS_DENOMINATOR")
        assert hasattr(settings, "ANNUAL_TRADING_DAYS")
        assert hasattr(settings, "SHARPE_DEFAULT_MAX_ITERATIONS")