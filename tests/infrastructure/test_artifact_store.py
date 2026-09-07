"""Tests for the file-system artifact store."""

from __future__ import annotations

import pandas as pd
import pytest

from cps.domain import EvaluationSummary, PortfolioResult, RunArtifacts
from cps.infrastructure.stores import FileArtifactStore


def _sample_artifacts() -> RunArtifacts:
    """Return a small RunArtifacts bundle for round-trip tests."""
    returns = pd.DataFrame(
        {"a": [0.01, -0.02, 0.03], "b": [0.02, -0.01, 0.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    market = returns.mean(axis=1)
    trade = PortfolioResult(
        strategy="baseline",
        horizon_days=3,
        rebalance_date=returns.index[0],
        exit_date=returns.index[2],
        selected_assets=("a", "b"),
        weights={"a": 0.5, "b": 0.5},
        turnover=1.0,
        gross_return=0.01,
        net_return=0.009,
    )
    summary = EvaluationSummary(
        strategy="baseline",
        horizon_days=3,
        average_trade=0.009,
        win_rate=1.0,
        profit_factor=float("inf"),
        var_95=-0.01,
        mes_95=-0.01,
        omega_0=float("inf"),
        trade_count=1,
    )
    import numpy as np

    return RunArtifacts(
        returns=returns,
        market_returns=market,
        trades=(trade,),
        summary=(summary,),
        similarity_matrices={"baseline_h3_t0": np.zeros((2, 2))},
    )


class TestFileArtifactStore:
    def test_round_trip(self, tmp_path):
        store = FileArtifactStore(tmp_path)
        artifacts = _sample_artifacts()
        paths = store.write_run(
            "run_1",
            artifacts,
            metrics={"counters": {"runs": 1}, "timings_millis": {"fit": (1.5,)}},
            events=[{"event": "pipeline_started", "rows": 100, "assets": 2}],
        )
        assert paths.trades_json.exists()
        assert paths.summary_json.exists()
        assert paths.log_returns_csv.exists()
        assert paths.metrics_json.exists()
        assert paths.events_jsonl.exists()
        assert paths.similarity_dir.is_dir()

        trades = store.read_trades("run_1")
        assert trades[0]["strategy"] == "baseline"

        summary = store.read_summary("run_1")
        assert summary[0]["horizon_days"] == 3

        metrics = store.read_metrics("run_1")
        assert metrics["counters"]["runs"] == 1

        log_returns = store.read_log_returns_text("run_1")
        assert "a,b" in log_returns.splitlines()[0]

    def test_run_dir_raises_for_missing(self, tmp_path):
        store = FileArtifactStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.run_dir("does-not-exist")

    def test_write_upload_creates_file(self, tmp_path):
        store = FileArtifactStore(tmp_path)
        path = store.write_upload("run_x", "date,a\n2024-01-01,1\n")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "date,a\n2024-01-01,1\n"

    def test_top_level_layout(self, tmp_path):
        store = FileArtifactStore(tmp_path)
        store.write_run(
            "run_x",
            _sample_artifacts(),
            metrics={"counters": {}, "timings_millis": {}},
            events=[],
        )
        # Run directory must contain exactly the canonical set of files.
        run_dir = tmp_path / "run_x"
        assert (run_dir / "trades.json").is_file()
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "log_returns.csv").is_file()
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "events.jsonl").is_file()
        assert (run_dir / "similarity").is_dir()
