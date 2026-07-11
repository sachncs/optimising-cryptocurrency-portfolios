"""Shared pytest fixtures for the cps test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def prices_csv(tmp_path: Path) -> Path:
    """Write a 20-row, 4-asset prices CSV and return its path."""
    path = tmp_path / "prices.csv"
    lines = ["date,a,b,c"]
    for day in range(20):
        date = f"2024-01-{day + 1:02d}"
        lines.append(f"{date},{100 + day},{200 + 0.5 * day},{300 - 0.25 * day}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def seeded_returns(rows: int = 80, cols: int = 4, seed: int = 7) -> pd.DataFrame:
    """Return a deterministic zero-mean returns frame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(rows, cols)),
        columns=[f"asset_{i}" for i in range(cols)],
    )


@pytest.fixture()
def price_rows() -> list[list[float | str]]:
    """Return a 20-row, 3-asset inline prices payload."""
    rows: list[list[float | str]] = [["date", "btc", "eth", "sol"]]
    for day in range(20):
        rows.append([f"2024-01-{day + 1:02d}", 100.0 + day, 50.0 + 0.5 * day, 10.0 + 0.5 * day])
    return rows


@pytest.fixture()
def base_pipeline_config() -> dict[str, object]:
    """Small test configuration for API and CLI smoke tests."""
    return {
        "train_window_days": 5,
        "correlation_window_days": 3,
        "rebalance_step_days": 2,
        "consensus_runs": 2,
        "max_volatility_annual": 5.0,
    }