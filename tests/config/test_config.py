"""Config layer tests."""

from __future__ import annotations

import pytest

from cps.config import (
    ForecasterConfig,
    GARCHForecastConfig,
    LSTMTrainingConfig,
    PipelineConfig,
    default_strategy_specs,
)
from cps.domain import Horizon


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.train_window_days == 180
        assert config.consensus_runs == 20
        assert config.forecast_method == "arima"
        assert config.horizons == (Horizon(1), Horizon(3), Horizon(7), Horizon(14))

    def test_rejects_empty_horizons(self):
        with pytest.raises(ValueError):
            PipelineConfig(horizons=())

    def test_rejects_zero_consensus_runs(self):
        with pytest.raises(ValueError):
            PipelineConfig(consensus_runs=0)

    def test_rejects_invalid_majority_threshold(self):
        with pytest.raises(ValueError):
            PipelineConfig(majority_threshold=0.0)
        with pytest.raises(ValueError):
            PipelineConfig(majority_threshold=2.0)

    def test_rejects_non_positive_weight_cap(self):
        with pytest.raises(ValueError):
            PipelineConfig(weight_cap=0.0)


class TestForecasterConfig:
    def test_defaults(self):
        config = ForecasterConfig()
        assert isinstance(config.garch, GARCHForecastConfig)
        assert isinstance(config.lstm, LSTMTrainingConfig)

    def test_nested_override(self):
        lstm = LSTMTrainingConfig(lookback=20, hidden_size=64)
        config = ForecasterConfig(lstm=lstm)
        assert config.lstm.lookback == 20


class TestStrategySpecs:
    def test_default_strategy_specs_returns_four(self):
        specs = default_strategy_specs()
        assert len(specs) == 4
        assert {spec.name for spec in specs} == {"baseline", "s", "p", "p-s"}
        flags = {(spec.name, spec.use_prediction, spec.use_shifts) for spec in specs}
        assert ("baseline", False, False) in flags
        assert ("p-s", True, True) in flags
