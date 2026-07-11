"""Tests for the forecaster implementations and registry."""

from __future__ import annotations

import pandas as pd
import pytest

from cps.config import GARCHForecastConfig, LSTMTrainingConfig
from cps.infrastructure.forecasters import (
    ArimaForecaster,
    ForecasterRegistry,
    GarchForecaster,
    LstmForecaster,
    LstmForecasterFactory,
    NaiveForecaster,
    default_registry,
)


class TestNaiveForecaster:
    def test_name(self):
        assert NaiveForecaster().name == "naive"

    def test_forecast_shape_and_values(self):
        frame = pd.DataFrame({"a": [0.01, -0.02, 0.03], "b": [0.0, 0.01, 0.02]})
        forecast = NaiveForecaster().forecast(frame, 4)
        assert forecast.shape == (4, 2)
        # Every forecast row equals the last training row.
        last = frame.iloc[-1]
        for column, value in last.items():
            assert (forecast[column] == value).all()

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            NaiveForecaster().forecast(pd.DataFrame(), 3)

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            NaiveForecaster().forecast(pd.DataFrame({"a": [0.1]}), 0)


class TestArimaForecaster:
    def test_name(self):
        assert ArimaForecaster().name == "arima"

    def test_forecast_shape(self):
        frame = pd.DataFrame({"a": [0.01, -0.02, 0.03, 0.04, -0.01, 0.02, 0.05, 0.01]})
        forecast = ArimaForecaster().forecast(frame, 3)
        assert forecast.shape == (3, 1)

    def test_falls_back_to_naive_for_constant_series(self):
        frame = pd.DataFrame({"a": [0.01] * 10})
        forecast = ArimaForecaster().forecast(frame, 4)
        assert forecast.shape == (4, 1)
        assert (forecast["a"] == 0.01).all()


class TestGarchForecaster:
    def test_name(self):
        assert GarchForecaster().name == "garch"

    def test_forecast_shape_with_arch_available(self):
        arch = pytest.importorskip("arch")
        rng = np.random.default_rng(0)
        rows = rng.normal(0, 0.01, size=200)
        frame = pd.DataFrame({"a": rows})
        forecast = GarchForecaster().forecast(frame, 3)
        assert forecast.shape == (3, 1)

    def test_rejects_constant_series(self):
        pytest.importorskip("arch")
        with pytest.raises(ValueError):
            GarchForecaster().forecast(pd.DataFrame({"a": [0.01] * 30}), 3)

    def test_uses_supplied_config(self):
        arch = pytest.importorskip("arch")
        rng = np.random.default_rng(0)
        rows = rng.normal(0, 0.01, size=120)
        frame = pd.DataFrame({"a": rows})
        config = GARCHForecastConfig(dist="normal", auto_order=False)
        forecast = GarchForecaster(config).forecast(frame, 3, config=None)
        assert forecast.shape == (3, 1)


class TestLstmForecaster:
    torch = pytest.importorskip("torch")

    def test_factory_name(self):
        assert LstmForecasterFactory().name == "lstm"

    def test_forecast_shape(self):
        frame = pd.DataFrame(
            np.random.default_rng(7).normal(0, 0.01, size=(80, 3)),
            columns=["a", "b", "c"],
        )
        config = LSTMTrainingConfig(lookback=4, max_epochs=3, patience=1)
        factory = LstmForecasterFactory(config)
        forecast = factory.forecast(frame, 4)
        assert forecast.shape == (4, 3)

    def test_requires_minimum_rows(self):
        config = LSTMTrainingConfig(lookback=20)
        factory = LstmForecasterFactory(config)
        with pytest.raises(RuntimeError):
            factory.forecast(pd.DataFrame(np.zeros((5, 2))), 3)


class TestForecasterRegistry:
    def test_default_registry_has_builtins(self):
        registry = default_registry()
        assert set(registry.available()) == {"naive", "arima", "garch", "lstm"}

    def test_resolve_known_method(self):
        registry = default_registry()
        forecaster = registry.resolve("naive")
        assert isinstance(forecaster, NaiveForecaster)

    def test_resolve_unknown_raises_with_helpful_message(self):
        registry = ForecasterRegistry()
        registry.register(NaiveForecaster())
        with pytest.raises(KeyError, match="Unknown forecast method"):
            registry.resolve("gibberish")

    def test_custom_registration(self):
        registry = ForecasterRegistry()

        class StubForecaster:
            name = "stub"

            def forecast(self, returns, steps, *, config=None):
                return returns.iloc[-steps:]

        registry.register(StubForecaster())
        assert "stub" in registry.available()
        assert registry.resolve("stub") is not None


import numpy as np