from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cps.forecast import forecast_matrix  # noqa: E402
from cps.lstm_model import (  # noqa: E402
    LSTMTrainingConfig,
    MultiAssetLSTM,
    lstm_forecast_matrix,
    train_lstm,
)


def _returns(rows: int = 80, cols: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(rows, cols)),
        columns=[f"asset_{i}" for i in range(cols)],
    )


def test_lstm_model_forecast_shape_and_columns():
    frame = _returns()
    config = LSTMTrainingConfig(lookback=5, max_epochs=5, patience=2, seed=11)
    model = train_lstm(frame, config)
    forecast = model.forecast(4)
    assert forecast.shape == (4, frame.shape[1])
    assert list(forecast.columns) == list(frame.columns)


def test_lstm_forecast_matrix_direct():
    frame = _returns(rows=70, cols=3)
    config = LSTMTrainingConfig(lookback=4, max_epochs=3, patience=1, seed=3)
    out = lstm_forecast_matrix(frame, 5, config)
    assert out.shape == (5, 3)


def test_lstm_invalid_lookback_raises():
    with pytest.raises(ValueError):
        MultiAssetLSTM(n_assets=2, config=LSTMTrainingConfig(lookback=0))


def test_lstm_requires_min_rows():
    frame = _returns(rows=5, cols=2)
    config = LSTMTrainingConfig(lookback=10)
    with pytest.raises(ValueError):
        train_lstm(frame, config)


def test_lstm_forecast_before_fit_raises():
    model = MultiAssetLSTM(n_assets=3, config=LSTMTrainingConfig(lookback=4))
    with pytest.raises(RuntimeError):
        model.forecast(3)


def test_forecast_matrix_passes_lstm_config():
    frame = _returns(rows=60, cols=3)
    config = LSTMTrainingConfig(lookback=4, max_epochs=3, patience=1, seed=3)
    out = forecast_matrix(frame, 3, "lstm", lstm_config=config)
    assert out.shape == (3, 3)


def test_forecast_matrix_lstm_default_config():
    frame = _returns(rows=50, cols=2)
    out = forecast_matrix(frame, 2, "lstm")
    assert out.shape == (2, 2)
