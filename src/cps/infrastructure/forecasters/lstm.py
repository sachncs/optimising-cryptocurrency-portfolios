"""Shared multi-asset LSTM forecaster.

The forecaster is intentionally *shared*: a single LSTM with
``n_assets`` output heads emits next-step returns for every asset
jointly. The shared representation captures cross-asset non-linearities
that the univariate forecasters cannot.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...config.pipeline_config import ForecasterConfig, LSTMTrainingConfig
from ...infrastructure.resilience import require_optional


class LstmForecaster:
    """Single shared LSTM emitting next-step returns for every asset jointly.

    Two instances do not share state; training each constructs its own
    local NumPy RNG and (optionally) its own ``torch.Generator``. The
    process-global RNG is never touched.
    """

    name: ClassVar[str] = "lstm"

    def __init__(self, n_assets: int, config: LSTMTrainingConfig) -> None:
        """Construct the encoder and projection head."""
        torch = require_optional("torch", "forecast-lstm")
        nn = torch.nn
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        if config.lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.__torch = torch
        self.__config = config
        self.__lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.__head = nn.Linear(config.hidden_size, n_assets)
        self.__normalisation: tuple[np.ndarray, np.ndarray] | None = None
        self.__last_window: np.ndarray | None = None
        self.__columns: tuple[str, ...] | None = None
        self.__fitted = False

    @property
    def config(self) -> LSTMTrainingConfig:
        """Return the training configuration."""
        return self.__config

    def parameters(self) -> tuple[Any, ...]:
        """Return the trainable parameters of the encoder and head."""
        return (*self.__lstm.parameters(), *self.__head.parameters())

    def fit(self, returns: pd.DataFrame) -> LstmForecaster:
        """Fit the LSTM on a returns frame.

        Args:
            returns: ``pd.DataFrame`` of historical returns, one column
                per asset.

        Returns:
            ``self`` for fluent chaining.
        """
        if returns.empty:
            raise ValueError("Cannot train LSTM on empty returns frame")
        if returns.shape[0] <= self.__config.lookback:
            raise RuntimeError(
                f"Need at least lookback+1 rows ({self.__config.lookback + 1}) to train LSTM"
            )

        torch_gen = self.__torch.Generator().manual_seed(self.__config.seed)
        matrix = returns.to_numpy(dtype=float)
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        normalised = (matrix - mean) / std

        windows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for end in range(self.__config.lookback, normalised.shape[0]):
            windows.append(normalised[end - self.__config.lookback : end])
            targets.append(normalised[end])
        if not windows:
            raise ValueError("Not enough rows to build any training windows")

        windows_array = np.stack(windows)
        targets_array = np.stack(targets)

        split_index = max(1, int(len(windows_array) * (1.0 - self.__config.validation_fraction)))
        torch = self.__torch
        x_train = torch.tensor(windows_array[:split_index], dtype=torch.float32)
        y_train = torch.tensor(targets_array[:split_index], dtype=torch.float32)
        if split_index < len(windows_array):
            x_val = torch.tensor(windows_array[split_index:], dtype=torch.float32)
            y_val = torch.tensor(targets_array[split_index:], dtype=torch.float32)
        else:
            x_val = None
            y_val = None

        optimiser = torch.optim.Adam(self.parameters(), lr=self.__config.learning_rate)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        patience_used = 0
        best_state: dict[str, dict[str, np.ndarray]] | None = None

        for _ in range(self.__config.max_epochs):
            self.__lstm.train()
            self.__head.train()
            permutation = torch.randperm(x_train.shape[0], generator=torch_gen)
            for start in range(0, x_train.shape[0], self.__config.batch_size):
                batch_index = permutation[start : start + self.__config.batch_size]
                optimiser.zero_grad()
                lstm_output, _ = self.__lstm(x_train[batch_index])
                prediction = self.__head(lstm_output[:, -1, :])
                loss = loss_fn(prediction, y_train[batch_index])
                loss.backward()
                optimiser.step()

            if x_val is not None and y_val is not None:
                self.__lstm.eval()
                self.__head.eval()
                with torch.no_grad():
                    lstm_output, _ = self.__lstm(x_val)
                    val_prediction = self.__head(lstm_output[:, -1, :])
                    val_loss = float(loss_fn(val_prediction, y_val).item())
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_used = 0
                    best_state = {
                        "lstm": {k: v.detach().numpy().copy() for k, v in self.__lstm.state_dict().items()},
                        "head": {k: v.detach().numpy().copy() for k, v in self.__head.state_dict().items()},
                    }
                else:
                    patience_used += 1
                    if patience_used >= self.__config.patience:
                        break

        if best_state is not None:
            self.__lstm.load_state_dict({k: torch.tensor(v) for k, v in best_state["lstm"].items()})
            self.__head.load_state_dict({k: torch.tensor(v) for k, v in best_state["head"].items()})

        self.__normalisation = (mean.squeeze(0), std.squeeze(0))
        self.__last_window = normalised[-self.__config.lookback :].copy()
        self.__columns = tuple(returns.columns)
        self.__fitted = True
        return self

    def forecast(self, steps: int) -> pd.DataFrame:
        """Generate a recursive multi-step forecast."""
        if not self.__fitted:
            raise RuntimeError("LstmForecaster must be fit() before forecast()")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.__normalisation is None or self.__last_window is None:
            raise RuntimeError("LstmForecaster is in an invalid state: missing normalisation buffers")

        torch = self.__torch
        self.__lstm.eval()
        self.__head.eval()
        window = self.__last_window.copy()
        mean, std = self.__normalisation
        n_assets = window.shape[1]
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(steps):
                tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32)
                lstm_output, _ = self.__lstm(tensor)
                next_step = self.__head(lstm_output[:, -1, :]).numpy().reshape(-1)
                predictions.append(next_step)
                window = np.vstack([window[1:], next_step.reshape(1, n_assets)])

        denormalised = np.stack(predictions) * std + mean
        columns = self.__columns if self.__columns is not None else tuple(f"asset_{i}" for i in range(n_assets))
        return pd.DataFrame(denormalised, columns=list(columns))


class LstmForecasterFactory:
    """Factory that constructs an :class:`LstmForecaster` per request.

    The LSTM needs ``n_assets`` at construction time, which is only
    known once the training returns are available. This factory wraps
    that decision behind the :class:`Forecaster` Protocol so the
    registry can dispatch to the LSTM without exposing the
    construction detail.
    """

    name: ClassVar[str] = "lstm"

    def __init__(self, default_config: LSTMTrainingConfig | None = None) -> None:
        """Initialise the factory with a default config."""
        self.__default_config = default_config or LSTMTrainingConfig()

    def forecast(
        self,
        returns: pd.DataFrame,
        steps: int,
        *,
        config: ForecasterConfig | None = None,
    ) -> pd.DataFrame:
        """Build, fit, and forecast an LSTM on ``returns``."""
        cfg = (config.lstm if config is not None else None) or self.__default_config
        forecaster = LstmForecaster(n_assets=returns.shape[1], config=cfg)
        return forecaster.fit(returns).forecast(steps)


__all__ = ["LstmForecaster", "LstmForecasterFactory"]
