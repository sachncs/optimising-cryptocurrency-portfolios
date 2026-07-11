"""Shared multi-asset LSTM forecaster.

This module is self-contained: it depends on ``torch`` (declared in the
``forecast-lstm`` optional extra). Importing this module without ``torch``
installed raises an actionable error so callers can install the correct
extra. The dataclass :class:`LSTMTrainingConfig` is the only symbol that
does not require ``torch``; importing it lets callers build configs (and
type-annotate code) without paying the torch import cost.

The forecaster is intentionally *shared*: a single LSTM with ``n_assets``
output heads emits next-step returns for every asset jointly. This is
different from training one independent network per asset -- the shared
representation captures cross-asset non-linearities that the naive
univariate forecasters (ARIMA, GARCH) cannot.

Lifecycle::

    model = MultiAssetLSTM(n_assets, config).fit(returns)
    forecasts = model.forecast(steps)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _require_torch() -> None:
    """Lazy guard for the optional ``torch`` dependency.

    Raises:
        RuntimeError: With a message instructing the caller to install
            the ``[forecast-lstm]`` extra.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise RuntimeError(
            "The LSTM forecaster requires the 'torch' package. "
            "Install the optional extra with: pip install 'crypto-portfolio-system[forecast-lstm]'"
        ) from exc


@dataclass(frozen=True)
class LSTMTrainingConfig:
    """Hyper-parameters for the shared multi-asset LSTM.

    Attributes:
        lookback: Number of past observations used to predict the next
            step. Defaults to ``10``.
        hidden_size: Width of the LSTM hidden state. Defaults to ``16``.
        num_layers: Number of stacked LSTM layers. Defaults to ``1``.
        dropout: Inter-layer dropout. Applied only when ``num_layers > 1``.
            Defaults to ``0.0``.
        max_epochs: Maximum number of training epochs. Defaults to ``80``.
        batch_size: Mini-batch size for the Adam optimiser. Defaults to
            ``32``.
        learning_rate: Adam learning rate. Defaults to ``1e-3``.
        patience: Early-stopping patience in epochs (validation loss).
            Defaults to ``10``.
        validation_fraction: Fraction of the training windows reserved
            for validation / early stopping. Defaults to ``0.2``.
        seed: Random seed for ``torch.manual_seed`` and NumPy. Defaults
            to ``42``.
    """

    lookback: int = 10
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    max_epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 10
    validation_fraction: float = 0.2
    seed: int = 42


class MultiAssetLSTM:
    """Single shared LSTM that emits next-step returns for every asset jointly.

    The model wraps a ``torch.nn.LSTM`` encoder and a linear projection
    head of width ``n_assets``. During training, windows of length
    ``config.lookback`` from the standardised return matrix are used to
    predict the next row of the same matrix. During forecasting, the
    model autoregressively rolls forward by feeding its own predictions
    back as inputs.

    The model is *not* thread-safe in the sense that two threads sharing
    the same instance will see interference on the normalisation buffers
    and on the ``_initialised`` flag -- create one instance per training
    run.

    Attributes:
        config: The :class:`LSTMTrainingConfig` driving this instance.
        lstm: The encoder ``torch.nn.LSTM`` (post-construction).
        head: The linear projection ``torch.nn.Linear`` (post-construction).
    """

    def __init__(self, n_assets: int, config: LSTMTrainingConfig) -> None:
        _require_torch()
        import torch.nn as nn

        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        if config.lookback < 1:
            raise ValueError("lookback must be >= 1")

        self.config = config
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_size, n_assets)
        # Normalisation buffers (mean, std) and the trailing window used
        # as the autoregressive seed for forecasting. Populated by ``fit``.
        self._normalisation: tuple[np.ndarray, np.ndarray] | None = None
        self._last_window: np.ndarray | None = None
        self._columns: Sequence[str] | None = None
        self._initialised = False

    def fit(self, returns: pd.DataFrame) -> MultiAssetLSTM:
        """Fit the LSTM on a returns frame.

        Args:
            returns: ``pd.DataFrame`` of historical returns, one column
                per asset, with at least ``lookback + 1`` rows.

        Returns:
            ``self`` -- fluent-style chaining is supported.

        Raises:
            ValueError: When ``returns`` is empty or has fewer than
                ``lookback + 1`` rows.

        Algorithm:
            1. Z-score each column independently. Columns with zero
               standard deviation (constant series) are passed through
               unmodified to avoid division-by-zero.
            2. Build sliding windows of length ``lookback`` and the
               following row as the target.
            3. Split windows into a training and a validation set using
               ``config.validation_fraction``.
            4. Train with Adam + MSE; track the best validation loss and
               snapshot the encoder / head weights at the best epoch.
            5. Restore the best snapshot at the end of training so the
               returned model reflects the lowest validation loss.
        """
        import torch
        import torch.nn as nn

        if returns.empty:
            raise ValueError("Cannot train LSTM on empty returns frame")
        if returns.shape[0] <= self.config.lookback:
            raise ValueError(f"Need at least lookback+1 rows ({self.config.lookback + 1}) to train LSTM")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        matrix = returns.to_numpy(dtype=float)
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        # Guard against zero standard deviation (constant columns): any
        # value below ``1e-12`` is replaced with ``1.0`` so the column is
        # passed through unchanged after normalisation.
        std = np.where(std < 1e-12, 1.0, std)
        normalised = (matrix - mean) / std

        windows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for end in range(self.config.lookback, normalised.shape[0]):
            windows.append(normalised[end - self.config.lookback : end])
            targets.append(normalised[end])
        if not windows:
            raise ValueError("Not enough rows to build any training windows")

        windows_array = np.stack(windows)
        targets_array = np.stack(targets)

        # Train / validation split: keep the first ``(1 - val_frac)`` of
        # windows for training. ``max(1, ...)`` guarantees at least one
        # training window even when the split collapses.
        split_index = max(1, int(len(windows_array) * (1.0 - self.config.validation_fraction)))
        x_train = torch.tensor(windows_array[:split_index], dtype=torch.float32)
        y_train = torch.tensor(targets_array[:split_index], dtype=torch.float32)
        if split_index < len(windows_array):
            x_val = torch.tensor(windows_array[split_index:], dtype=torch.float32)
            y_val = torch.tensor(targets_array[split_index:], dtype=torch.float32)
        else:
            x_val = None
            y_val = None

        parameters = list(self.lstm.parameters()) + list(self.head.parameters())
        optimiser = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        patience_used = 0
        # Snapshot of the encoder / head weights at the best epoch so we
        # can restore them after training finishes.
        best_state: dict[str, dict[str, np.ndarray]] | None = None

        for _epoch in range(self.config.max_epochs):
            self.lstm.train()
            self.head.train()
            permutation = torch.randperm(x_train.shape[0])
            for start in range(0, x_train.shape[0], self.config.batch_size):
                batch_index = permutation[start : start + self.config.batch_size]
                optimiser.zero_grad()
                lstm_output, _ = self.lstm(x_train[batch_index])
                prediction = self.head(lstm_output[:, -1, :])
                loss = loss_fn(prediction, y_train[batch_index])
                loss.backward()
                optimiser.step()

            if x_val is not None and y_val is not None:
                self.lstm.eval()
                self.head.eval()
                with torch.no_grad():
                    lstm_output, _ = self.lstm(x_val)
                    val_prediction = self.head(lstm_output[:, -1, :])
                    val_loss = float(loss_fn(val_prediction, y_val).item())
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_used = 0
                    # Snapshot weights as NumPy arrays so the
                    # ``best_state`` container is cheap to compare /
                    # move between epochs.
                    best_state = {
                        "lstm": {k: v.detach().numpy().copy() for k, v in self.lstm.state_dict().items()},
                        "head": {k: v.detach().numpy().copy() for k, v in self.head.state_dict().items()},
                    }
                else:
                    patience_used += 1
                    if patience_used >= self.config.patience:
                        break

        if best_state is not None:
            # Restore the snapshot from the best epoch. We materialise
            # the NumPy arrays back into ``torch.Tensor`` because
            # ``state_dict`` is type-checked.
            self.lstm.load_state_dict({k: torch.tensor(v) for k, v in best_state["lstm"].items()})
            self.head.load_state_dict({k: torch.tensor(v) for k, v in best_state["head"].items()})

        # Persist the normalisation buffers and the trailing window for
        # the autoregressive forecast.
        self._normalisation = (mean.squeeze(0), std.squeeze(0))
        self._last_window = normalised[-self.config.lookback :].copy()
        self._columns = tuple(returns.columns)
        self._initialised = True
        return self

    def forecast(self, steps: int) -> pd.DataFrame:
        """Generate a recursive multi-step forecast.

        At each step the model consumes the trailing ``lookback`` rows of
        its own history and emits one prediction per asset. The
        prediction is appended to the history and the next step consumes
        the updated window.

        Args:
            steps: Number of forward steps to project.

        Returns:
            ``pd.DataFrame`` of shape ``(steps, n_assets)`` with columns
            matching the training frame. Values are in the original
            (un-standardised) return scale.

        Raises:
            RuntimeError: When called before :meth:`fit`.
            ValueError: When ``steps < 1``.
        """
        import torch

        if not self._initialised:
            raise RuntimeError("LSTM must be fit() before forecast()")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self._normalisation is None or self._last_window is None:
            raise RuntimeError("LSTM is in an invalid state: missing normalisation buffers")

        self.lstm.eval()
        self.head.eval()
        # Copy the trailing window so the loop can mutate its own buffer
        # without disturbing the persisted seed.
        window = self._last_window.copy()
        mean, std = self._normalisation
        n_assets = window.shape[1]
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(steps):
                tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32)
                lstm_output, _ = self.lstm(tensor)
                next_step = self.head(lstm_output[:, -1, :]).numpy().reshape(-1)
                predictions.append(next_step)
                # Slide the window forward by one row, appending the new
                # prediction. The dtype stays float32-friendly.
                window = np.vstack([window[1:], next_step.reshape(1, n_assets)])

        # Undo the per-column z-score so the forecast is in the original
        # return scale the rest of the pipeline expects.
        denormalised = np.stack(predictions) * std + mean
        columns = self._columns if self._columns is not None else tuple(f"asset_{i}" for i in range(n_assets))
        return pd.DataFrame(denormalised, columns=list(columns))


def train_lstm(
    returns: pd.DataFrame,
    config: LSTMTrainingConfig | None = None,
) -> MultiAssetLSTM:
    """Train a shared LSTM on the entire ``returns`` frame and return the model.

    Convenience wrapper around :class:`MultiAssetLSTM` construction +
    :meth:`MultiAssetLSTM.fit`.

    Args:
        returns: ``pd.DataFrame`` of historical returns with at least one
            asset column.
        config: Optional :class:`LSTMTrainingConfig`. Defaults are used
            when omitted.

    Returns:
        A fitted :class:`MultiAssetLSTM`.

    Raises:
        ValueError: When ``returns`` has no asset columns or is too short
            to fit (``lookback + 1`` rows required).
    """
    cfg = config or LSTMTrainingConfig()
    if returns.shape[1] < 1:
        raise ValueError("returns must have at least one asset column")
    return MultiAssetLSTM(n_assets=returns.shape[1], config=cfg).fit(returns)


def lstm_forecast_matrix(
    train_returns: pd.DataFrame,
    steps: int,
    config: LSTMTrainingConfig | None = None,
) -> pd.DataFrame:
    """Train an LSTM on all assets jointly and forecast ``steps`` ahead.

    Args:
        train_returns: ``pd.DataFrame`` of historical returns.
        steps: Number of forward steps to project.
        config: Optional :class:`LSTMTrainingConfig`.

    Returns:
        ``pd.DataFrame`` of shape ``(steps, n_assets)``.

    Raises:
        ValueError: When ``train_returns`` is empty or ``steps < 1``.
    """
    if train_returns.empty:
        raise ValueError("Train return frame is empty")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return train_lstm(train_returns, config).forecast(steps)
