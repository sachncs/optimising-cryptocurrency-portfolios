"""ARIMA(p, d, q) forecaster.

Silently falls back to the naive baseline when ``statsmodels`` raises
during fitting or when the training series has fewer than two distinct
values.
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ...domain.protocols import Forecaster, ForecasterConfig
from .naive import NaiveForecaster


class ArimaForecaster:
    """ARIMA forecaster with graceful fallback to the naive baseline."""

    name: ClassVar[str] = "arima"

    def __init__(self, naive_fallback: NaiveForecaster | None = None) -> None:
        """Initialise with an optional pre-built naive fallback instance."""
        self.__fallback = naive_fallback or NaiveForecaster()

    def forecast(
        self,
        returns: pd.DataFrame,
        steps: int,
        *,
        config: ForecasterConfig | None = None,
    ) -> pd.DataFrame:
        """Fit one ARIMA model per asset and stack the forecasts."""
        if returns.empty:
            raise ValueError("Train return frame is empty")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        cols: dict[str, pd.Series] = {}
        for col in returns.columns:
            cols[col] = self._forecast_one(returns[col].astype(float), steps)
        return pd.DataFrame(cols)

    def _forecast_one(self, series: pd.Series, steps: int) -> pd.Series:
        """Forecast one asset; fall back to the naive baseline on any failure."""
        if series.nunique() < 2:
            return self.__fallback.forecast(series.to_frame(), steps).iloc[:, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = ARIMA(series, order=(1, 0, 1)).fit()
                pred = fit.forecast(steps=steps)
                return pd.Series(np.asarray(pred, dtype=float), index=range(steps))
            except Exception:
                return self.__fallback.forecast(series.to_frame(), steps).iloc[:, 0]


__all__ = ["ArimaForecaster"]