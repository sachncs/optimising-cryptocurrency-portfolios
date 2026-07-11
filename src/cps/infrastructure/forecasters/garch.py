"""GARCH(p, o, q) forecaster with optional AIC-based order selection."""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...config.pipeline_config import ForecasterConfig, GARCHForecastConfig
from ...config.settings import GARCH_AUTO_ORDER_CANDIDATES
from ...infrastructure.resilience import require_optional


class GarchForecaster:
    """GARCH forecaster.

    When ``auto_order`` is enabled, fits the user-supplied order plus a
    small candidate grid and selects the model with the lowest AIC.
    """

    name: ClassVar[str] = "garch"

    def __init__(self, config: GARCHForecastConfig | None = None) -> None:
        """Initialise with an optional default config."""
        self.__default_config = config or GARCHForecastConfig()

    def forecast(
        self,
        returns: pd.DataFrame,
        steps: int,
        *,
        config: ForecasterConfig | None = None,
    ) -> pd.DataFrame:
        """Forecast one GARCH model per asset and stack the forecasts."""
        require_optional("arch", "forecast-garch")
        if returns.empty:
            raise ValueError("Train return frame is empty")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        cfg = self._resolve_config(config)
        cols: dict[str, pd.Series] = {}
        for col in returns.columns:
            cols[col] = self._forecast_one(returns[col].astype(float), steps, cfg)
        return pd.DataFrame(cols)

    def _resolve_config(self, forecaster_config: ForecasterConfig | None) -> GARCHForecastConfig:
        """Return the user-provided GARCH config or the default."""
        if forecaster_config is not None and forecaster_config.garch is not None:
            return forecaster_config.garch
        return self.__default_config

    def _forecast_one(
        self, series: pd.Series, steps: int, cfg: GARCHForecastConfig
    ) -> pd.Series:
        """Fit one GARCH model and return its mean path."""

        cleaned = series.dropna()
        if cleaned.empty or cleaned.nunique() < 2:
            raise ValueError("GARCH requires a series with at least 2 distinct values")
        scaled = cleaned * cfg.rescale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_result, best_order = self._select_best_fit(scaled, cfg)
        forecast = best_result.forecast(horizon=steps)
        mean_forecast = np.asarray(forecast.mean).flatten()[-steps:]
        if mean_forecast.size != steps:
            raise RuntimeError(
                f"GARCH forecast returned {mean_forecast.size} steps but {steps} were requested"
            )
        return pd.Series(mean_forecast / cfg.rescale, index=range(steps))

    @staticmethod
    def _select_best_fit(scaled: pd.Series, cfg: GARCHForecastConfig) -> tuple[Any, tuple[int, int, int]]:
        """Fit the user-supplied order plus the auto-grid; pick the lowest AIC."""
        from arch import arch_model  # local import

        try:
            seed_model = arch_model(
                scaled,
                mean=cfg.mean,
                vol="GARCH",
                p=cfg.p,
                o=cfg.o,
                q=cfg.q,
                dist=cfg.dist,
            )
            seed_result = seed_model.fit(disp="off", show_warning=False)
            best_aic = float(seed_result.aic)
            best_order = (cfg.p, cfg.o, cfg.q)
            best_result = seed_result
        except Exception as exc:
            raise RuntimeError(f"GARCH auto-fit failed: {exc}") from exc

        if not cfg.auto_order:
            return best_result, best_order

        for p, o, q in GARCH_AUTO_ORDER_CANDIDATES:
            if (p, o, q) == best_order:
                continue
            try:
                candidate_model = arch_model(
                    scaled, mean=cfg.mean, vol="GARCH", p=p, o=o, q=q, dist=cfg.dist
                )
                candidate_result = candidate_model.fit(disp="off", show_warning=False)
            except Exception:
                continue
            candidate_aic = float(candidate_result.aic)
            if np.isfinite(candidate_aic) and candidate_aic < best_aic:
                best_aic = candidate_aic
                best_order = (p, o, q)
                best_result = candidate_result
        return best_result, best_order


__all__ = ["GarchForecaster"]
