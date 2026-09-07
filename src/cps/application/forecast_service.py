"""Forecast service: thin dispatcher over the :class:`ForecasterRegistry`."""

from __future__ import annotations

import pandas as pd

from ..config.pipeline_config import ForecasterConfig
from ..domain.protocols import Forecaster
from ..infrastructure.forecasters import ForecasterRegistry, default_registry


class ForecastService:
    """Resolve and invoke a :class:`Forecaster` by name."""

    def __init__(self, registry: ForecasterRegistry | None = None) -> None:
        """Initialise the service with a forecaster registry.

        Args:
            registry: Forecaster registry. Defaults to the built-in one.
        """
        self.__registry = registry if registry is not None else default_registry()

    @property
    def registry(self) -> ForecasterRegistry:
        """Return the underlying forecaster registry."""
        return self.__registry

    def available(self) -> tuple[str, ...]:
        """Return the registered forecaster names."""
        return self.__registry.available()

    def forecast_matrix(
        self,
        train_returns: pd.DataFrame,
        steps: int,
        method: str,
        *,
        config: ForecasterConfig | None = None,
    ) -> pd.DataFrame:
        """Forecast ``steps`` ahead using the named forecaster.

        Args:
            train_returns: ``pd.DataFrame`` of historical returns.
            steps: Forward horizon.
            method: Forecaster name; must be in ``self.available()``.
            config: Optional per-call configuration override.

        Returns:
            ``pd.DataFrame`` of shape ``(steps, n_assets)``.

        Raises:
            KeyError: When ``method`` is not registered.
        """
        forecaster: Forecaster = self.__registry.resolve(method)
        return forecaster.forecast(train_returns, steps, config=config)


__all__ = ["ForecastService"]
