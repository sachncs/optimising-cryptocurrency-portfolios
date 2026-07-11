"""Forecaster registry: resolves a forecaster by name from a collection of strategies.

Replaces the previous string-dispatch ``if method == "naive"/"arima"/...``
in :func:`cps.application.forecast_service.forecast_matrix`. Adding a new
forecaster is now a one-line ``registry.register(NewForecaster())`` call.
"""

from __future__ import annotations

from ...domain.protocols import Forecaster
from .arima import ArimaForecaster
from .garch import GarchForecaster
from .lstm import LstmForecasterFactory
from .naive import NaiveForecaster


class ForecasterRegistry:
    """Resolve a :class:`Forecaster` by name."""

    def __init__(self) -> None:
        self.__forecasters: dict[str, Forecaster] = {}

    def register(self, forecaster: Forecaster) -> None:
        """Register a forecaster under its ``name`` class variable."""
        self.__forecasters[forecaster.name] = forecaster

    def resolve(self, name: str) -> Forecaster:
        """Return the registered forecaster for ``name``.

        Raises:
            KeyError: When no forecaster is registered under ``name``.
        """
        try:
            return self.__forecasters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.__forecasters.keys())) or "<none>"
            raise KeyError(
                f"Unknown forecast method {name!r}. Available: {available}."
            ) from exc

    def available(self) -> tuple[str, ...]:
        """Return the registered forecaster names in insertion order."""
        return tuple(self.__forecasters.keys())

    def unregister(self, name: str) -> None:
        """Remove a registered forecaster (no-op when absent)."""
        self.__forecasters.pop(name, None)


def default_registry() -> ForecasterRegistry:
    """Return a registry pre-populated with the built-in forecasters."""
    registry = ForecasterRegistry()
    registry.register(NaiveForecaster())
    registry.register(ArimaForecaster())
    registry.register(GarchForecaster())
    registry.register(LstmForecasterFactory())
    return registry


__all__ = ["ForecasterRegistry", "default_registry"]