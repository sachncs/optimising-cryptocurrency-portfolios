"""Synthetic price-frame ingestor."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd


class SyntheticIngestor:
    """Generates a 3-factor latent-model price frame for smoke tests / CI."""

    name: ClassVar[str] = "synthetic"

    def __init__(self, days: int = 500, assets: int = 12, seed: int = 7) -> None:
        """Initialise with the requested dimensions and RNG seed."""
        if days < 2:
            raise ValueError("days must be >= 2")
        if assets < 1:
            raise ValueError("assets must be >= 1")
        self.__days = days
        self.__assets = assets
        self.__seed = seed

    def fetch(self) -> pd.DataFrame:
        """Generate the synthetic frame."""
        rng = np.random.default_rng(self.__seed)
        dates = pd.date_range("2020-01-01", periods=self.__days, freq="D")
        factors = rng.normal(0.0005, 0.02, size=(self.__days, 3))
        exposures = rng.normal(0, 1, size=(self.__assets, 3))
        idio = rng.normal(0, 0.015, size=(self.__days, self.__assets))
        returns = factors @ exposures.T + idio
        prices = 100 * np.exp(np.cumsum(returns, axis=0))
        columns = [f"asset_{i:02d}" for i in range(self.__assets)]
        return pd.DataFrame(prices, index=dates, columns=columns)


__all__ = ["SyntheticIngestor"]
