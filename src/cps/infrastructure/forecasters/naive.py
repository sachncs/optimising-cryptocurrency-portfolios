"""Naive last-value forecaster."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from ...config.pipeline_config import ForecasterConfig


class NaiveForecaster:
    """Constant last-value projection: the last training return repeated ``steps`` times."""

    name: ClassVar[str] = "naive"

    def forecast(
        self,
        returns: pd.DataFrame,
        steps: int,
        *,
        config: ForecasterConfig | None = None,
    ) -> pd.DataFrame:
        """Return a frame of shape ``(steps, n_assets)`` where every row equals the last training return.

        Args:
            returns: Historical returns per asset.
            steps: Number of forward steps to project.
            config: Ignored -- the naive forecaster has no parameters.

        Returns:
            ``pd.DataFrame`` of shape ``(steps, n_assets)``.
        """
        if returns.empty:
            raise ValueError("Train return frame is empty")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        last_row = returns.iloc[-1].astype(float)
        forecast = pd.concat([last_row.to_frame().T] * steps, ignore_index=True)
        forecast.index = pd.RangeIndex(steps)
        forecast.columns = returns.columns
        return forecast


__all__ = ["NaiveForecaster"]
