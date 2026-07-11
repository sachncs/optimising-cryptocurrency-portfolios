"""Domain policies.

Encapsulates the operational risk limits and the forecast-drift
detector. Both are pure domain rules that belong with the domain they
govern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .primitives import CovarianceMatrix, Weights


MIN_HISTORY_FOR_DRIFT: int = 10


@dataclass(frozen=True)
class RiskLimits:
    """Operational risk limits applied to every rebalance.

    Attributes:
        max_assets: Upper bound on the number of assets in the
            portfolio.
        min_assets: Lower bound on the number of assets.
        max_weight_per_asset: Maximum weight allowed on any single
            asset.
        max_volatility_annual: Maximum realised portfolio volatility
            (annualised by ``sqrt(365)``).
    """

    max_assets: int = 25
    min_assets: int = 2
    max_weight_per_asset: float = 0.35
    max_volatility_annual: float = 1.2

    def validate(
        self,
        selected: list[str],
        weights: Weights,
        covariance: CovarianceMatrix,
    ) -> None:
        """Raise ``ValueError`` when ``(selected, weights, covariance)`` violates the limits."""
        n = len(selected)
        if n < self.min_assets:
            raise ValueError("Selected assets below minimum risk limit")
        if n > self.max_assets:
            raise ValueError("Selected assets above maximum risk limit")
        effective_cap = compute_effective_weight_cap(self.max_weight_per_asset, n)
        if max(weights.mapping.values()) > effective_cap + 1e-8:
            raise ValueError("Per-asset weight exceeds configured cap")
        cov_df = covariance.to_dataframe().reindex(index=selected, columns=selected)
        w = weights.to_series().reindex(selected)
        annual_volatility = float(
            (w.to_numpy() @ cov_df.to_numpy() @ w.to_numpy()) ** 0.5 * (365.0**0.5)
        )
        if annual_volatility > self.max_volatility_annual:
            raise ValueError("Portfolio annualized volatility exceeds configured maximum")


def compute_effective_weight_cap(configured_cap: float, assets_count: int) -> float:
    """Return ``min(1.0, max(configured_cap, 1 / n))``.

    Args:
        configured_cap: User-configured per-asset cap.
        assets_count: Number of assets in the portfolio.

    Returns:
        Effective cap in ``(0, 1]``.

    Raises:
        ValueError: When ``assets_count <= 0`` or ``configured_cap <= 0``.
    """
    if assets_count <= 0:
        raise ValueError("assets_count must be positive")
    if configured_cap <= 0:
        raise ValueError("configured_cap must be positive")
    return min(1.0, max(configured_cap, 1.0 / assets_count))


def apply_weight_cap(
    weights: Weights,
    cap: float,
    *,
    max_iterations: int = 100,
) -> Weights:
    """Enforce a per-asset cap on a long-only weight vector (water-filling).

    Args:
        weights: Long-only weights summing to ``1``.
        cap: Configured per-asset cap.
        max_iterations: Hard iteration cap. Defaults to ``100``.

    Returns:
        A new :class:`Weights` with the cap enforced.
    """
    from .primitives import Weights as _Weights

    effective_cap = compute_effective_weight_cap(cap, len(weights.mapping))
    weights_dict = dict(weights.mapping)
    total = sum(weights_dict.values())
    if total <= 0:
        return _Weights.equal_weight(list(weights.mapping.keys()))
    weights_dict = {k: v / total for k, v in weights_dict.items()}

    for _ in range(max_iterations):
        over_cap = {k: v for k, v in weights_dict.items() if v > effective_cap}
        if not over_cap:
            break
        excess = sum(v - effective_cap for v in over_cap.values())
        for k in over_cap:
            weights_dict[k] = effective_cap
        under_cap = {k: v for k, v in weights_dict.items() if v < effective_cap}
        room = sum(effective_cap - v for v in under_cap.values())
        if room <= 1e-12:
            break
        for k, v in under_cap.items():
            weights_dict[k] = v + (effective_cap - v) / room * excess
        total = sum(weights_dict.values())
        weights_dict = {k: v / total for k, v in weights_dict.items()}

    return _Weights(weights_dict)


@dataclass
class ForecastGovernance:
    """Rolling MSE recorder with simple drift detection.

    Drift is *latching*: once :meth:`is_drift_detected` returns
    ``True`` it remains ``True`` for the lifetime of this instance
    because subsequent MSE observations can mask the original spike
    when they roll into the baseline window.
    """

    drift_threshold_multiplier: float = 2.0
    mse_history: list[float] = field(default_factory=list)
    drift_detected: bool = False

    def record_error(self, mse_value: float) -> None:
        """Append an MSE observation and re-check for drift."""
        self.mse_history.append(float(mse_value))
        if len(self.mse_history) >= MIN_HISTORY_FOR_DRIFT:
            baseline = float(np.mean(self.mse_history[:-1]))
            latest = self.mse_history[-1]
            if latest > baseline * self.drift_threshold_multiplier:
                self.drift_detected = True

    def is_drift_detected(self) -> bool:
        """Return ``True`` once an MSE observation exceeded the trailing baseline."""
        return self.drift_detected

    def snapshot(self) -> tuple[float, ...]:
        """Return an immutable copy of the recorded history."""
        return tuple(self.mse_history)


__all__ = [
    "MIN_HISTORY_FOR_DRIFT",
    "ForecastGovernance",
    "RiskLimits",
    "apply_weight_cap",
    "compute_effective_weight_cap",
]
