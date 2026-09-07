"""Risk validation service.

Thin wrapper over :class:`cps.domain.policies.RiskLimits` so callers
that already hold a domain reference do not need to know the concrete
type.
"""

from __future__ import annotations

from ..domain.policies import RiskLimits


class RiskService:
    """Operational risk enforcement."""

    def __init__(self, limits: RiskLimits) -> None:
        """Initialise the service with the operational risk limits."""
        self.__limits = limits

    @property
    def limits(self) -> RiskLimits:
        """Return the operational risk limits."""
        return self.__limits

    def effective_weight_cap(self, configured_cap: float, n_assets: int) -> float:
        """Return the per-asset cap actually enforced."""
        from ..domain.policies import compute_effective_weight_cap

        return compute_effective_weight_cap(configured_cap, n_assets)


__all__ = ["RiskService"]
