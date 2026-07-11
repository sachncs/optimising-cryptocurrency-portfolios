"""Portfolio risk constraints.

This module enforces the three operational risk limits applied after the
mean-variance optimiser has produced a candidate weight vector:

1. **Asset-count limits** -- the portfolio must contain between
   ``min_assets`` and ``max_assets`` assets. The pipeline never lets a
   portfolio degenerate to a single asset (degenerate Sharpe) or balloon
   to dozens of near-zero positions (over-fit).
2. **Per-asset cap** -- each weight is bounded above by
   ``max_weight_per_asset``. The cap is *effective*: it is raised to
   ``1/n`` when ``max_weight_per_asset < 1/n`` so that the unit simplex
   can always be reached.
3. **Annualised volatility ceiling** -- the realised volatility of the
   portfolio (computed from the supplied covariance matrix and scaled
   by ``sqrt(365)``) must not exceed ``max_volatility_annual``.

The cap enforcement uses an *iterative water-filling* algorithm that
redistributes excess weight from capped positions to under-cap positions
in a fixed-point loop. This is faster than a quadratic programme and
produces weights that respect the cap on every asset.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    """Operational risk limits applied to every rebalance.

    Attributes:
        max_assets: Upper bound on the number of assets in the portfolio.
            Defaults to ``25``.
        min_assets: Lower bound on the number of assets. Defaults to ``2``.
        max_weight_per_asset: Maximum weight allowed on any single asset,
            as a fraction. Defaults to ``0.35``. The *effective* cap is
            raised to ``1 / n`` when this value is below ``1 / n`` (see
            :func:`compute_effective_weight_cap`).
        max_volatility_annual: Maximum realised portfolio volatility
            (annualised by ``sqrt(365)``). Defaults to ``1.2`` (120%).
    """

    max_assets: int = 25
    min_assets: int = 2
    max_weight_per_asset: float = 0.35
    max_volatility_annual: float = 1.2


def compute_effective_weight_cap(configured_cap: float, assets_count: int) -> float:
    """Compute the effective per-asset cap used by :func:`apply_weight_cap`.

    The effective cap is::

        min(1.0, max(configured_cap, 1 / n))

    The lower bound ``1 / n`` ensures the simplex constraint (weights sum
    to 1) is satisfiable even when the configured cap is below ``1 / n``.
    The upper bound ``1.0`` caps the value when the configured cap is
    absurdly large.

    Args:
        configured_cap: User-configured per-asset cap (e.g. ``0.35``).
        assets_count: Number of assets in the portfolio (``n``).

    Returns:
        The effective cap in ``(0, 1]``.

    Raises:
        ValueError: When ``assets_count <= 0`` or ``configured_cap <= 0``.
    """
    if assets_count <= 0:
        raise ValueError("assets_count must be positive")
    if configured_cap <= 0:
        raise ValueError("configured_cap must be positive")
    return min(1.0, max(configured_cap, 1.0 / assets_count))


def apply_weight_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Enforce a per-asset cap on a long-only weight vector.

    Args:
        weights: Long-only weights summing to ``1`` (the result of the
            Sharpe-ratio optimiser followed by an optional
            :func:`cps.portfolio.project_weights_to_simplex` projection).
        cap: The user-configured per-asset cap. See
            :func:`compute_effective_weight_cap` for the effective cap
            formula.

    Returns:
        A new ``pd.Series`` whose weights lie on the long-only unit
        simplex and respect the effective per-asset cap.

    Algorithm:
        Iterative water-filling in at most 100 iterations:

        1. Clip negative weights to zero and renormalise.
        2. Identify assets exceeding the cap; cap them at ``cap`` and
           record the excess.
        3. Distribute the excess to under-cap assets in proportion to
           their remaining headroom.
        4. Repeat until no asset exceeds the cap or redistribution
           cannot proceed (numerical floor ``1e-12``).

    Complexity: O(n) per iteration, <= 100 iterations.
    """
    effective_cap = compute_effective_weight_cap(cap, len(weights))
    current = weights.clip(lower=0.0)
    if current.sum() <= 0:
        # Degenerate input (all-zero weights): fall back to equal weight
        # rather than letting the loop below divide by zero.
        return pd.Series(1.0 / len(weights), index=weights.index)
    current = current / current.sum()

    for _ in range(100):
        over_cap = current > effective_cap
        if not over_cap.any():
            # Fixed point reached -- no asset exceeds the cap.
            break
        excess = float((current[over_cap] - effective_cap).sum())
        current[over_cap] = effective_cap
        under_cap = current < effective_cap
        room = float((effective_cap - current[under_cap]).sum())
        if room <= 1e-12:
            # No under-cap room to redistribute into; the loop is
            # numerically stuck. This should not happen because the
            # effective cap satisfies ``n * cap >= 1``, but the guard is
            # cheap insurance.
            break
        # Distribute ``excess`` to under-cap assets in proportion to the
        # remaining headroom ``(cap - current)``.
        increment = (effective_cap - current[under_cap]) / room * excess
        current[under_cap] = current[under_cap] + increment
        current = current / current.sum()

    current = current.clip(lower=0.0)
    current = current / current.sum()
    return current


def validate_trade_risk(
    selected_assets: list[str],
    weights: pd.Series,
    covariance: pd.DataFrame,
    limits: RiskLimits,
) -> None:
    """Raise ``ValueError`` when a candidate trade violates the risk limits.

    Args:
        selected_assets: Asset tickers in the candidate portfolio. Length
            must lie in ``[limits.min_assets, limits.max_assets]``.
        weights: Long-only weights summing to ``1``, indexed by the same
            asset tickers.
        covariance: Covariance matrix of the selected assets, indexed and
            columned by the same asset tickers.
        limits: Operational risk limits.

    Raises:
        ValueError: If the asset-count constraint, the per-asset cap, or
            the annualised volatility ceiling is violated.
    """
    if len(selected_assets) < limits.min_assets:
        raise ValueError("Selected assets below minimum risk limit")
    if len(selected_assets) > limits.max_assets:
        raise ValueError("Selected assets above maximum risk limit")
    # Use the *effective* cap so the validation matches what the optimiser
    # and ``apply_weight_cap`` actually enforce.
    effective_cap = compute_effective_weight_cap(limits.max_weight_per_asset, len(selected_assets))
    if float(weights.max()) > effective_cap + 1e-8:
        raise ValueError("Per-asset weight exceeds configured cap")
    # Annualisation factor: the covariance matrix is computed on daily
    # log-returns, so daily variance is scaled by ``365`` and the std by
    # ``sqrt(365)``.
    annual_volatility = float((weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy()) ** 0.5) * (365.0**0.5)
    if annual_volatility > limits.max_volatility_annual:
        raise ValueError("Portfolio annualized volatility exceeds configured maximum")
