"""Portfolio construction: covariance regularisation, optimisation, returns.

Pure numerical primitives used by :class:`cps.application.PortfolioService`.
None of these functions perform I/O; they accept ``pandas`` objects
and return ``pandas`` / ``numpy`` objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import (
    LEDOIT_WOLF_DENOMINATOR_FLOOR,
    LEDOIT_WOLF_VARIANCE_FLOOR,
    SHARPE_DEFAULT_LEARNING_STEP,
    SHARPE_DEFAULT_MAX_ITERATIONS,
)


def compute_ledoit_wolf_constant_variance_covariance(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the Ledoit-Wolf constant-variance shrinkage covariance.

    Pseudocode (Ledoit & Wolf, 2004)::

        n = number of assets
        S = sample covariance of returns
        mu = tr(S) / n
        F = mu * I_n                       # constant-variance target
        phi = sum_{i,j} Var(s_ij) / n^2     # estimation noise
        gamma = ||S - F||_F^2              # distance to target
        kappa = phi / gamma                # (clamped to [0, T])
        shrinkage = clamp(kappa / T, 0, 1)
        return shrinkage * F + (1 - shrinkage) * S
    """
    matrix = returns.to_numpy(dtype=float)
    observations_count, assets_count = matrix.shape
    if assets_count == 1:
        variance = (
            float(np.var(matrix[:, 0], ddof=1))
            if observations_count > 1
            else LEDOIT_WOLF_VARIANCE_FLOOR
        )
        return pd.DataFrame(
            [[max(variance, LEDOIT_WOLF_VARIANCE_FLOOR)]],
            index=returns.columns,
            columns=returns.columns,
        )

    sample_covariance = np.cov(matrix, rowvar=False, ddof=1)
    average_variance = np.trace(sample_covariance) / assets_count
    target_covariance = np.eye(assets_count) * average_variance

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    squared = centered**2
    phi_matrix = (
        (squared.T @ squared) / observations_count
        - 2 * (centered.T @ centered) * sample_covariance / observations_count
        + sample_covariance**2
    )
    phi = np.sum(phi_matrix)

    gamma = np.linalg.norm(sample_covariance - target_covariance, ord="fro") ** 2
    kappa = phi / gamma if gamma > 0 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / observations_count))
    return pd.DataFrame(
        shrinkage * target_covariance + (1 - shrinkage) * sample_covariance,
        index=returns.columns,
        columns=returns.columns,
    )


def project_weights_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project an arbitrary weight vector onto the long-only unit simplex.

    Held-Wolfe-Crowder sort-and-cumsum projection. The input is
    declared "already on the simplex" only when the sum deviates from
    1.0 by less than ``1e-12`` to avoid the early-return masking
    near-simplex updates (typical 0.05-step gradient updates can
    disturb the sum by ~1e-5 which is well within ``np.isclose``'s
    default tolerance but still off the simplex).
    """
    if (
        abs(float(weights.sum()) - 1.0) < 1e-12
        and np.all(weights >= -1e-12)
    ):
        return np.maximum(weights, 0.0)
    sorted_weights = np.sort(weights)[::-1]
    cumulative_sum = np.cumsum(sorted_weights)
    rho = np.nonzero(
        sorted_weights * np.arange(1, len(weights) + 1) > (cumulative_sum - 1)
    )[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    return np.asarray(np.maximum(weights - theta, 0.0))


def optimize_maximum_sharpe_ratio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    daily_risk_free_rate: float,
    max_iterations: int = SHARPE_DEFAULT_MAX_ITERATIONS,
    learning_step: float = SHARPE_DEFAULT_LEARNING_STEP,
) -> pd.Series:
    """Maximise the Sharpe ratio over the long-only unit simplex."""
    mean_returns = expected_returns.to_numpy(dtype=float)
    covariance_matrix = covariance.to_numpy(dtype=float)
    assets_count = len(mean_returns)
    if assets_count == 1:
        return pd.Series([1.0], index=expected_returns.index)

    weights = np.ones(assets_count, dtype=float) / assets_count
    for _ in range(max_iterations):
        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(weights @ covariance_matrix @ weights)
        portfolio_std = np.sqrt(max(portfolio_variance, LEDOIT_WOLF_DENOMINATOR_FLOOR))
        gradient = (
            mean_returns * portfolio_std
            - (portfolio_return - daily_risk_free_rate)
            * (covariance_matrix @ weights)
            / portfolio_std
        ) / max(portfolio_variance, LEDOIT_WOLF_DENOMINATOR_FLOOR)
        weights = project_weights_to_simplex(weights + learning_step * gradient)
    return pd.Series(weights, index=expected_returns.index)


def compute_portfolio_simple_return(
    future_returns: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Compound per-period simple returns over a holding window."""
    aligned_returns = future_returns[weights.index]
    compounded_returns = (1.0 + aligned_returns).prod(axis=0) - 1.0
    return float(
        np.dot(
            compounded_returns.to_numpy(dtype=float),
            weights.to_numpy(dtype=float),
        )
    )


__all__ = [
    "compute_ledoit_wolf_constant_variance_covariance",
    "compute_portfolio_simple_return",
    "optimize_maximum_sharpe_ratio",
    "project_weights_to_simplex",
]
