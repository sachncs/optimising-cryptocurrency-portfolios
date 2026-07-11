"""Portfolio construction: covariance regularisation, optimisation, returns.

This module implements the three numerical primitives that turn a
candidate asset selection into an actionable, risk-controlled portfolio:

1. **Covariance regularisation** -- Ledoit-Wolf constant-variance
   shrinkage, which is well-conditioned even when the number of assets
   is comparable to the number of observations (the typical regime for
   crypto backtests).
2. **Sharpe-ratio maximisation** -- a gradient-ascent solver on the
   long-only unit simplex. Returns are projected onto the simplex at
   every step to maintain the long-only constraint without resorting to
   quadratic programming.
3. **Hold-period return accounting** -- compound the per-asset simple
   returns over the holding window and take the dot product with the
   portfolio weights.

References
----------
* Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
  covariance matrices", *Journal of Multivariate Analysis*.
* Held, Wolfe & Crowder (1974), "Validation of subgradient optimization",
  *Mathematical Programming* -- origin of the simplex projection used in
  :func:`project_weights_to_simplex`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ledoit_wolf_constant_variance_covariance(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the Ledoit-Wolf constant-variance shrinkage covariance.

    The estimator blends the sample covariance matrix with the
    constant-variance target ``tr(S) / n * I_n`` using a closed-form
    shrinkage intensity chosen to minimise the expected squared Frobenius
    error.

    Args:
        returns: ``pd.DataFrame`` of returns with one column per asset and
            ``T`` rows (observations). ``T > 1`` is required.

    Returns:
        Symmetric, positive-definite ``pd.DataFrame`` covariance matrix
        indexed and columned by ``returns.columns``.

    Algorithm:
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

    Notes:
        ``phi`` is computed via the standard identity
        ``Var(s_ij) = E[x_ij^2] - (E[x_ij])^2`` applied element-wise; the
        closed form above is algebraically equivalent and runs in ``O(T*n^2)``.
    """
    matrix = returns.to_numpy(dtype=float)
    observations_count, assets_count = matrix.shape
    if assets_count == 1:
        # Single-asset edge case: the sample variance *is* the only thing
        # we can estimate. Clamp to ``1e-8`` to avoid a zero variance when
        # ``T == 1`` (which yields ``np.var(..., ddof=1) == 0``).
        variance = float(np.var(matrix[:, 0], ddof=1)) if observations_count > 1 else 1e-8
        return pd.DataFrame(
            [[max(variance, 1e-8)]],
            index=returns.columns,
            columns=returns.columns,
        )

    sample_covariance = np.cov(matrix, rowvar=False, ddof=1)
    average_variance = np.trace(sample_covariance) / assets_count
    target_covariance = np.eye(assets_count) * average_variance

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    squared = centered**2
    # The closed-form ``phi`` from Ledoit-Wolf: trace of the variance of
    # the outer-product estimator, scaled by ``1 / T^2``.
    phi_matrix = (
        (squared.T @ squared) / observations_count
        - 2 * (centered.T @ centered) * sample_covariance / observations_count
        + sample_covariance**2
    )
    phi = np.sum(phi_matrix)

    gamma = np.linalg.norm(sample_covariance - target_covariance, ord="fro") ** 2
    # When ``S`` happens to equal the constant-variance target the formula
    # collapses to ``0 / 0``; fall back to zero shrinkage so we return the
    # sample matrix unchanged.
    kappa = phi / gamma if gamma > 0 else 0.0
    # Clamp to ``[0, 1]`` for theoretical validity -- the closed-form
    # shrinkage can produce values slightly outside the unit interval for
    # small ``T``.
    shrinkage = max(0.0, min(1.0, kappa / observations_count))
    shrunk_covariance = shrinkage * target_covariance + (1 - shrinkage) * sample_covariance
    return pd.DataFrame(
        shrunk_covariance,
        index=returns.columns,
        columns=returns.columns,
    )


def project_weights_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project an arbitrary weight vector onto the long-only unit simplex.

    The algorithm is the Held-Wolfe-Crowder sort-and-cumsum projection:
    sort the weights in descending order, find the largest ``rho`` such
    that the running cumulative sum minus 1 is less than ``rho + 1``
    times the corresponding sorted weight, then threshold.

    Args:
        weights: 1-D ``np.ndarray`` of unconstrained (possibly negative)
            weights.

    Returns:
        1-D ``np.ndarray`` of non-negative weights summing to ``1``.

    Algorithm:
        Pseudocode (Held, Wolfe, Crowder, 1974)::

            sorted = sort_descending(weights)
            cumsum = cumulative_sum(sorted)
            rho = max { j : sorted[j] * (j + 1) > cumsum[j] - 1 }
            theta = (cumsum[rho] - 1) / (rho + 1)
            return max(weights - theta, 0)

    Complexity: O(n log n) due to the sort; projection itself is O(n).
    """
    if np.isclose(weights.sum(), 1.0) and np.all(weights >= 0):
        # Already on the simplex -- skip the projection to avoid
        # numerical drift for inputs that come straight out of a previous
        # projection.
        return weights
    sorted_weights = np.sort(weights)[::-1]
    cumulative_sum = np.cumsum(sorted_weights)
    # Find the largest ``rho`` such that
    # ``sorted[rho] * (rho + 1) > cumulative_sum[rho] - 1``.
    # The result is the threshold below which the projected weights go
    # to zero.
    rho = np.nonzero(sorted_weights * np.arange(1, len(weights) + 1) > (cumulative_sum - 1))[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    projected = np.maximum(weights - theta, 0)
    return np.asarray(projected, dtype=float)


def optimize_maximum_sharpe_ratio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    daily_risk_free_rate: float,
    max_iterations: int = 2000,
    learning_step: float = 0.05,
) -> pd.Series:
    """Maximise the Sharpe ratio over the long-only unit simplex.

    Uses projected gradient ascent: at every step compute the gradient
    of the Sharpe ratio with respect to the weights, take a fixed-step
    move, and project back onto the simplex via
    :func:`project_weights_to_simplex`. The simplex projection acts as
    a barrier that keeps the search on the feasible region without an
    explicit Lagrangian.

    Args:
        expected_returns: ``pd.Series`` of per-asset expected returns
            (typically the training-window mean).
        covariance: Covariance matrix aligned with ``expected_returns``.
        daily_risk_free_rate: Risk-free rate expressed in *daily*
            units. Use :func:`cps.pipeline.compute_daily_risk_free_rate`
            to convert from an annual rate.
        max_iterations: Number of gradient steps. Defaults to ``2000``.
        learning_step: Fixed step size for the gradient ascent. Defaults
            to ``0.05``.

    Returns:
        ``pd.Series`` of long-only weights summing to ``1``, indexed by
        ``expected_returns.index``.

    Algorithm:
        Pseudocode (projected gradient ascent on the Sharpe ratio)::

            repeat max_iterations times:
                r   = w^T mu                # portfolio expected return
                v   = w^T Sigma w           # portfolio variance
                std = sqrt(v)
                grad = (mu * std - (r - rf) * Sigma * w / std) / v
                w   = project_simplex(w + learning_step * grad)
            return w

    Notes:
        A single-asset input short-circuits the optimiser and returns
        the unit vector -- the gradient is undefined for ``n = 1`` and
        the optimisation is trivial in any case.
    """
    mean_returns = expected_returns.to_numpy(dtype=float)
    covariance_matrix = covariance.to_numpy(dtype=float)
    assets_count = len(mean_returns)
    if assets_count == 1:
        return pd.Series([1.0], index=expected_returns.index)

    weights = np.ones(assets_count, dtype=float) / assets_count
    for iteration_index in range(max_iterations):
        del iteration_index  # Loop variable is intentional; suppress ruff's F841.
        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(weights @ covariance_matrix @ weights)
        # ``max(..., 1e-12)`` defends against the all-equal-weights
        # starting point producing a near-zero variance on a degenerate
        # covariance matrix.
        portfolio_std = np.sqrt(max(portfolio_variance, 1e-12))
        # Closed-form gradient of ``Sharpe(w) = (w^T mu - rf) / std(w)``.
        # Standard calculus of variations; the ``1 / max(variance, eps)``
        # factor absorbs the implicit chain rule from the ``1 / std``
        # denominator.
        gradient = (
            mean_returns * portfolio_std
            - (portfolio_return - daily_risk_free_rate) * (covariance_matrix @ weights) / portfolio_std
        ) / max(portfolio_variance, 1e-12)
        weights = project_weights_to_simplex(weights + learning_step * gradient)
    return pd.Series(weights, index=expected_returns.index)


def compute_portfolio_simple_return(
    future_returns: pd.DataFrame,
    weights: pd.Series,
) -> float:
    """Compute the realised simple return over a holding window.

    For each asset in ``weights``, compound the per-period simple returns
    over the holding window (``(1 + r_1)(1 + r_2)...(1 + r_T) - 1``) and
    then take the dot product with the weights.

    Args:
        future_returns: Per-period log returns of the selected assets over
            the holding window. Index is time; columns are assets.
        weights: Long-only weights summing to ``1``.

    Returns:
        Portfolio simple return over the holding window.
    """
    aligned_returns = future_returns[weights.index]
    compounded_returns = (1.0 + aligned_returns).prod(axis=0) - 1.0
    return float(
        np.dot(
            compounded_returns.to_numpy(dtype=float),
            weights.to_numpy(dtype=float),
        )
    )
