"""Portfolio construction service.

Decomposes the inner rebalance loop into discrete steps:

1. Estimate the covariance matrix (Ledoit-Wolf shrinkage).
2. Solve the Sharpe-ratio maximisation (projected gradient ascent).
3. Enforce the per-asset weight cap (iterative water-filling).
4. Validate against the configured :class:`RiskLimits`.
5. Compound the per-period simple returns into a net return.
"""

from __future__ import annotations

import pandas as pd

from ..config.settings import ANNUAL_TRADING_DAYS
from ..domain import (
    CovarianceMatrix,
    ExecutionCostConfig,
    GrossReturn,
    NetReturn,
    Weights,
    compute_ledoit_wolf_constant_variance_covariance,
    compute_portfolio_simple_return,
    compute_total_cost_rate,
    optimize_maximum_sharpe_ratio,
)
from ..domain.policies import RiskLimits, apply_weight_cap


def _realised_turnover(weights: Weights, previous: Weights | None) -> float:
    """Sum of absolute weight changes versus the previous portfolio.

    Falls back to ``1.0`` for the first rebalance (no prior portfolio)
    so the full bps cost is applied to a fresh deployment.
    """
    if previous is None:
        return 1.0
    keys = set(weights.mapping) | set(previous.mapping)
    delta = 0.0
    for asset in keys:
        delta += abs(weights.mapping.get(asset, 0.0) - previous.mapping.get(asset, 0.0))
    return delta


def realised_turnover(weights: Weights, previous: Weights | None) -> float:
    """Public wrapper for :func:`_realised_turnover`."""
    return _realised_turnover(weights, previous)


class PortfolioConstructionError(ValueError):
    """Raised when portfolio construction cannot produce a valid solution."""


class PortfolioService:
    """Build a long-only portfolio from a candidate selection."""

    def __init__(
        self,
        risk_limits: RiskLimits,
        cost_config: ExecutionCostConfig,
        *,
        daily_risk_free_rate: float,
        max_iterations: int,
        learning_step: float,
    ) -> None:
        """Initialise the service with the operational limits and cost config."""
        self.__risk_limits = risk_limits
        self.__cost_config = cost_config
        self.__daily_risk_free_rate = daily_risk_free_rate
        self.__max_iterations = max_iterations
        self.__learning_step = learning_step

    def build(
        self,
        selected_assets: list[str],
        train_returns: pd.DataFrame,
        future_returns: pd.DataFrame,
        previous_weights: Weights | None = None,
    ) -> tuple[Weights, CovarianceMatrix, GrossReturn, NetReturn]:
        """Build the portfolio, validate it, and compute gross + net returns.

        Args:
            selected_assets: Tickers in the candidate portfolio.
            train_returns: Training-window returns for the selected
                assets.
            future_returns: Holding-window returns for the selected
                assets.
            previous_weights: Portfolio held at the previous rebalance.
                When supplied, the execution cost scales with the L1
                change versus the prior portfolio; the first rebalance
                of a run has no previous portfolio and incurs the full
                bps cost.

        Returns:
            ``(weights, covariance, gross_return, net_return)``.

        Raises:
            PortfolioConstructionError: When constraints cannot be
                satisfied.
        """
        if len(selected_assets) < self.__risk_limits.min_assets:
            raise PortfolioConstructionError("Selected assets below minimum risk limit")
        if len(selected_assets) > self.__risk_limits.max_assets:
            raise PortfolioConstructionError("Selected assets above maximum risk limit")

        covariance = CovarianceMatrix.from_dataframe(compute_ledoit_wolf_constant_variance_covariance(train_returns))
        expected_returns = train_returns.mean(axis=0)
        raw_weights = optimize_maximum_sharpe_ratio(
            expected_returns,
            covariance.to_dataframe(),
            self.__daily_risk_free_rate,
            max_iterations=self.__max_iterations,
            learning_step=self.__learning_step,
        )
        weights = apply_weight_cap(Weights.from_series(raw_weights), self.__risk_limits.max_weight_per_asset)
        self.__risk_limits.validate(selected_assets, weights, covariance)

        gross = GrossReturn(compute_portfolio_simple_return(future_returns, raw_weights))
        turnover = _realised_turnover(weights, previous_weights)
        cost_rate = compute_total_cost_rate(self.__cost_config, turnover)
        net = NetReturn.from_gross_and_cost(gross, cost_rate)
        return weights, covariance, gross, net

    @staticmethod
    def _turnover_fraction(weights: Weights, previous: Weights | None) -> float:
        """Public helper for the L1 portfolio-turnover computation."""
        return _realised_turnover(weights, previous)

    @staticmethod
    def annual_volatility(weights: Weights, covariance: CovarianceMatrix) -> float:
        """Annualised portfolio volatility for diagnostics."""
        cov_df = covariance.to_dataframe().reindex(index=weights.assets, columns=weights.assets)
        w = weights.to_series()
        return float((w.to_numpy() @ cov_df.to_numpy() @ w.to_numpy()) ** 0.5 * (ANNUAL_TRADING_DAYS**0.5))


__all__ = ["PortfolioConstructionError", "PortfolioService", "realised_turnover"]
