"""Execution-cost modelling for portfolio net returns.

Cost components are denominated in basis points (``1 bp = 0.0001``):

* ``transaction_cost_bps`` -- one-way commission (or maker/taker fee)
  charged by the venue.
* ``slippage_bps`` -- expected price impact when crossing the
  spread. Applied symmetrically to buys and sells.

The total cost is scaled by turnover (``sum(|weights|)``) so that
rebalances with larger position changes pay proportionally more. A
long-only portfolio with weights summing to 1.0 incurs the full
quoted cost; a partial rebalance incurs the corresponding fraction.

The cost model is multiplicative on the gross return::

    net = (1 + gross) * (1 - cost_rate) - 1
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.settings import BPS_DENOMINATOR


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Per-trade execution cost configuration.

    Attributes:
        transaction_cost_bps: One-way commission in basis points.
        slippage_bps: Expected price impact in basis points.
    """

    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0


def compute_total_cost_rate(cost_config: ExecutionCostConfig, turnover: float) -> float:
    """Translate the bps-denominated cost config into a decimal cost rate.

    Args:
        cost_config: The bps-denominated cost configuration.
        turnover: Sum of absolute portfolio weights.

    Returns:
        The total cost rate as a decimal.
    """
    bps_total = (cost_config.transaction_cost_bps + cost_config.slippage_bps) * max(turnover, 0.0)
    return bps_total / BPS_DENOMINATOR


def apply_execution_costs(gross_return: float, cost_rate: float) -> float:
    """Subtract a multiplicative cost from a simple gross return."""
    return (1.0 + gross_return) * (1.0 - cost_rate) - 1.0


__all__ = ["ExecutionCostConfig", "apply_execution_costs", "compute_total_cost_rate"]