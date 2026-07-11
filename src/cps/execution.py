"""Execution-cost modelling for portfolio net returns.

This module isolates the small arithmetic used to translate gross
per-trade returns into net-of-cost returns. Costs are modelled as two
additive basis-point components:

* ``transaction_cost_bps`` -- the per-side commission (or maker/taker
  fee) charged by the venue.
* ``slippage_bps`` -- the expected price impact when crossing the
  spread. This is applied symmetrically for simplicity; asymmetric
  slippage (separate buy/sell impact) is intentionally not modelled.

Both components are denominated in *basis points* (``1 bp = 0.01%``). The
total cost is scaled by the portfolio's turnover (sum of absolute
weights) so that rebalances with larger position changes pay proportionally
more -- a long-only portfolio with weights summing to 1.0 incurs the full
quoted cost, while a partial rebalance incurs the corresponding fraction.

The cost model is multiplicative on the gross return::

    net = (1 + gross) * (1 - cost_rate) - 1

This matches the convention used by most execution simulators: costs are
charged against the trade *gross* return, not added linearly, so a
``gross = 0%`` trade yields ``net = -cost_rate`` (a small loss equal to
the cost of trading).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Per-trade execution cost configuration.

    Attributes:
        transaction_cost_bps: One-way commission in basis points
            (``1 bp = 0.0001``). Defaults to ``10`` (0.10%).
        slippage_bps: Expected price impact in basis points. Defaults to
            ``5`` (0.05%). Applied symmetrically to buys and sells.
    """

    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0


def compute_total_cost_rate(cost_config: ExecutionCostConfig, turnover: float) -> float:
    """Translate the bps-denominated cost config into a decimal cost rate.

    Args:
        cost_config: The bps-denominated cost configuration.
        turnover: Sum of absolute portfolio weights (``>= 0``). Negative
            turnovers are clamped to ``0`` to defend against accidentally
            flipped signs in caller code.

    Returns:
        The total cost rate as a *decimal* (e.g. ``0.0015`` for ``15 bp``
        of total cost on a fully invested portfolio). Suitable for direct
        multiplication against the gross return.
    """
    # Combine transaction and slippage bps then divide by 10_000 to convert
    # to a decimal. ``max(turnover, 0.0)`` defends against accidental sign
    # flips in caller code -- a negative turnover would otherwise produce a
    # negative cost rate and a perverse *boost* to the net return.
    bps_total = (cost_config.transaction_cost_bps + cost_config.slippage_bps) * max(turnover, 0.0)
    return bps_total / 10000.0


def apply_execution_costs(gross_return: float, cost_rate: float) -> float:
    """Subtract a multiplicative cost from a simple gross return.

    Args:
        gross_return: Trade gross return (simple, not log).
        cost_rate: Decimal cost rate (see :func:`compute_total_cost_rate`).

    Returns:
        Net return after costs, computed as
        ``(1 + gross_return) * (1 - cost_rate) - 1``.
    """
    return (1.0 + gross_return) * (1.0 - cost_rate) - 1.0
