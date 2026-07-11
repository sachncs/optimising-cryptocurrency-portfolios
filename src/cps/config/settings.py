"""Centralized settings and named constants.

Magic numbers previously scattered through ``forecast.py``,
``portfolio.py``, ``risk.py``, ``execution.py``, ``governance.py``, and
``pipeline.py`` live here so the rest of the codebase references named
constants instead of bare values.
"""

from __future__ import annotations

ANNUAL_TRADING_DAYS: int = 365
"""Trading days per year used to annualise volatility and risk-free rates."""

BPS_DENOMINATOR: float = 10000.0
"""Number of basis points per whole unit (1 bp = 0.0001)."""

SHARPE_DEFAULT_MAX_ITERATIONS: int = 2000
"""Default iteration cap for the Sharpe-ratio gradient ascent."""

SHARPE_DEFAULT_LEARNING_STEP: float = 0.05
"""Default learning step for the Sharpe-ratio gradient ascent."""

WEIGHT_CAP_DEFAULT_ITERATIONS: int = 100
"""Default iteration cap for the weight-cap water-filling algorithm."""

LEDOIT_WOLF_VARIANCE_FLOOR: float = 1e-8
"""Lower bound applied to sample variances before Ledoit-Wolf shrinkage."""

LEDOIT_WOLF_DENOMINATOR_FLOOR: float = 1e-12
"""Lower bound used as a denominator inside the Ledoit-Wolf estimator."""

GARCH_DEFAULT_RESCALE: float = 100.0
"""Multiplicative rescale applied to the input series for GARCH stability."""

GARCH_AUTO_ORDER_CANDIDATES: tuple[tuple[int, int, int], ...] = (
    (1, 0, 1),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 0, 1),
)
"""Default GARCH ``(p, o, q)`` candidate grid for AIC-based order selection."""

CCXT_RATE_LIMIT_OPTION: bool = True
"""Whether ccxt exchanges are constructed with ``enableRateLimit=True``."""

CCXT_SUPPORTED_TIMEFRAMES: frozenset[str] = frozenset(
    {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"}
)
"""Timeframes accepted by the ccxt poller."""
