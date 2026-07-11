"""Lazy import guard for optional dependencies.

The ``crypto-portfolio-system`` package installs its hard dependencies
(numpy, pandas, networkx, statsmodels) by default and gates the
heavier optional stacks behind ``pip install
'crypto-portfolio-system[extra]'`` selectors. Every optional feature
-- GARCH forecasting, LSTM forecasting, yfinance ingestion, ccxt
real-time polling, FastAPI -- performs a single runtime check that
raises an actionable error if the underlying package is not
installed.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require_optional(module_name: str, extra: str) -> ModuleType:
    """Import ``module_name`` or raise an actionable install hint.

    Args:
        module_name: The distribution name to import.
        extra: The optional-extras selector that installs this module.

    Returns:
        The imported module object.

    Raises:
        RuntimeError: When ``module_name`` cannot be imported.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"The '{module_name}' package is required for this feature. "
            f"Install the optional extra with: "
            f"pip install 'crypto-portfolio-system[{extra}]'"
        ) from exc


__all__ = ["require_optional"]