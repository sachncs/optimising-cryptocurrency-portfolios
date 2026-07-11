"""Resilience layer: retry policy and optional-dependency guard."""

from .optional import require_optional
from .retry import RetryPolicy, execute_with_retry

__all__ = ["RetryPolicy", "execute_with_retry", "require_optional"]
