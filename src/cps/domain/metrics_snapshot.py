"""Metrics snapshot value object.

Re-exported from :mod:`cps.infrastructure.observability.metrics` for
type-level convenience in domain code that needs the snapshot type
without taking a dependency on the infrastructure layer.
"""

from ..infrastructure.observability.metrics import MetricsSnapshot

__all__ = ["MetricsSnapshot"]