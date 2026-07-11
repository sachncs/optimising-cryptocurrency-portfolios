"""Bounded-exponential-backoff retry helper.

Used in three places:

* around the price ingestion call in the CLI;
* around the pipeline execution call in the CLI and the REST API;
* around the ccxt polling call in :mod:`cps.infrastructure.ingestors.ccxt_ingestor`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ...domain.protocols import SleepCallable

ReturnType = TypeVar("ReturnType")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded-exponential-backoff retry policy.

    Attributes:
        max_attempts: Total number of attempts (initial call + retries).
        initial_backoff_seconds: Sleep duration before the first retry.
        backoff_multiplier: Multiplier applied to the sleep duration
            after each failure.
        sleep: Callable invoked with the backoff duration after each
            failed attempt. Production callers use ``time.sleep``;
            tests inject a recording ``sleep`` for deterministic
            scheduling.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    sleep: SleepCallable = field(default=time.sleep)  # type: ignore[assignment]  # SleepCallable is a Protocol satisfying time.sleep's signature.

    def __post_init__(self) -> None:
        """Validate the policy at construction time."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")


def execute_with_retry(
    callable_fn: Callable[[], ReturnType], policy: RetryPolicy
) -> ReturnType:
    """Invoke ``callable_fn`` with bounded-exponential-backoff retries.

    Args:
        callable_fn: Zero-argument callable to invoke.
        policy: Retry policy.

    Returns:
        Whatever ``callable_fn`` returns on the first successful attempt.

    Raises:
        ValueError: When ``policy.max_attempts < 1``.
        Exception: The exception raised by the final failing attempt
            is re-raised unmodified.
    """
    if policy.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempt = 0
    delay = policy.initial_backoff_seconds
    while True:
        try:
            return callable_fn()
        except Exception:
            attempt += 1
            if attempt >= policy.max_attempts:
                raise
            policy.sleep(delay)
            delay *= policy.backoff_multiplier


__all__ = ["RetryPolicy", "execute_with_retry"]