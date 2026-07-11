"""Retry-with-exponential-backoff helper.

This module exposes a single utility, :func:`execute_with_retry`, that wraps
a callable in a bounded retry loop. It is used in three places:

* around the price ingestion call in the CLI (so transient I/O errors
  during CSV loading do not abort the run);
* around the pipeline execution call in the CLI and the REST API (so a
  single bad rebalance does not poison an otherwise healthy run);
* around the ccxt polling call in :mod:`cps.realtime` (so a single dropped
  REST request does not bring the poller down).

Backoff schedule
----------------
After each failure the helper sleeps for ``initial_backoff_seconds`` and
then multiplies the sleep duration by ``backoff_multiplier`` for the next
attempt. With the defaults (``initial=0.1``, ``multiplier=2.0``) the
sleeps are ``0.1, 0.2, 0.4, 0.8, ...`` seconds.

Failure contract
----------------
The helper re-raises whatever exception the callable raised on the last
attempt. It does *not* wrap, log, or swallow exceptions -- that is the
caller's responsibility. This keeps the surface area small and avoids
mismatches between what the caller expects (``requests.exceptions.HTTPError``)
and what the helper returns.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

ReturnType = TypeVar("ReturnType")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for the bounded-exponential-backoff retry loop.

    Attributes:
        max_attempts: Total number of attempts (initial call + retries).
            Must be ``>= 1``; ``1`` disables retries entirely.
        initial_backoff_seconds: Sleep duration before the *first* retry.
            Defaults to ``0.1``.
        backoff_multiplier: Multiplier applied to the sleep duration
            after each failure. Defaults to ``2.0`` for a classic
            exponential schedule (``0.1, 0.2, 0.4, ...``).
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0


def execute_with_retry(callable_fn: Callable[[], ReturnType], config: RetryConfig) -> ReturnType:
    """Invoke ``callable_fn`` with bounded-exponential-backoff retries.

    Args:
        callable_fn: Zero-argument callable to invoke. Typically a
            ``lambda`` or :func:`functools.partial` around the original
            call (this helper does not forward arguments).
        config: Retry policy. ``max_attempts < 1`` raises immediately.

    Returns:
        Whatever ``callable_fn`` returns on the first successful attempt.

    Raises:
        ValueError: When ``config.max_attempts < 1``.
        Exception: The exception raised by the final failing attempt is
            re-raised unmodified so callers can attach context-specific
            handling.
    """
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    attempt = 0
    delay = config.initial_backoff_seconds
    while True:
        try:
            return callable_fn()
        except Exception:
            attempt += 1
            if attempt >= config.max_attempts:
                # No retries left -- surface the last error verbatim so the
                # caller can decide how to handle it (log, re-raise with
                # extra context, fall back to cached data, ...).
                raise
            # Exponential backoff: ``delay`` doubles (or scales by
            # ``backoff_multiplier``) between consecutive retries.
            time.sleep(delay)
            delay *= config.backoff_multiplier
