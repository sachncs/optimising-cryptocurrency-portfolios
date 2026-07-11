"""Observability primitives: counters, timings, structured events.

This module provides the small in-process telemetry surface used by the
pipeline, the CLI, and the REST API. It is intentionally lightweight
(``dataclasses`` and the standard ``logging`` library) so the
observability layer has zero third-party dependencies.

Three pieces:

* :class:`MetricsRegistry` -- thread-safe-by-construction counters and
  per-call timing samples. Stored in process memory only; the CLI dumps
  the contents to ``metrics.json`` at the end of each run.
* :class:`StructuredLogger` -- wraps a named ``logging.Logger`` and
  emits one JSON line per event. Events are also appended to an
  optional on-disk JSONL file (typically ``events.jsonl``).
* :class:`Timer` -- ``time.perf_counter`` wrapper for measuring
  pipeline stages with millisecond precision.

Design note on the structured logger
------------------------------------
:py:meth:`StructuredLogger.__init__` clears any handlers previously
attached to the underlying ``logging.Logger`` before installing a single
``StreamHandler``. This is deliberate: the CLI runs in a single process
and we want the human-readable console output to be owned by *this*
logger, not whatever third-party library happens to have registered
itself on the same logger name. The trade-off is that callers cannot
customise the handler chain after construction -- if they need a
different sink they should construct a new :class:`StructuredLogger`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MetricsRegistry:
    """In-process counters and per-call timing samples.

    Attributes:
        counters: Mapping ``name -> count``. ``MetricsRegistry.increment``
            mutates this dict in place.
        timings_millis: Mapping ``name -> list[float]`` of timing
            samples in milliseconds. ``MetricsRegistry.record_timing_millis``
            appends one entry per call; aggregation (mean, p50, p95)
            is left to the consumer.

    Notes:
        Concurrent updates from multiple threads are not explicitly
        synchronised -- the registry is intended for the single-threaded
        CLI and the FastAPI worker (which is async and runs the pipeline
        in the request coroutine).
    """

    counters: dict[str, int] = field(default_factory=dict)
    timings_millis: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter by ``amount`` (default ``1``).

        Args:
            name: Counter name.
            amount: Increment value. Defaults to ``1``.
        """
        self.counters[name] = self.counters.get(name, 0) + amount

    def record_timing_millis(self, name: str, elapsed_millis: float) -> None:
        """Append a millisecond timing sample.

        Args:
            name: Timing name (typically a pipeline stage).
            elapsed_millis: Elapsed time in milliseconds.
        """
        values = self.timings_millis.setdefault(name, [])
        values.append(float(elapsed_millis))


class StructuredLogger:
    """JSON-line event logger with optional on-disk persistence.

    Each call to :meth:`log_event` emits a single JSON line on the
    underlying ``StreamHandler`` and, when ``log_path`` is provided,
    appends the same line to the JSONL file at that path.

    Attributes:
        logger: The wrapped ``logging.Logger``.
        log_path: The on-disk JSONL sink (``None`` when no file is
            configured).
    """

    def __init__(self, name: str, log_path: str | None = None) -> None:
        """Construct a named structured logger.

        Args:
            name: Logger name. Conventional names in this project are
                ``"crypto_portfolio"`` (CLI) and ``"cps_api"`` (REST).
            log_path: Optional filesystem path. Created (with parents)
                on the first event when provided.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # Reset the handler chain so this logger owns its output. Any
        # third-party library that registered handlers earlier will be
        # silently disconnected -- this is intentional for the CLI's
        # single-process run.
        self.logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(stream_handler)
        self.log_path = Path(log_path) if log_path else None

    def log_event(self, event: str, payload: dict[str, object]) -> None:
        """Emit a single JSON-line event.

        Args:
            event: Event name (e.g. ``"pipeline_started"``).
            payload: Arbitrary JSON-serialisable fields merged into the
                event line. Values that ``json.dumps`` cannot encode
                (NumPy scalars, ``pd.Timestamp``, ...) are coerced via
                the ``default=str`` fallback.
        """
        message = {"event": event, **payload}
        line = json.dumps(message, default=str)
        self.logger.info(line)
        if self.log_path is not None:
            # Persist to disk *after* the console write so a slow disk
            # never delays operator feedback.
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(line + "\n")


class Timer:
    """Wall-clock timer based on :py:func:`time.perf_counter`.

    Use ``Timer.elapsed_millis()`` to read the cumulative elapsed time
    since construction. Re-instantiate the timer to reset it.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed_millis(self) -> float:
        """Return the elapsed time in milliseconds since construction."""
        return (time.perf_counter() - self.started) * 1000.0
