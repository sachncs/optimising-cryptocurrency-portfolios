"""In-process counters and per-call timing samples.

Used by the pipeline, the CLI, and the REST API to track stage
durations and event counts. State is private; callers access the
contents through :meth:`snapshot`, which returns a frozen
:class:`MetricsSnapshot` dataclass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable snapshot of a :class:`MetricsRegistry`."""

    counters: Mapping[str, int]
    timings_millis: Mapping[str, tuple[float, ...]]


@dataclass
class MetricsRegistry:
    """In-process counters and per-call timing samples.

    State is private; the :meth:`snapshot` method is the only public
    read path. Concurrent updates from multiple threads are not
    explicitly synchronised; the registry is intended for the
    single-threaded CLI and the FastAPI worker.
    """

    __counters: dict[str, int] = field(default_factory=dict)
    __timings_millis: dict[str, list[float]] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter by ``amount`` (default ``1``)."""
        self.__counters[name] = self.__counters.get(name, 0) + amount

    def record_timing_millis(self, name: str, elapsed_millis: float) -> None:
        """Append a millisecond timing sample."""
        values = self.__timings_millis.setdefault(name, [])
        values.append(float(elapsed_millis))

    def snapshot(self) -> MetricsSnapshot:
        """Return an immutable :class:`MetricsSnapshot` of the current state."""
        return MetricsSnapshot(
            counters=dict(self.__counters),
            timings_millis={
                name: tuple(samples) for name, samples in self.__timings_millis.items()
            },
        )


__all__ = ["MetricsRegistry", "MetricsSnapshot"]