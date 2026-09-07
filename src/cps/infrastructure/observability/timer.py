"""Wall-clock timer based on :py:func:`time.perf_counter`."""

from __future__ import annotations

import time


class Timer:
    """Measure elapsed milliseconds with sub-microsecond precision."""

    def __init__(self) -> None:
        self.__started = time.perf_counter()

    def elapsed_millis(self) -> float:
        """Return elapsed milliseconds since construction."""
        return (time.perf_counter() - self.__started) * 1000.0


__all__ = ["Timer"]
