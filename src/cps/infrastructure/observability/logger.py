"""JSON-line event logger with optional on-disk persistence and typed event support.

Each call to :meth:`publish` emits one JSON line per event, forwards the
typed event to every registered listener, and (optionally) appends the
same line to a JSONL file.

Lifecycle of listeners
-----------------------
Listeners are added with :meth:`add_listener` and removed with
:meth:`remove_listener`. The :meth:`close` method releases the file
sink (no-op for the listener list) and is safe to invoke via a
``with`` statement.

The logger no longer clears pre-existing handlers on the named
``logging.Logger``. Instead it composes with whatever handlers the host
application has installed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...domain.events import EventPayload, PipelineEvent


EventListener = Callable[[PipelineEvent, EventPayload], None]


class StructuredLogger:
    """Typed-event JSON-line logger with optional on-disk persistence."""

    def __init__(self, name: str, log_path: str | Path | None = None) -> None:
        """Construct a named structured logger.

        Args:
            name: Logger name. Conventional names in this project are
                ``"crypto_portfolio"`` (CLI) and ``"cps_api"`` (REST).
            log_path: Optional filesystem path. Created (with parents)
                on the first event when provided.
        """
        self.__logger = logging.getLogger(name)
        self.__logger.setLevel(logging.INFO)
        # Always attach our own stream handler so log lines land on stdout
        # regardless of what handlers the host application has already
        # configured. This avoids silently swallowing events when the
        # only pre-existing handler is a NullHandler.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        stream_handler.set_name(f"cps:{name}:stdout")
        self.__logger.addHandler(stream_handler)
        self.__log_path = Path(log_path) if log_path else None
        self.__listeners: list[EventListener] = []


    @property
    def log_path(self) -> Path | None:
        """Return the configured on-disk JSONL sink (``None`` when absent)."""
        return self.__log_path

    def add_listener(self, listener: EventListener) -> None:
        """Register a listener that receives every typed event."""
        self.__listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        """Unregister a previously-added listener (no-op when not registered)."""
        with contextlib.suppress(ValueError):
            self.__listeners.remove(listener)

    def publish(self, event: PipelineEvent, payload: EventPayload) -> None:
        """Emit one typed event.

        Args:
            event: Event kind.
            payload: Typed payload for the event.
        """
        message = {"event": event.value, **asdict(payload)}
        line = json.dumps(message, default=str)
        self.__logger.info(line)
        for listener in self.__listeners:
            listener(event, payload)
        if self.__log_path is not None:
            self.__log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.__log_path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(line + "\n")

    def close(self) -> None:
        """Release the on-disk sink; listeners are not auto-removed."""
        return None

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["EventListener", "StructuredLogger"]