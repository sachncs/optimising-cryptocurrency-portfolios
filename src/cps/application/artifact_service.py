"""Artifact read-back service.

A thin layer over :class:`cps.domain.ArtifactStore` that exposes
typed read methods for the REST API and the CLI ``summary`` command.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..domain.protocols import ArtifactStore


class ArtifactService:
    """Read and forward artifact bundles from a :class:`ArtifactStore`."""

    def __init__(self, store: ArtifactStore) -> None:
        """Initialise with the underlying artifact store."""
        self.__store = store

    @property
    def store(self) -> ArtifactStore:
        """Return the underlying artifact store."""
        return self.__store

    def read_trades(self, run_id: str) -> list[dict[str, object]]:
        """Return the trades records for ``run_id``."""
        return self.__store.read_trades(run_id)

    def read_summary(self, run_id: str) -> list[dict[str, object]]:
        """Return the summary records for ``run_id``."""
        return self.__store.read_summary(run_id)

    def read_metrics(self, run_id: str) -> dict[str, object]:
        """Return the metrics payload for ``run_id``."""
        return self.__store.read_metrics(run_id)

    def read_log_returns(self, run_id: str) -> pd.DataFrame:
        """Return the cleaned log-returns frame for ``run_id``."""
        from io import StringIO

        text = self.__store.read_log_returns_text(run_id)
        return pd.read_csv(StringIO(text), index_col=0, parse_dates=True)

    def run_dir(self, run_id: str) -> Path:
        """Return the directory containing ``run_id``'s artifacts."""
        return self.__store.run_dir(run_id)


__all__ = ["ArtifactService"]
