"""File-system backed :class:`cps.domain.ArtifactStore` implementation."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ...domain import EvaluationSummary, PortfolioResult, RunArtifacts
from ...domain.protocols import RunPaths


class FileArtifactStore:
    """Persists a :class:`RunArtifacts` bundle to the local file system.

    The same layout is used by both the CLI and the REST API::

        <base_dir>/<run_id>/
            trades.json
            summary.json
            log_returns.csv
            metrics.json
            events.jsonl
            similarity/<key>.npy
    """

    def __init__(self, base_dir: str | Path) -> None:
        """Initialise the store with a base directory.

        Args:
            base_dir: Root directory for run artifacts. Created if it
                does not already exist.
        """
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_run(
        self,
        run_id: str,
        artifacts: RunArtifacts,
        *,
        metrics: Mapping[str, object],
        events: Sequence[Mapping[str, object]],
    ) -> RunPaths:
        """Persist the canonical artifact bundle for one run.

        Args:
            run_id: Identifier for the run.
            artifacts: The :class:`RunArtifacts` produced by the
                pipeline.
            metrics: Metrics payload to serialise into
                ``metrics.json``.
            events: Structured events to serialise into ``events.jsonl``.

        Returns:
            Absolute paths of every artifact written.
        """
        run_dir = self._resolve_run_dir(run_id)
        self._write_trades(run_dir, artifacts.trades)
        self._write_summary(run_dir, artifacts.summary)
        self._write_log_returns(run_dir, artifacts.returns)
        self._write_metrics(run_dir, metrics)
        self._write_events(run_dir, events)
        similarity_dir = self._write_similarity(run_dir, artifacts.similarity_matrices)
        return RunPaths(
            trades_json=run_dir / "trades.json",
            summary_json=run_dir / "summary.json",
            log_returns_csv=run_dir / "log_returns.csv",
            metrics_json=run_dir / "metrics.json",
            events_jsonl=run_dir / "events.jsonl",
            similarity_dir=similarity_dir,
        )

    def read_trades(self, run_id: str) -> list[dict[str, object]]:
        """Read back the trades JSON for ``run_id``."""
        path = self.run_dir(run_id) / "trades.json"
        payload = self._read_json_list(path)
        return [{str(k): v for k, v in item.items()} for item in payload]

    def read_summary(self, run_id: str) -> list[dict[str, object]]:
        """Read back the summary JSON for ``run_id``."""
        path = self.run_dir(run_id) / "summary.json"
        payload = self._read_json_list(path)
        return [{str(k): v for k, v in item.items()} for item in payload]

    def read_metrics(self, run_id: str) -> dict[str, object]:
        """Read back the metrics JSON for ``run_id``."""
        path = self.run_dir(run_id) / "metrics.json"
        payload = self._read_json_dict(path)
        return {str(k): v for k, v in payload.items()}

    def read_log_returns_text(self, run_id: str) -> str:
        """Return the raw CSV text of the log-returns file for ``run_id``."""
        return (self.run_dir(run_id) / "log_returns.csv").read_text(encoding="utf-8")

    def run_dir(self, run_id: str) -> Path:
        """Return the directory containing ``run_id``'s artifacts.

        Raises:
            FileNotFoundError: When the run directory does not exist.
        """
        run_dir = self.base_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run '{run_id}' not found in {self.base_dir}")
        return run_dir

    def write_upload(self, run_id: str, content: str) -> Path:
        """Persist an uploaded CSV string under ``base_dir/uploads``.

        Args:
            run_id: Identifier used to namespace the upload.
            content: Raw CSV text supplied by the caller.

        Returns:
            The on-disk path of the uploaded file.
        """
        upload_dir = self.base_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        csv_path = upload_dir / f"{run_id}.csv"
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def _resolve_run_dir(self, run_id: str) -> Path:
        """Compute and create the per-run directory."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _write_trades(run_dir: Path, trades: Sequence[PortfolioResult]) -> None:
        """Write the ``trades.json`` file for a run."""
        path = run_dir / "trades.json"
        path.write_text(
            json.dumps(FileArtifactStore.trades_to_records(trades), default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _write_summary(run_dir: Path, summaries: Sequence[EvaluationSummary]) -> None:
        """Write the ``summary.json`` file for a run."""
        path = run_dir / "summary.json"
        path.write_text(
            json.dumps(FileArtifactStore.summaries_to_records(summaries), default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _write_log_returns(run_dir: Path, returns: pd.DataFrame) -> None:
        """Write the ``log_returns.csv`` file for a run."""
        path = run_dir / "log_returns.csv"
        path.write_text(returns.to_csv(), encoding="utf-8")

    @staticmethod
    def _write_metrics(run_dir: Path, metrics: Mapping[str, object]) -> None:
        """Write the ``metrics.json`` file for a run."""
        path = run_dir / "metrics.json"
        path.write_text(json.dumps(dict(metrics), default=str), encoding="utf-8")

    @staticmethod
    def _write_events(run_dir: Path, events: Sequence[Mapping[str, object]]) -> None:
        """Write the ``events.jsonl`` file for a run."""
        path = run_dir / "events.jsonl"
        path.write_text(
            "\n".join(json.dumps(dict(event), default=str) for event in events) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_similarity(run_dir: Path, matrices: Mapping[str, np.ndarray]) -> Path:
        """Write the per-scenario similarity ``.npy`` files for a run."""
        similarity_dir = run_dir / "similarity"
        similarity_dir.mkdir(exist_ok=True)
        for key, matrix in matrices.items():
            (similarity_dir / f"{key}.npy").write_bytes(FileArtifactStore._numpy_save_bytes(matrix))
        return similarity_dir

    @staticmethod
    def _numpy_save_bytes(matrix: np.ndarray) -> bytes:
        """Serialise a NumPy array to an in-memory ``.npy`` byte buffer."""
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def _read_json_list(path: Path) -> list[Mapping[str, object]]:
        """Read a JSON file containing a top-level list."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected list at root of {path}")
        return payload

    @staticmethod
    def _read_json_dict(path: Path) -> Mapping[str, object]:
        """Read a JSON file containing a top-level object."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object at root of {path}")
        return payload

    @staticmethod
    def trades_to_records(trades: Sequence[PortfolioResult]) -> list[dict[str, object]]:
        """Serialise a sequence of trades to JSON-friendly dicts."""
        return [
            {
                "strategy": trade.strategy,
                "horizon_days": trade.horizon_days,
                "rebalance_date": trade.rebalance_date.isoformat(),
                "exit_date": trade.exit_date.isoformat(),
                "selected_assets": list(trade.selected_assets),
                "weights": dict(trade.weights),
                "turnover": trade.turnover,
                "gross_return": trade.gross_return,
                "net_return": trade.net_return,
            }
            for trade in trades
        ]

    @staticmethod
    def summaries_to_records(
        summaries: Sequence[EvaluationSummary],
    ) -> list[dict[str, object]]:
        """Serialise a sequence of summaries to JSON-friendly dicts."""
        return [asdict(summary) for summary in summaries]


__all__ = ["FileArtifactStore"]
