"""Idempotent run markers and content-hash run identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .config import PipelineConfig


def build_run_id(config: PipelineConfig) -> str:
    """Compute a deterministic 16-hex-character run identifier for ``config``."""
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ensure_idempotent_run(run_directory: str, run_id: str) -> Path:
    """Reserve ``run_directory/<run_id>.done`` and return its path."""
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{run_id}.done"
    if marker.exists():
        raise ValueError(f"Run {run_id} has already completed in {directory}")
    return marker


def mark_run_complete(marker_path: Path) -> None:
    """Write the completion marker after a successful run."""
    marker_path.write_text("completed\n", encoding="utf-8")


__all__ = ["build_run_id", "ensure_idempotent_run", "mark_run_complete"]