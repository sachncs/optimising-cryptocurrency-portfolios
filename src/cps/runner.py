"""Idempotent run markers and content-hash run IDs.

This module implements the *idempotency contract* of the pipeline: given
the same :class:`cps.pipeline.PipelineConfig`, running the CLI twice in a
row must not silently produce two distinct artifacts on disk. It also
provides the canonical run-id derivation used in artifact filenames.

The contract is enforced by two cooperating pieces:

* :func:`build_run_id` -- produces a deterministic 16-hex-character run
  identifier from the config. Identical configurations map to identical
  identifiers.
* :func:`ensure_idempotent_run` -- before the run starts, raises if a
  completion marker for that identifier already exists.

Together they make the CLI safe to re-invoke (for example, after a failed
run aborted before the marker was written).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .pipeline import PipelineConfig


def build_run_id(config: PipelineConfig) -> str:
    """Compute a deterministic 16-hex-character run identifier for ``config``.

    The identifier is the first 16 hex characters of ``SHA-256`` applied to
    a JSON-canonicalised representation of the config (sorted keys, default
    ``dataclasses.asdict`` serialisation).

    Args:
        config: The pipeline configuration to fingerprint.

    Returns:
        A 16-character hexadecimal string. Identical configurations always
        produce identical identifiers; any change to a config field
        produces a different one with overwhelming probability.

    Notes:
        A SHA-256 prefix of 16 hex digits encodes 64 bits of entropy --
        ample for the deployment scales this project targets (single-digit
        thousands of distinct runs per day). Truncating the full digest
        keeps file-system paths short and human-readable.
    """
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ensure_idempotent_run(run_directory: str, run_id: str) -> Path:
    """Reserve ``run_directory/<run_id>.done`` and return its path.

    Creates the directory if it does not already exist. If the marker is
    present, the run is considered already completed and a ``ValueError``
    is raised so the CLI aborts cleanly rather than overwriting previous
    artifacts.

    Args:
        run_directory: Directory in which the marker should live. Created
            if missing.
        run_id: The run identifier (typically from :func:`build_run_id`).

    Returns:
        The :class:`pathlib.Path` of the marker file. The marker does not
        yet exist on disk -- :func:`mark_run_complete` must be called once
        the run finishes successfully.

    Raises:
        ValueError: If the marker already exists. The caller should treat
            this as a successful no-op (the previous run already wrote the
            same artifacts).
    """
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{run_id}.done"
    if marker.exists():
        raise ValueError(f"Run {run_id} has already completed in {directory}")
    return marker


def mark_run_complete(marker_path: Path) -> None:
    """Write the completion marker after a successful run.

    Args:
        marker_path: The path returned by :func:`ensure_idempotent_run`.
            Overwrites any existing content (the marker is created by the
            previous call but not populated).
    """
    marker_path.write_text("completed\n", encoding="utf-8")
