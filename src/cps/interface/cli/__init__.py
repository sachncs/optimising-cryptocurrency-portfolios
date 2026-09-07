"""CLI entry point for ``crypto-portfolio`` and ``cps-realtime``.

The console scripts (``crypto-portfolio``, ``cps-realtime``) live in
``cps.interface.cli.main``. They are intentionally *not* re-exported
from this package's ``__init__`` so that ``cps.interface.cli.main``
always resolves to the submodule -- downstream tests that need to
monkeypatch the binding for ``CCXTPoller`` rely on this.

Use ``from cps.interface.cli.main import main`` to import the entry
point directly, or ``from cps import main`` (which re-exports it from
the top-level ``cps`` package).
"""

from .main import (
    CLIArgs,
    RealtimeCLIArgs,
    parse_arguments,
    parse_realtime_arguments,
)

__all__ = [
    "CLIArgs",
    "RealtimeCLIArgs",
    "parse_arguments",
    "parse_realtime_arguments",
]
