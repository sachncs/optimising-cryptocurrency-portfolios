"""Interface layer: CLI entry points and REST API."""

from .api import create_app
from .cli import (
    CLIArgs,
    RealtimeCLIArgs,
    parse_arguments,
    parse_realtime_arguments,
)
from .cli.main import main, realtime_main

__all__ = [
    "CLIArgs",
    "RealtimeCLIArgs",
    "create_app",
    "main",
    "parse_arguments",
    "parse_realtime_arguments",
    "realtime_main",
]
