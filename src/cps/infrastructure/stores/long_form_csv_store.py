"""Long-form OHLCV CSV store with deduplication.

Extracted from the previous ``realtime.py`` implementation. Holds a
single dataset (``pd.DataFrame``) per on-disk file; appends de-duplicate
on ``(timestamp, symbol)`` so repeated polls never produce duplicate
rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...config.settings import CCXT_SUPPORTED_TIMEFRAMES


class LongFormCsvStore:
    """Append-only CSV store with deduplication on a key column.

    Each call to :meth:`append` reads the existing CSV (if present),
    concatenates the new rows, deduplicates on ``(timestamp_col,
    symbol_col)`` keeping the freshest copy, sorts, and writes back.
    """

    def __init__(
        self,
        path: Path,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        timestamp_col: str | None = None,
    ) -> None:
        """Initialise the store at ``path``.

        Args:
            path: Destination CSV file. Parent directories are created
                on the first append.
            date_col: Name of the date column. Defaults to ``"date"``.
                When ``timestamp_col`` is also supplied, ``timestamp_col``
                wins -- the ``date_col`` alias exists for callers that
                share column-name configuration across the long-form CSV
                and the wide price frame.
            symbol_col: Name of the symbol column. Defaults to
                ``"symbol"``.
            timestamp_col: Explicit column name override.
        """
        self.path = Path(path)
        self.timestamp_col = timestamp_col or date_col
        self.symbol_col = symbol_col

    def append(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Append ``frame`` to the CSV and return the rows added in this call.

        Args:
            frame: New rows to persist.

        Returns:
            The merged frame after append + deduplication.
        """
        if frame.empty:
            if self.path.exists():
                return pd.read_csv(self.path)
            return frame
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing = pd.read_csv(self.path)
            existing[self.timestamp_col] = pd.to_datetime(existing[self.timestamp_col], utc=True)
            merged = pd.concat([existing, frame], ignore_index=True)
        else:
            merged = frame
        merged = merged.drop_duplicates(subset=[self.timestamp_col, self.symbol_col], keep="last")
        merged = merged.sort_values([self.timestamp_col, self.symbol_col]).reset_index(drop=True)
        merged.to_csv(self.path, index=False)
        return merged

    @staticmethod
    def supported_timeframes() -> frozenset[str]:
        """Return the set of timeframes accepted by the ccxt poller."""
        return CCXT_SUPPORTED_TIMEFRAMES


__all__ = ["LongFormCsvStore"]
