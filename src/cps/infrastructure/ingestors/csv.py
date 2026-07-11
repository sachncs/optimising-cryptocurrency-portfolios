"""CSV-file ingestor."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from ...data import load_price_data
from ...domain.protocols import Ingestor


class CsvIngestor:
    """Loads a CSV file via :func:`cps.data.load_price_data`."""

    name: ClassVar[str] = "csv"

    def __init__(self, path: str, date_col: str = "date") -> None:
        """Initialise the ingestor with the CSV path and date column."""
        self.__path = path
        self.__date_col = date_col

    def fetch(self) -> pd.DataFrame:
        """Read the CSV and return the price frame."""
        return load_price_data(self.__path, date_col=self.__date_col)


__all__ = ["CsvIngestor"]