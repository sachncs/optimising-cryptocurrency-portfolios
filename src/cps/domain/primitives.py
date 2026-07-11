"""Domain value objects.

This module defines the immutable value types that flow through every
layer of the application. They replace the raw primitives (``dict``,
``float``, ``int``, ``pd.DataFrame``) previously scattered through the
codebase.

Every primitive is a frozen dataclass whose ``__post_init__`` enforces
the domain invariant at construction time -- type checkers therefore
catch violations before the pipeline runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Horizon:
    """Number of calendar days in a holding window.

    Args:
        days: Positive integer.

    Raises:
        ValueError: When ``days < 1``.

    Examples:
        >>> Horizon(1)
        Horizon(days=1)
    """

    days: int

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError(f"horizon days must be >= 1, got {self.days}")

    def annual_to_daily_risk_free_rate(self, annual_rate: float) -> float:
        """Compound an annual risk-free rate to the horizon's daily rate.

        Uses ``((1 + r) ** (1 / 365)) - 1`` so the daily rate compounds
        exactly to the supplied annual rate over a 365-day window.

        Args:
            annual_rate: Annualised rate as a decimal.

        Returns:
            Daily rate as a decimal.
        """
        return float((1.0 + annual_rate) ** (1.0 / 365.0) - 1.0)


@dataclass(frozen=True)
class GrossReturn:
    """Compounded simple return over a holding window, before costs."""

    value: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.value <= 10.0):
            raise ValueError(f"gross return {self.value} outside plausible range [-1, 10]")


@dataclass(frozen=True)
class NetReturn:
    """Compounded simple return over a holding window, after costs."""

    value: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.value <= 10.0):
            raise ValueError(f"net return {self.value} outside plausible range [-1, 10]")

    @classmethod
    def from_gross_and_cost(cls, gross: GrossReturn, cost_rate: float) -> "NetReturn":
        """Apply the multiplicative cost model ``(1 + g) * (1 - c) - 1``."""
        if not (0.0 <= cost_rate <= 1.0):
            raise ValueError(f"cost_rate must be in [0, 1], got {cost_rate}")
        return cls((1.0 + gross.value) * (1.0 - cost_rate) - 1.0)


@dataclass(frozen=True)
class Weights:
    """Long-only portfolio weights on the unit simplex.

    Weights are non-negative and sum to 1. Construction validates the
    invariant; invalid inputs raise ``ValueError`` at the boundary.
    The internal mapping is read-only; callers receive an immutable
    view.

    Attributes:
        mapping: Read-only ``Mapping[str, float]`` of asset -> weight.
        tolerance: Allowed deviation from the simplex sum.
    """

    mapping: Mapping[str, float]
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        weights = dict(self.mapping)
        if not weights:
            raise ValueError("Weights must contain at least one asset")
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError(f"weights must be non-negative; got {weights}")
        total = sum(weights.values())
        if abs(total - 1.0) > self.tolerance:
            raise ValueError(f"weights must sum to 1, got {total}")
        object.__setattr__(self, "mapping", MappingProxyType(weights))

    @classmethod
    def equal_weight(cls, assets: Sequence[str]) -> "Weights":
        """Build an equal-weighted portfolio for the given assets."""
        if not assets:
            raise ValueError("assets must be non-empty")
        weight = 1.0 / len(assets)
        return cls({asset: weight for asset in assets})

    @classmethod
    def from_series(cls, series: pd.Series, tolerance: float = 1e-8) -> "Weights":
        """Build from a ``pd.Series`` indexed by asset."""
        if series.empty:
            raise ValueError("series must be non-empty")
        weights = {str(asset): float(value) for asset, value in series.items()}
        return cls(weights, tolerance=tolerance)

    def to_series(self) -> pd.Series:
        """Materialise as a ``pd.Series`` indexed by asset."""
        return pd.Series(dict(self.mapping), name="weight", dtype=float)

    @property
    def assets(self) -> tuple[str, ...]:
        """The asset names in insertion order."""
        return tuple(self.mapping.keys())

    @property
    def turnover(self) -> float:
        """``sum(|weights|)`` -- equals 1.0 for simplex weights."""
        return float(sum(abs(value) for value in self.mapping.values()))


@dataclass(frozen=True)
class ScenarioKey:
    """Stable identifier for a single pipeline scenario.

    Replaces the ad-hoc ``f"{strategy}_h{horizon}_t{rebalance_index}"``
    strings used to key the consensus similarity matrices.
    """

    strategy: str
    horizon: Horizon
    rebalance_index: int

    def __post_init__(self) -> None:
        if self.rebalance_index < 0:
            raise ValueError(f"rebalance_index must be >= 0, got {self.rebalance_index}")

    def __str__(self) -> str:
        return f"{self.strategy}_h{self.horizon.days}_t{self.rebalance_index}"


@dataclass(frozen=True)
class CovarianceMatrix:
    """Symmetric positive-semidefinite covariance matrix.

    Stored as a read-only mapping view plus an explicit asset ordering.
    Construction validates symmetry and finite values; positive
    definiteness is not enforced (singular matrices are valid inputs
    for shrinkage estimators).
    """

    assets: tuple[str, ...]
    matrix: Mapping[tuple[str, str], float]
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        matrix = dict(self.matrix)
        n = len(self.assets)
        for i, asset_a in enumerate(self.assets):
            for j, asset_b in enumerate(self.assets):
                if (asset_a, asset_b) not in matrix:
                    raise ValueError(f"missing covariance entry ({asset_a!r}, {asset_b!r})")
                value = matrix[(asset_a, asset_b)]
                if not np.isfinite(value):
                    raise ValueError(f"non-finite covariance entry ({asset_a!r}, {asset_b!r})")
                # Symmetry check only on the upper-triangle to avoid
                # double-checking; if the matrix is square and n x n,
                # the lower triangle must match the upper.
                if i < j:
                    other = matrix[(asset_b, asset_a)]
                    if abs(value - other) > self.tolerance * max(1.0, abs(value), abs(other)):
                        raise ValueError(
                            f"asymmetric covariance entry ({asset_a!r}, {asset_b!r}): "
                            f"{value} != {other}"
                        )
        object.__setattr__(self, "matrix", MappingProxyType(matrix))

    @classmethod
    def from_dataframe(
        cls, frame: pd.DataFrame, tolerance: float = 1e-6
    ) -> "CovarianceMatrix":
        """Build from a square ``pd.DataFrame`` indexed by asset."""
        if frame.empty or frame.shape[0] != frame.shape[1]:
            raise ValueError("covariance frame must be non-empty and square")
        assets = tuple(str(c) for c in frame.columns)
        matrix: dict[tuple[str, str], float] = {}
        for asset_a in assets:
            for asset_b in assets:
                matrix[(asset_a, asset_b)] = float(frame.loc[asset_a, asset_b])
        return cls(assets, matrix, tolerance=tolerance)

    def to_dataframe(self) -> pd.DataFrame:
        """Materialise as a ``pd.DataFrame`` indexed by asset."""
        n = len(self.assets)
        data = [[self.matrix[(self.assets[i], self.assets[j])] for j in range(n)] for i in range(n)]
        return pd.DataFrame(data, index=list(self.assets), columns=list(self.assets))


def freeze_trades(trades: list["PortfolioResult"]) -> tuple["PortfolioResult", ...]:  # type: ignore[name-defined]  # noqa: F821
    """Convert a list of trades into an immutable tuple."""
    return tuple(trades)


def freeze_summary(summary: list["EvaluationSummary"]) -> tuple["EvaluationSummary", ...]:  # type: ignore[name-defined]  # noqa: F821
    """Convert a list of summaries into an immutable tuple."""
    return tuple(summary)


def freeze_similarity_matrices(
    matrices: dict[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    """Convert a dict of similarity matrices into a read-only mapping view."""
    return MappingProxyType(dict[str, np.ndarray](matrices))