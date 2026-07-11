"""``PipelineConfig`` and the per-algorithm forecaster configurations.

The previous monolithic ``PipelineConfig`` carried flat ``lstm_*`` and
``garch_*`` fields. They have been promoted to proper
``LSTMTrainingConfig`` and ``GARCHForecastConfig`` instances grouped
inside a single ``ForecasterConfig`` so the inner loop stops rebuilding
nested dataclasses by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..domain.primitives import Horizon

GARCHMeanModel = Literal["Zero", "Constant", "AR"]
GARCHDistribution = Literal["normal", "t", "skewt"]


@dataclass(frozen=True)
class GARCHForecastConfig:
    """Configuration for the GARCH forecaster.

    Attributes:
        mean: GARCH mean model. ``"Zero"`` is the conventional choice
            for return series; ``"Constant"`` fits an intercept;
            ``"AR"`` adds an autoregressive mean.
        p: GARCH lag order.
        o: Asymmetry / news-impact order (``o=0`` collapses to GARCH,
            ``o=1`` adds a leverage term).
        q: ARCH lag order.
        dist: Innovation distribution.
        rescale: Multiplicative rescale for numerical stability.
        auto_order: When ``True``, fit a small AIC candidate grid and
            pick the model with the lowest AIC.
    """

    mean: GARCHMeanModel = "Zero"
    p: int = 1
    o: int = 1
    q: int = 1
    dist: GARCHDistribution = "t"
    rescale: float = 100.0
    auto_order: bool = True


@dataclass(frozen=True)
class LSTMTrainingConfig:
    """Hyper-parameters for the LSTM forecaster."""

    lookback: int = 10
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    max_epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 10
    validation_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class StrategySpec:
    """A single strategy variant exercised at every rebalance."""

    name: str
    use_prediction: bool
    use_shifts: bool


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for portfolio construction and evaluation pipeline.

    Attributes:
        train_window_days: Number of days of returns used to fit the
            forecaster and estimate the covariance matrix.
        correlation_window_days: Width of the rolling correlation
            window used to build each consensus similarity matrix.
        rebalance_step_days: Number of days between rebalances.
        horizons: Holding periods to evaluate.
        consensus_runs: Number of independent Louvain partitions per
            rebalance.
        majority_threshold: Co-membership probability cutoff for
            declaring two assets stable neighbours.
        risk_free_rate_annual: Annualised risk-free rate.
        forecast_method: One of ``"naive"``, ``"arima"``, ``"garch"``,
            ``"lstm"``.
        random_seed: Seed for the NumPy RNG used by Louvain passes.
        weight_cap: Per-asset cap.
        max_assets: Upper bound on selected assets.
        min_assets: Lower bound on selected assets.
        max_volatility_annual: Annualised volatility ceiling.
        transaction_cost_bps: One-way commission in bps.
        slippage_bps: Expected price impact in bps.
        forecaster: Composite per-algorithm forecaster configuration.
    """

    train_window_days: int = 180
    correlation_window_days: int = 60
    rebalance_step_days: int = 30
    horizons: tuple[Horizon, ...] = (Horizon(1), Horizon(3), Horizon(7), Horizon(14))
    consensus_runs: int = 20
    majority_threshold: float = 0.5
    risk_free_rate_annual: float = 0.045
    forecast_method: str = "arima"
    random_seed: int = 42
    weight_cap: float = 0.35
    max_assets: int = 25
    min_assets: int = 2
    max_volatility_annual: float = 1.2
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    forecaster: "ForecasterConfig" = field(default_factory=lambda: ForecasterConfig())

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("at least one horizon is required")
        if self.consensus_runs < 1:
            raise ValueError("consensus_runs must be >= 1")
        if not (0.0 < self.majority_threshold <= 1.0):
            raise ValueError("majority_threshold must be in (0, 1]")
        if self.weight_cap <= 0:
            raise ValueError("weight_cap must be > 0")

    @classmethod
    def with_overrides(cls, **aliases: object) -> "PipelineConfig":
        """Build a config accepting short aliases for the longer canonical names.

        Recognised aliases:
            ``seed`` -> ``random_seed``
            ``rf_annual`` -> ``risk_free_rate_annual``
        """
        renamed = dict(aliases)
        if "seed" in renamed and "random_seed" not in renamed:
            renamed["random_seed"] = renamed.pop("seed")
        if "rf_annual" in renamed and "risk_free_rate_annual" not in renamed:
            renamed["risk_free_rate_annual"] = renamed.pop("rf_annual")
        return cls(**renamed)


@dataclass(frozen=True)
class ForecasterConfig:
    """Composite configuration for the forecaster registry."""

    garch: GARCHForecastConfig = field(default_factory=GARCHForecastConfig)
    lstm: LSTMTrainingConfig = field(default_factory=LSTMTrainingConfig)


def default_strategy_specs() -> tuple[StrategySpec, ...]:
    """The four fixed strategy variants exercised at every rebalance."""
    return (
        StrategySpec("baseline", use_prediction=False, use_shifts=False),
        StrategySpec("s", use_prediction=False, use_shifts=True),
        StrategySpec("p", use_prediction=True, use_shifts=False),
        StrategySpec("p-s", use_prediction=True, use_shifts=True),
    )


__all__ = [
    "ForecasterConfig",
    "GARCHDistribution",
    "GARCHForecastConfig",
    "GARCHMeanModel",
    "LSTMTrainingConfig",
    "PipelineConfig",
    "StrategySpec",
    "default_strategy_specs",
]