"""Return forecasting: naive, ARIMA, GARCH, and a dispatcher.

This module implements the return forecasting layer used by
:func:`cps.pipeline.build_consensus_partitions`. Three families of
forecaster are provided:

* **Naive** -- constant last-value projection. Trivial baseline, useful
  as a sanity check and as the silent fallback for ARIMA on degenerate
  inputs.
* **ARIMA** -- ``statsmodels.tsa.arima.model.ARIMA`` fit on the training
  window. Silently falls back to naive on any fitting error so the
  pipeline never aborts on a single degenerate asset.
* **GARCH** -- ``arch.arch_model`` with optional AIC-based order
  selection. Requires the optional ``[forecast-garch]`` extra (the
  ``arch`` package). Rescaling inputs by 100.0 improves numerical
  stability of the MLE optimiser.

The dispatcher :func:`forecast_matrix` accepts a ``method`` string and
threads an optional ``garch_config`` / ``lstm_config`` through to the
specialised forecasters. The LSTM path is dispatched to
:func:`cps.lstm_model.lstm_forecast_matrix` so the heavy import is
delayed until needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .lstm_model import LSTMTrainingConfig

#: Mean-model literals accepted by ``arch.arch_model``.
GARCHMeanModel = Literal["Zero", "Constant", "AR"]
#: Innovation-distribution literals accepted by ``arch.arch_model``.
GARCHDistribution = Literal["normal", "t", "skewt"]


@dataclass(frozen=True)
class GARCHForecastConfig:
    """Configuration for the GARCH forecaster.

    Attributes:
        mean: Mean model. ``"Zero"`` (no constant mean -- the standard
            choice for return series), ``"Constant"`` (fit an intercept),
            or ``"AR"`` (autoregressive mean).
        p: GARCH lag order (long-run volatility memory).
        o: Asymmetry / news-impact order (``o=0`` collapses to GARCH,
            ``o=1`` adds a leverage term -- the GJR-GARCH variant).
        q: ARCH lag order (short-run volatility memory).
        dist: Innovation distribution (``"normal"``, ``"t"``,
            ``"skewt"``). ``"t"`` is the conventional choice for
            crypto returns because of their heavy tails.
        rescale: Multiplicative rescale applied to the input series.
            ``arch``'s MLE optimiser is more numerically stable when the
            data are scaled away from ``1e-3`` magnitudes -- we rescale
            by ``100`` so percentages become hundreds.
        auto_order: When ``True``, fit the user-supplied ``(p, o, q)``
            plus a small candidate grid and pick the model with the
            lowest AIC. When ``False``, fit only the supplied order.
    """

    mean: GARCHMeanModel = "Zero"
    p: int = 1
    o: int = 1
    q: int = 1
    dist: GARCHDistribution = "t"
    rescale: float = 100.0
    auto_order: bool = True


def _require_arch() -> None:
    """Lazy guard for the optional ``arch`` dependency.

    Raises:
        RuntimeError: With a message instructing the caller to install
            the ``[forecast-garch]`` extra.
    """
    try:
        import arch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise RuntimeError(
            "The GARCH forecaster requires the 'arch' package. "
            "Install the optional extra with: pip install 'crypto-portfolio-system[forecast-garch]'"
        ) from exc


def naive_forecast(train_returns: pd.Series, steps: int) -> pd.Series:
    """Return a constant last-value forecast.

    Args:
        train_returns: Historical returns for a single asset.
        steps: Number of forward steps to project.

    Returns:
        ``pd.Series`` of length ``steps`` whose every entry equals the
        last observation of ``train_returns``.

    Raises:
        ValueError: When ``train_returns`` is empty.
    """
    if train_returns.empty:
        raise ValueError("Train return series is empty")
    return pd.Series([train_returns.iloc[-1]] * steps, index=range(steps), dtype=float)


def arima_forecast(train_returns: pd.Series, steps: int, order: tuple[int, int, int] = (1, 0, 1)) -> pd.Series:
    """Fit an ARIMA(p, d, q) model and forecast ``steps`` ahead.

    Args:
        train_returns: Historical returns for a single asset.
        steps: Number of forward steps to project.
        order: ``(p, d, q)`` ARIMA order. Defaults to ``(1, 0, 1)``.

    Returns:
        ``pd.Series`` of length ``steps`` with the ARIMA forecast.
        Falls back to :func:`naive_forecast` when the training series has
        fewer than two distinct values or when ``statsmodels`` raises
        during fitting (e.g. on a perfectly constant series or on a
        non-stationary input the optimiser refuses).

    Raises:
        ValueError: When ``train_returns`` is empty (propagated from
            :func:`naive_forecast` if the fallback path is taken with an
            empty input).
    """
    if train_returns.nunique() < 2:
        # A constant series cannot be fit by ARIMA -- short-circuit.
        return naive_forecast(train_returns, steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(train_returns, order=order)
            fit = model.fit()
            pred = fit.forecast(steps=steps)
            return pd.Series(pred, index=range(steps), dtype=float)
        except Exception:
            # ``statsmodels`` can raise a wide variety of convergence /
            # numerical errors on degenerate inputs. The pipeline's
            # stability contract demands that forecasting never aborts the
            # whole run, so we fall back to the naive baseline.
            return naive_forecast(train_returns, steps)


def garch_forecast(
    train_returns: pd.Series,
    steps: int,
    config: GARCHForecastConfig | None = None,
) -> pd.Series:
    """Fit a GARCH(p, o, q) model and forecast ``steps`` ahead.

    Args:
        train_returns: Historical returns for a single asset.
        steps: Number of forward steps to project.
        config: Optional :class:`GARCHForecastConfig` overriding the
            defaults.

    Returns:
        ``pd.Series`` of length ``steps`` containing the *mean* path of
        the fitted GARCH process.

    Raises:
        ValueError: When ``train_returns`` is empty, ``steps < 1``, the
            rescale is non-positive, or the series has fewer than two
            distinct values (GARCH cannot fit a constant series).
        RuntimeError: When even the initial ``(p, o, q)`` fit fails or
            when the returned forecast has the wrong shape.

    Algorithm:
        When ``config.auto_order`` is ``True`` the function fits the
        user-supplied ``(p, o, q)`` plus the candidate grid
        ``[(1,0,1), (1,1,1), (2,1,1), (1,1,2), (2,0,1)]`` and selects
        the model with the lowest AIC. The candidate grid is small by
        design -- larger grids quickly become dominated by over-fitting
        on crypto return series.
    """
    _require_arch()
    if train_returns.empty:
        raise ValueError("Train return series is empty")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    cfg = config or GARCHForecastConfig()
    if cfg.rescale <= 0:
        raise ValueError("rescale must be positive")

    from arch import arch_model

    series = train_returns.astype(float).dropna()
    if series.empty or series.nunique() < 2:
        raise ValueError("GARCH requires a series with at least 2 distinct values")

    scaled = series * cfg.rescale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.auto_order:
            # Fit the user-supplied order first; it is the baseline that
            # the candidate grid tries to beat. If even this fit fails
            # there is nothing to compare against, so we raise.
            try:
                model = arch_model(
                    scaled,
                    mean=cfg.mean,
                    vol="GARCH",
                    p=cfg.p,
                    o=cfg.o,
                    q=cfg.q,
                    dist=cfg.dist,
                )
                auto_result = model.fit(disp="off", show_warning=False)
                best_aic = float(auto_result.aic)
                best_order = (cfg.p, cfg.o, cfg.q)
                best_result = auto_result
            except Exception as exc:
                raise RuntimeError(f"GARCH auto-fit failed: {exc}") from exc

            # Small AIC candidate grid. Each candidate is fit in turn;
            # failures (non-stationarity, convergence issues) are
            # silently skipped because the next candidate may still
            # converge.
            for candidate_p, candidate_o, candidate_q in [
                (1, 0, 1),
                (1, 1, 1),
                (2, 1, 1),
                (1, 1, 2),
                (2, 0, 1),
            ]:
                if (candidate_p, candidate_o, candidate_q) == best_order:
                    # Skip the seed order -- it was already fit above.
                    continue
                try:
                    candidate_model = arch_model(
                        scaled,
                        mean=cfg.mean,
                        vol="GARCH",
                        p=candidate_p,
                        o=candidate_o,
                        q=candidate_q,
                        dist=cfg.dist,
                    )
                    candidate_result = candidate_model.fit(disp="off", show_warning=False)
                except Exception:
                    continue
                candidate_aic = float(candidate_result.aic)
                if np.isfinite(candidate_aic) and candidate_aic < best_aic:
                    best_aic = candidate_aic
                    best_order = (candidate_p, candidate_o, candidate_q)
                    best_result = candidate_result

            forecast = best_result.forecast(horizon=steps)
            # ``arch`` returns the mean path as a 2-D ``(1, steps)``
            # array; flatten and take the trailing ``steps`` entries so
            # the shape is invariant to whatever auxiliary rows the
            # library prepends.
            mean_forecast = np.asarray(forecast.mean).flatten()[-steps:]
        else:
            model = arch_model(
                scaled,
                mean=cfg.mean,
                vol="GARCH",
                p=cfg.p,
                o=cfg.o,
                q=cfg.q,
                dist=cfg.dist,
            )
            fit = model.fit(disp="off", show_warning=False)
            forecast = fit.forecast(horizon=steps)
            mean_forecast = np.asarray(forecast.mean).flatten()[-steps:]

    if mean_forecast.size != steps:
        raise RuntimeError(f"GARCH forecast returned {mean_forecast.size} steps but {steps} were requested")
    return pd.Series(mean_forecast / cfg.rescale, index=range(steps), dtype=float)


def forecast_matrix(
    train_returns: pd.DataFrame,
    steps: int,
    method: str,
    garch_config: GARCHForecastConfig | None = None,
    lstm_config: LSTMTrainingConfig | None = None,
) -> pd.DataFrame:
    """Dispatch a per-asset forecast across a return matrix.

    Args:
        train_returns: ``pd.DataFrame`` of returns, one column per asset.
        steps: Number of forward steps to project per asset.
        method: One of ``"naive"``, ``"arima"``, ``"garch"``, ``"lstm"``.
        garch_config: Optional :class:`GARCHForecastConfig` (only used
            when ``method == "garch"``).
        lstm_config: Optional :class:`cps.lstm_model.LSTMTrainingConfig`
            (only used when ``method == "lstm"``).

    Returns:
        ``pd.DataFrame`` of shape ``(steps, n_assets)`` with one column
        per asset.

    Raises:
        ValueError: When ``method`` is not one of the recognised names.
    """
    if method == "lstm":
        # Lazy import: ``torch`` is an optional dependency. Importing it
        # only when the LSTM path is requested keeps the lightweight
        # default install usable.
        from .lstm_model import LSTMTrainingConfig, lstm_forecast_matrix

        return lstm_forecast_matrix(train_returns, steps, lstm_config or LSTMTrainingConfig())

    cols: dict[str, pd.Series] = {}
    for col in train_returns.columns:
        s = train_returns[col].astype(float)
        if method == "naive":
            cols[col] = naive_forecast(s, steps)
        elif method == "arima":
            cols[col] = arima_forecast(s, steps)
        elif method == "garch":
            cols[col] = garch_forecast(s, steps, garch_config or GARCHForecastConfig())
        else:
            raise ValueError(f"Unknown forecast method: {method}")
    return pd.DataFrame(cols)
