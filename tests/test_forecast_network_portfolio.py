from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cps.forecast import (
    GARCHForecastConfig,
    arima_forecast,
    forecast_matrix,
    garch_forecast,
    naive_forecast,
)
from cps.networking import (
    build_weighted_graph_from_distance,
    consensus_similarity_matrix,
    correlation_distance_matrix,
    louvain_partition,
    stable_clusters_from_similarity,
)
from cps.portfolio import (
    compute_ledoit_wolf_constant_variance_covariance,
    compute_portfolio_simple_return,
    optimize_maximum_sharpe_ratio,
    project_weights_to_simplex,
)


def test_forecast_methods_shapes():
    series = pd.Series([0.01, -0.02, 0.03, 0.01, 0.0])
    naive = naive_forecast(series, 3)
    arima = arima_forecast(series, 3)
    assert len(naive) == 3
    assert len(arima) == 3


def test_forecast_matrix_unknown_method_raises():
    frame = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.0, 0.01, -0.01]})
    with pytest.raises(ValueError):
        forecast_matrix(frame, 2, "invalid")


def test_forecast_matrix_lstm_dispatch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    rows = rng.normal(0, 0.01, size=(60, 3))
    frame = pd.DataFrame(rows, columns=["a", "b", "c"])
    out = forecast_matrix(frame, 2, "lstm")
    assert out.shape == (2, 3)
    assert list(out.columns) == ["a", "b", "c"]
    del torch


def test_forecast_matrix_garch_dispatch():
    pytest.importorskip("arch")
    rng = np.random.default_rng(0)
    rows = rng.normal(0, 0.01, size=(120, 2))
    frame = pd.DataFrame(rows, columns=["a", "b"])
    out = forecast_matrix(frame, 3, "garch")
    assert out.shape == (3, 2)


def test_garch_forecast_direct_with_explicit_config():
    pytest.importorskip("arch")
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 0.01, size=120))
    config = GARCHForecastConfig(p=1, o=1, q=1, dist="normal", auto_order=False)
    forecast = garch_forecast(series, 3, config)
    assert len(forecast) == 3


def test_garch_forecast_rejects_empty_series():
    pytest.importorskip("arch")
    with pytest.raises(ValueError):
        garch_forecast(pd.Series(dtype=float), 3)


def test_garch_forecast_rejects_constant_series():
    pytest.importorskip("arch")
    with pytest.raises(ValueError):
        garch_forecast(pd.Series([0.01] * 30), 3)


def test_garch_forecast_rejects_invalid_rescale():
    pytest.importorskip("arch")
    series = pd.Series([0.01, -0.02, 0.03] * 10)
    with pytest.raises(ValueError):
        garch_forecast(series, 3, GARCHForecastConfig(rescale=-1.0))


def test_garch_forecast_rejects_zero_steps():
    pytest.importorskip("arch")
    series = pd.Series([0.01, -0.02, 0.03] * 10)
    with pytest.raises(ValueError):
        garch_forecast(series, 0)


def test_forecast_matrix_passes_garch_config():
    pytest.importorskip("arch")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, 0.01, size=(120, 2)), columns=["a", "b"])
    config = GARCHForecastConfig(auto_order=False, dist="normal")
    out = forecast_matrix(frame, 2, "garch", garch_config=config)
    assert out.shape == (2, 2)


def test_network_and_consensus_pipeline_components():
    returns = pd.DataFrame(
        {
            "a": [0.01, 0.02, 0.01, 0.0],
            "b": [0.01, 0.021, 0.009, -0.001],
            "c": [-0.01, -0.02, -0.01, 0.0],
        }
    )
    distance = correlation_distance_matrix(returns)
    graph = build_weighted_graph_from_distance(distance)
    partition = louvain_partition(graph, seed=10)
    similarity = consensus_similarity_matrix([partition], list(returns.columns))
    clusters = stable_clusters_from_similarity(similarity, list(returns.columns), threshold=0.5)
    assert distance.shape == (3, 3)
    assert graph.number_of_nodes() == 3
    assert len(partition) >= 1
    assert similarity.shape == (3, 3)
    assert len(clusters) >= 1


def test_portfolio_helpers_constraints_and_return():
    returns = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.015], "b": [0.0, 0.01, 0.02, -0.005]})
    covariance = compute_ledoit_wolf_constant_variance_covariance(returns)
    weights = optimize_maximum_sharpe_ratio(returns.mean(), covariance, 0.0, max_iterations=200)
    projected = project_weights_to_simplex(np.array([0.6, 0.7]))
    future = pd.DataFrame({"a": [0.01, 0.01], "b": [0.02, -0.01]})
    value = compute_portfolio_simple_return(future, weights)
    assert covariance.shape == (2, 2)
    assert abs(weights.sum() - 1.0) < 1e-5
    assert (weights >= -1e-10).all()
    assert abs(projected.sum() - 1.0) < 1e-8
    assert isinstance(value, float)
