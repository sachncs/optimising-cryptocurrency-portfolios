"""Domain layer tests: primitives, events, policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cps.domain import (
    MIN_HISTORY_FOR_DRIFT,
    CovarianceMatrix,
    ForecastDriftPayload,
    ForecastGovernance,
    GrossReturn,
    Horizon,
    NetReturn,
    PipelineEvent,
    PipelineStartedPayload,
    RebalanceExecutedPayload,
    RiskLimits,
    ScenarioKey,
    Weights,
    apply_weight_cap,
    compute_effective_weight_cap,
)

# ----- Horizon -----


class TestHorizon:
    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            Horizon(0)
        with pytest.raises(ValueError):
            Horizon(-1)

    def test_annual_to_daily_compounding(self):
        # ((1 + 0.045) ** (1/365)) - 1
        expected = (1.045 ** (1.0 / 365.0)) - 1.0
        assert Horizon(1).annual_to_daily_risk_free_rate(0.045) == pytest.approx(expected)


# ----- Returns -----


class TestReturns:
    def test_gross_return_validation(self):
        with pytest.raises(ValueError):
            GrossReturn(-2.0)
        with pytest.raises(ValueError):
            GrossReturn(20.0)

    def test_net_return_from_gross(self):
        gross = GrossReturn(0.10)
        net = NetReturn.from_gross_and_cost(gross, 0.001)
        assert net.value == pytest.approx((1.1) * 0.999 - 1.0)

    def test_cost_rate_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            NetReturn.from_gross_and_cost(GrossReturn(0.10), -0.1)
        with pytest.raises(ValueError):
            NetReturn.from_gross_and_cost(GrossReturn(0.10), 1.5)


# ----- Weights -----


class TestWeights:
    def test_valid_simplex_accepted(self):
        w = Weights({"a": 0.6, "b": 0.4})
        assert w.assets == ("a", "b")
        assert w.turnover == pytest.approx(1.0)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            Weights({})

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            Weights({"a": -0.1, "b": 1.1})

    def test_rejects_not_summing_to_one(self):
        with pytest.raises(ValueError):
            Weights({"a": 0.5, "b": 0.3})

    def test_equal_weight(self):
        w = Weights.equal_weight(["a", "b", "c"])
        assert w.assets == ("a", "b", "c")
        for value in w.mapping.values():
            assert value == pytest.approx(1 / 3)

    def test_round_trip_through_series(self):
        original = pd.Series({"a": 0.2, "b": 0.3, "c": 0.5})
        w = Weights.from_series(original)
        assert w.to_series().to_dict() == pytest.approx(original.to_dict())


# ----- ScenarioKey -----


class TestScenarioKey:
    def test_string_representation(self):
        key = ScenarioKey("baseline", Horizon(7), 42)
        assert str(key) == "baseline_h7_t42"

    def test_rejects_negative_index(self):
        with pytest.raises(ValueError):
            ScenarioKey("baseline", Horizon(7), -1)


# ----- CovarianceMatrix -----


class TestCovarianceMatrix:
    def test_from_dataframe_round_trip(self):
        df = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["a", "b"],
            columns=["a", "b"],
        )
        matrix = CovarianceMatrix.from_dataframe(df)
        assert matrix.assets == ("a", "b")
        np.testing.assert_allclose(matrix.to_dataframe().to_numpy(), df.to_numpy())

    def test_rejects_non_square(self):
        with pytest.raises(ValueError):
            df = pd.DataFrame([[0.04, 0.01]], index=["a", "b"], columns=["a"])
            CovarianceMatrix.from_dataframe(df)

    def test_rejects_missing_entries(self):
        with pytest.raises(ValueError):
            CovarianceMatrix(assets=("a", "b"), matrix={("a", "a"): 0.1})

    def test_rejects_asymmetric_entries(self):
        with pytest.raises(ValueError):
            CovarianceMatrix(
                assets=("a", "b"),
                matrix={("a", "a"): 0.1, ("a", "b"): 0.01, ("b", "a"): 0.5, ("b", "b"): 0.2},
            )


# ----- Events -----


class TestEvents:
    def test_event_enum_string_values(self):
        assert PipelineEvent.PIPELINE_STARTED.value == "pipeline_started"
        assert PipelineEvent.PIPELINE_COMPLETED.value == "pipeline_completed"

    def test_event_payloads_are_frozen(self):
        payload = PipelineStartedPayload(rows=100, assets=4)
        with pytest.raises(Exception):
            payload.rows = 5  # type: ignore[misc]

    def test_rebalance_payload_carries_expected_fields(self):
        payload = RebalanceExecutedPayload(
            strategy="baseline",
            horizon_days=7,
            rebalance_index=3,
            n_assets_selected=5,
            net_return=0.012,
        )
        assert payload.strategy == "baseline"

    def test_drift_payload(self):
        payload = ForecastDriftPayload(history_points=15)
        assert payload.history_points == 15


# ----- Policies -----


class TestRiskLimitsAndWeightCap:
    def test_compute_effective_weight_cap_raises_on_invalid(self):
        with pytest.raises(ValueError):
            compute_effective_weight_cap(0.0, 4)
        with pytest.raises(ValueError):
            compute_effective_weight_cap(0.1, 0)
        with pytest.raises(ValueError):
            compute_effective_weight_cap(-0.1, 4)

    def test_compute_effective_weight_cap_clamps_to_1_over_n(self):
        # When configured_cap is below 1/n, the cap is raised to 1/n.
        assert compute_effective_weight_cap(0.05, 5) == pytest.approx(0.2)

    def test_compute_effective_weight_cap_caps_at_one(self):
        assert compute_effective_weight_cap(5.0, 2) == 1.0

    def test_apply_weight_cap_redistributes_excess(self):
        weights = Weights({"a": 0.9, "b": 0.1})
        capped = apply_weight_cap(weights, 0.6)
        assert max(capped.mapping.values()) <= 0.6 + 1e-9
        assert sum(capped.mapping.values()) == pytest.approx(1.0)

    def test_risk_limits_validate_raises_when_constraints_violated(self):
        limits = RiskLimits(min_assets=3, max_assets=10, max_weight_per_asset=0.5, max_volatility_annual=10.0)
        weights = Weights({"a": 0.5, "b": 0.5})
        covariance = CovarianceMatrix.from_dataframe(
            pd.DataFrame(
                [[0.04, 0.001], [0.001, 0.04]],
                index=["a", "b"],
                columns=["a", "b"],
            )
        )
        with pytest.raises(ValueError, match="below minimum"):
            limits.validate(["a", "b"], weights, covariance)


class TestForecastGovernance:
    def test_drift_requires_minimum_history(self):
        gov = ForecastGovernance()
        for _ in range(MIN_HISTORY_FOR_DRIFT - 1):
            gov.record_error(0.1)
        assert not gov.is_drift_detected()

    def test_drift_detected_when_spike(self):
        gov = ForecastGovernance(drift_threshold_multiplier=1.5)
        for _ in range(MIN_HISTORY_FOR_DRIFT):
            gov.record_error(0.1)
        gov.record_error(0.5)
        assert gov.is_drift_detected()

    def test_snapshot_returns_immutable_tuple(self):
        gov = ForecastGovernance()
        gov.record_error(0.1)
        snapshot = gov.snapshot()
        assert isinstance(snapshot, tuple)
        assert snapshot == (0.1,)
