"""Tests for the observability and resilience primitives."""

from __future__ import annotations

import json

import pytest

from cps.domain import PipelineEvent, PipelineStartedPayload
from cps.infrastructure.observability import (
    MetricsRegistry,
    StructuredLogger,
    Timer,
)
from cps.infrastructure.resilience import RetryPolicy, execute_with_retry, require_optional


class TestMetricsRegistry:
    def test_increment_and_record(self):
        registry = MetricsRegistry()
        registry.increment("runs")
        registry.increment("runs", 3)
        registry.record_timing_millis("fit", 12.5)
        snapshot = registry.snapshot()
        assert snapshot.counters["runs"] == 4
        assert snapshot.timings_millis["fit"] == (12.5,)

    def test_snapshot_returns_immutable_mapping(self):
        registry = MetricsRegistry()
        registry.increment("x")
        snapshot = registry.snapshot()
        assert isinstance(snapshot.counters, dict)
        # Snapshot is decoupled from the registry.
        registry.increment("x", 5)
        assert snapshot.counters["x"] == 1


class TestStructuredLogger:
    def test_publish_console_only(self, capsys):
        logger = StructuredLogger("cps_test_console")
        logger.publish(
            PipelineEvent.PIPELINE_STARTED,
            PipelineStartedPayload(rows=10, assets=3),
        )
        captured = capsys.readouterr()
        assert "pipeline_started" in captured.out

    def test_publish_persists_to_disk(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        logger = StructuredLogger("cps_test_disk", str(log_path))
        logger.publish(
            PipelineEvent.PIPELINE_STARTED,
            PipelineStartedPayload(rows=5, assets=2),
        )
        contents = log_path.read_text(encoding="utf-8")
        assert "pipeline_started" in contents
        record = json.loads(contents.strip())
        assert record["event"] == "pipeline_started"
        assert record["rows"] == 5

    def test_listener_receives_typed_events(self):
        captured: list = []
        logger = StructuredLogger("cps_test_listener")
        logger.add_listener(lambda event, payload: captured.append((event, payload)))
        logger.publish(
            PipelineEvent.PIPELINE_STARTED,
            PipelineStartedPayload(rows=7, assets=4),
        )
        assert captured == [
            (
                PipelineEvent.PIPELINE_STARTED,
                PipelineStartedPayload(rows=7, assets=4),
            )
        ]

    def test_remove_unknown_listener_is_no_op(self):
        logger = StructuredLogger("cps_test_remove")
        logger.remove_listener(lambda event, payload: None)  # no error

    def test_close_is_safe(self):
        logger = StructuredLogger("cps_test_close")
        logger.close()

    def test_composes_with_existing_handlers(self, tmp_path):
        """The logger no longer clears pre-existing handlers on the named Logger."""
        import logging

        target = logging.getLogger("cps_test_compose")
        target.handlers.clear()
        pre_existing = logging.NullHandler()
        target.addHandler(pre_existing)
        logger = StructuredLogger("cps_test_compose")
        assert pre_existing in target.handlers
        # Our own handler was added on top.
        assert any(h is not pre_existing for h in target.handlers)


class TestTimer:
    def test_elapsed_millis_non_negative(self):
        timer = Timer()
        assert timer.elapsed_millis() >= 0.0


class TestRetryPolicy:
    def test_default_values(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_multiplier == 2.0

    def test_custom_values(self):
        policy = RetryPolicy(max_attempts=5, initial_backoff_seconds=0.5, backoff_multiplier=3.0)
        assert policy.max_attempts == 5


class TestExecuteWithRetry:
    def test_succeeds_first_attempt(self):
        policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0)
        assert execute_with_retry(lambda: 42, policy) == 42

    def test_eventually_succeeds(self):
        state = {"n": 0}

        def flaky() -> int:
            state["n"] += 1
            if state["n"] < 3:
                raise ValueError("not yet")
            return 7

        policy = RetryPolicy(max_attempts=5, initial_backoff_seconds=0.0)
        assert execute_with_retry(flaky, policy) == 7
        assert state["n"] == 3

    def test_exhausts_retries_and_raises(self):
        policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.0)
        with pytest.raises(RuntimeError, match="permanent"):
            execute_with_retry(lambda: (_ for _ in ()).throw(RuntimeError("permanent")), policy)

    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_backoff_schedule(self):
        policy = RetryPolicy(max_attempts=3, initial_backoff_seconds=0.01, backoff_multiplier=2.0)
        delays: list[float] = []

        def recording_sleep(d: float) -> None:
            delays.append(d)

        policy = RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.01,
            backoff_multiplier=2.0,
            sleep=recording_sleep,
        )
        with pytest.raises(ValueError):
            execute_with_retry(
                lambda: (_ for _ in ()).throw(ValueError("boom")), policy
            )
        assert delays == [0.01, 0.02]


class TestRequireOptional:
    def test_raises_with_helpful_message(self):
        with pytest.raises(RuntimeError, match="Install the optional extra"):
            require_optional("definitely_not_a_real_package_xyz", "fake-extra")
