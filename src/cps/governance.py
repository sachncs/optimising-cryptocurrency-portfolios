"""Forecast drift detection.

This module implements a tiny but useful governance primitive: every time
the pipeline produces a forecast, the in-sample MSE of the forecast against
the realised returns is recorded via :meth:`ForecastGovernance.record_error`,
and :meth:`ForecastGovernance.is_drift_detected` flags a drift event when
the most recent error is materially worse than the historical baseline.

The intent is to catch *distribution shift* early: if the model suddenly
gets worse (e.g. due to a regime change or data-pipeline regression), the
in-sample MSE spikes above the trailing baseline and the pipeline emits a
``forecast_drift_detected`` event so an operator can intervene.

Design notes
------------
* The heuristic is intentionally simple: ``latest > baseline * multiplier``
  with ``baseline = mean(history[:-1])``. This avoids the complexity of
  fitting a sequential change-point model while still surfacing obvious
  breakages.
* A minimum history of ``10`` samples is required before drift can be
  flagged. Below that threshold the heuristic is unreliable and silently
  returns ``False``.
* The detector is *additive*: callers may continue to feed it indefinitely
  and the baseline keeps shifting. There is no decay/weighting -- by
  design, drift is judged against the entire recorded history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ForecastGovernance:
    """Rolling MSE recorder with simple drift detection.

    The class is intentionally lightweight (a list plus two scalar
    thresholds) so it can be instantiated once per pipeline run and
    updated in the inner loop without measurable overhead.

    Attributes:
        mse_history: Per-rebalance MSE values in insertion order. Mutated
            by :meth:`record_error`.
        drift_threshold_multiplier: Multiplier on the historical baseline
            above which the latest sample is flagged as drift. Defaults to
            ``2.0`` (i.e. "twice as bad as the trailing average").
    """

    mse_history: list[float] = field(default_factory=list)
    drift_threshold_multiplier: float = 2.0

    def record_error(self, mse_value: float) -> None:
        """Append an MSE observation to the history.

        Args:
            mse_value: Non-negative MSE for the latest rebalance. Forced
                to ``float`` to make the storage type uniform regardless
                of caller arithmetic.
        """
        self.mse_history.append(float(mse_value))

    def is_drift_detected(self) -> bool:
        """Return ``True`` when the latest MSE exceeds the trailing baseline.

        The "baseline" excludes the most recent observation so a single
        spike is not absorbed into the baseline it is meant to be compared
        against. Drift is reported when::

            latest > mean(history[:-1]) * drift_threshold_multiplier

        Returns:
            ``False`` when fewer than ``10`` samples have been recorded or
            when the latest sample does not exceed the trailing baseline
            by ``drift_threshold_multiplier``.
        """
        if len(self.mse_history) < 10:
            # Below the minimum-history threshold the comparison is
            # unreliable -- any individual outlier would dominate the
            # baseline. Returning False avoids spurious alerts at startup.
            return False
        # The baseline deliberately excludes ``self.mse_history[-1]`` so
        # the spike being evaluated does not contribute to its own
        # comparison set.
        baseline = np.mean(self.mse_history[:-1])
        latest = self.mse_history[-1]
        return bool(latest > baseline * self.drift_threshold_multiplier)
