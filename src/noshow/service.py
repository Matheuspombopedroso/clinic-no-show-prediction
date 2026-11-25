"""Framework-agnostic prediction logic (no Ray, no HTTP).

:class:`PredictionService` takes validated :class:`~noshow.schemas.Appointment`
objects and returns :class:`~noshow.schemas.Prediction` objects. It is used by:

* :mod:`noshow.api` - a plain FastAPI app (used by the API tests and handy for
  local debugging without Ray);
* :mod:`noshow.serve_app` - the Ray Serve deployment, which only adds
  replication and dynamic batching around the same calls.

Keeping the logic here means the HTTP layer and the Serve layer cannot drift
apart, and unit tests do not need a Ray cluster.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from noshow.features import appointments_to_matrix
from noshow.risk import recommended_action, risk_level
from noshow.schemas import Appointment, Prediction


class ProbabilityModel(Protocol):
    """Anything with a version and a ``predict_proba`` (real model or a test fake)."""

    @property
    def version(self) -> str:
        """Model version string reported in responses."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one no-show probability per feature row."""


class PredictionService:
    """Turns appointments into predictions using a loaded model."""

    def __init__(self, model: ProbabilityModel) -> None:
        """Wrap a loaded model (real or fake) that will answer all predictions."""
        self.model = model

    @property
    def model_version(self) -> str:
        """Version of the model currently being served."""
        return self.model.version

    def predict_probabilities(self, appointments: Sequence[Appointment]) -> np.ndarray:
        """Vectorised scoring: one feature build and one model call for all rows."""
        return self.model.predict_proba(appointments_to_matrix(appointments))

    def predict(
        self, appointments: Sequence[Appointment], started_at: float | None = None
    ) -> list[Prediction]:
        """Score appointments and wrap each probability in a :class:`Prediction`.

        Args:
            appointments: Validated appointments.
            started_at: ``time.perf_counter()`` value taken when the request
                entered the server. If omitted, timing starts now. Passing it
                lets callers include queueing/batching time in
                ``inference_time_ms``.

        Returns:
            One prediction per appointment, in input order.
        """
        start = time.perf_counter() if started_at is None else started_at
        return self.predict_with_start_times(appointments, [start] * len(appointments))

    def predict_with_start_times(
        self, appointments: Sequence[Appointment], start_times: Sequence[float]
    ) -> list[Prediction]:
        """Score appointments that arrived at different moments.

        Used by the Ray Serve dynamic batcher: a batch merges requests that
        arrived a few milliseconds apart, and each one should report its own
        server-side latency (including the time it waited for the batch).

        Args:
            appointments: Validated appointments.
            start_times: One ``time.perf_counter()`` value per appointment.

        Returns:
            One prediction per appointment, in input order.
        """
        probabilities = self.predict_probabilities(appointments)
        now = time.perf_counter()
        return [
            self._to_prediction(appt.appointment_id, float(p), (now - start) * 1000.0)
            for appt, p, start in zip(appointments, probabilities, start_times, strict=True)
        ]

    def _to_prediction(self, appointment_id: str, probability: float, elapsed_ms: float) -> Prediction:
        """Attach risk level, action and metadata to a raw probability."""
        # Clip guards against tiny float overshoot (e.g. 1.0000001) so the
        # risk function's range check never fails on valid model output.
        probability = min(max(probability, 0.0), 1.0)
        level = risk_level(probability)
        return Prediction(
            appointment_id=appointment_id,
            no_show_probability=round(probability, 6),
            risk_level=level,
            recommended_action=recommended_action(level),
            model_version=self.model_version,
            inference_time_ms=round(elapsed_ms, 3),
        )
