"""Request and response contracts of the HTTP API (Pydantic models).

These models are the public interface of the service. They are shared by the
plain FastAPI app used in tests and by the Ray Serve ingress, so both expose
exactly the same JSON shapes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RiskLevel = Literal["low", "medium", "high"]


class Appointment(BaseModel):
    """One scheduled appointment plus the patient's history as of booking time.

    The history counters must describe the patient *before* this appointment
    was booked; sending counts that include the appointment itself would leak
    the answer, exactly as it would during training.
    """

    appointment_id: str = Field(..., examples=["apt-000123"])
    booked_at: datetime = Field(..., description="When the appointment was booked.")
    scheduled_at: datetime = Field(..., description="Start time of the appointment.")
    confirmed: bool = Field(False, description="Patient confirmed via reminder.")
    prior_appointments: int = Field(0, ge=0, description="Past appointments (any outcome).")
    prior_no_shows: int = Field(0, ge=0, description="Past appointments missed.")
    prior_cancellations: int = Field(0, ge=0, description="Past appointments cancelled.")
    days_since_last_visit: float | None = Field(
        None, ge=0, description="Days since last attended visit; null for new patients."
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> Appointment:
        """Reject impossible inputs instead of silently scoring them."""
        if (self.booked_at.tzinfo is None) != (self.scheduled_at.tzinfo is None):
            raise ValueError("booked_at and scheduled_at must both have, or both omit, a timezone")
        if self.scheduled_at < self.booked_at:
            raise ValueError("scheduled_at must not be earlier than booked_at")
        if self.prior_no_shows + self.prior_cancellations > self.prior_appointments:
            raise ValueError("prior_no_shows + prior_cancellations cannot exceed prior_appointments")
        return self


class BatchRequest(BaseModel):
    """Several appointments scored in one HTTP call (e.g. tomorrow's agenda)."""

    appointments: list[Appointment] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    """Model output for one appointment."""

    appointment_id: str
    no_show_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    recommended_action: str
    model_version: str
    inference_time_ms: float = Field(
        ...,
        description=(
            "Server-side wall time inside the model replica, from the moment the "
            "handler received the request until the prediction was ready. For "
            "/predict it includes the time spent waiting in the dynamic batch."
        ),
    )


class BatchResponse(BaseModel):
    """Response of ``POST /predict/batch``."""

    predictions: list[Prediction]
    model_version: str
    inference_time_ms: float
