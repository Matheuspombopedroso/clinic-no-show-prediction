"""Unit tests for feature engineering (shared by training and serving)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from noshow.features import (
    FEATURE_COLUMNS,
    PRIOR_NO_SHOW_RATE,
    RAW_COLUMNS,
    appointments_to_frame,
    build_features,
    smoothed_rate,
)
from tests.conftest import make_appointment


def _features(**overrides: object) -> dict[str, float]:
    frame = build_features(appointments_to_frame([make_appointment(**overrides)]))
    return frame.iloc[0].to_dict()


def test_columns_and_order_are_stable() -> None:
    frame = build_features(appointments_to_frame([make_appointment()]))
    assert list(frame.columns) == FEATURE_COLUMNS


def test_lead_time_weekday_and_hour() -> None:
    f = _features(booked_at="2025-03-03T09:00:00", scheduled_at="2025-03-10T14:30:00")
    assert f["lead_time_days"] == pytest.approx(7 + 5.5 / 24)
    assert f["weekday"] == 0  # 2025-03-10 is a Monday
    assert f["hour"] == 14


def test_same_day_booking_has_fractional_lead_time() -> None:
    f = _features(booked_at="2025-03-10T08:00:00", scheduled_at="2025-03-10T14:00:00")
    assert f["lead_time_days"] == pytest.approx(0.25)


def test_new_patient_gets_prior_rate_and_missing_recency() -> None:
    f = _features(prior_appointments=0, prior_no_shows=0, prior_cancellations=0, days_since_last_visit=None)
    assert f["is_new_patient"] == 1
    assert f["prior_no_show_rate"] == pytest.approx(PRIOR_NO_SHOW_RATE)
    assert math.isnan(f["days_since_last_visit"])


def test_smoothed_rate_moves_toward_observed_rate_with_history() -> None:
    few = smoothed_rate(np.array([1]), np.array([1]), 0.2)[0]
    many = smoothed_rate(np.array([100]), np.array([100]), 0.2)[0]
    assert 0.2 < few < many < 1.0
    assert many == pytest.approx(1.0, abs=0.02)


def test_confirmed_is_binary() -> None:
    assert _features(confirmed=True)["confirmed"] == 1
    assert _features(confirmed=False)["confirmed"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"booked_at": "2025-03-11T00:00:00", "scheduled_at": "2025-03-10T00:00:00"},
        {"prior_appointments": 1, "prior_no_shows": 1, "prior_cancellations": 1},
        {"prior_appointments": -1},
    ],
)
def test_invalid_inputs_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_appointment(**overrides)


def test_generated_history_has_no_leakage(small_dataset) -> None:  # type: ignore[no-untyped-def]
    """prior_* counters must equal the count of the patient's earlier rows."""
    for _, group in small_dataset.groupby("patient_id"):
        group = group.sort_values("scheduled_at")
        assert group["prior_appointments"].tolist() == list(range(len(group)))
        expected_no_shows = group["no_show"].cumsum().shift(fill_value=0).tolist()
        assert group["prior_no_shows"].tolist() == expected_no_shows


def test_generator_is_deterministic() -> None:
    from noshow.data import generate

    a, b = generate(n_patients=50, seed=1), generate(n_patients=50, seed=1)
    assert a.equals(b)


def test_every_generated_row_is_a_valid_api_request(small_dataset) -> None:  # type: ignore[no-untyped-def]
    """The generator must never emit rows the API would reject (e.g. negative recency)."""
    from noshow.schemas import Appointment

    cols = [
        "appointment_id",
        "booked_at",
        "scheduled_at",
        "confirmed",
        "prior_appointments",
        "prior_no_shows",
        "prior_cancellations",
        "days_since_last_visit",
    ]
    for row in small_dataset[cols].to_dict("records"):
        days = row["days_since_last_visit"]
        row["days_since_last_visit"] = None if days != days else days
        Appointment.model_validate(row)


def test_fast_path_matches_pandas_features(small_dataset) -> None:  # type: ignore[no-untyped-def]
    """Serving (NumPy) and training (pandas) features must be identical."""
    from noshow.features import appointments_to_matrix
    from noshow.schemas import Appointment

    rows = small_dataset.head(3000)
    appointments = []
    for row in rows[["appointment_id", *RAW_COLUMNS]].to_dict("records"):
        days = row["days_since_last_visit"]
        row["days_since_last_visit"] = None if days != days else days
        appointments.append(Appointment.model_validate(row))
    fast = appointments_to_matrix(appointments)
    slow = build_features(rows).to_numpy(dtype=np.float32)
    np.testing.assert_allclose(fast, slow, rtol=1e-6, equal_nan=True)
