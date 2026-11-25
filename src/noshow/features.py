"""Feature engineering shared by training and serving.

A classic source of train/serve skew is computing features one way in a
notebook and another way in the API. Here all feature logic lives in one
module with two entry points:

* :func:`build_features` - vectorised pandas version used for training and
  evaluation on tens of thousands of rows;
* :func:`appointments_to_matrix` - allocation-light NumPy version used by the
  online service, where pandas overhead dominates latency for small inputs.

A parity test asserts both produce identical features on generated data, so
the optimisation cannot silently change model inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from noshow.schemas import Appointment

# Column order matters for XGBoost: the model is trained and served with
# exactly this list, and it is stored in the model metadata as a safety check.
FEATURE_COLUMNS: list[str] = [
    "lead_time_days",
    "weekday",
    "hour",
    "confirmed",
    "prior_appointments",
    "prior_no_show_rate",
    "prior_cancellation_rate",
    "days_since_last_visit",
    "is_new_patient",
]

# Raw columns that :func:`build_features` expects in its input frame.
RAW_COLUMNS: list[str] = [
    "booked_at",
    "scheduled_at",
    "confirmed",
    "prior_appointments",
    "prior_no_shows",
    "prior_cancellations",
    "days_since_last_visit",
]

# Beta prior used to smooth rates for patients with little history.
# With 0 past visits the smoothed no-show rate equals PRIOR_NO_SHOW_RATE
# instead of an undefined 0/0; after many visits the patient's own rate wins.
PRIOR_STRENGTH = 2.0
PRIOR_NO_SHOW_RATE = 0.2
PRIOR_CANCELLATION_RATE = 0.1


def smoothed_rate(events: np.ndarray, total: np.ndarray, prior_rate: float) -> np.ndarray:
    """Return ``(events + k * prior_rate) / (total + k)`` (Bayesian smoothing).

    Args:
        events: Count of events (e.g. no-shows) per row.
        total: Count of opportunities (e.g. past appointments) per row.
        prior_rate: Rate assumed when there is no history.

    Returns:
        Array of smoothed rates in ``[0, 1]``.
    """
    events = np.asarray(events, dtype=float)
    total = np.asarray(total, dtype=float)
    return (events + PRIOR_STRENGTH * prior_rate) / (total + PRIOR_STRENGTH)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the model feature matrix from raw appointment columns.

    Args:
        raw: DataFrame containing at least :data:`RAW_COLUMNS`. ``booked_at``
            and ``scheduled_at`` may be strings or datetimes.

    Returns:
        DataFrame with columns :data:`FEATURE_COLUMNS` (float dtype), one row
        per input row, in the same order.
    """
    # format="ISO8601" accepts both "2025-03-10T14:30:00" and
    # "2025-03-10 14:30:00.5"; without it pandas infers the format from the
    # first row and fails on rows written differently.
    booked = pd.to_datetime(raw["booked_at"], format="ISO8601")
    scheduled = pd.to_datetime(raw["scheduled_at"], format="ISO8601")
    prior = raw["prior_appointments"].to_numpy(dtype=float)

    feats = pd.DataFrame(index=raw.index)
    # Lead time in fractional days: same-day bookings get values < 1.
    feats["lead_time_days"] = (scheduled - booked).dt.total_seconds() / 86_400.0
    # Integer weekday (Mon=0) and hour; trees split on these directly, so no
    # one-hot encoding is needed.
    feats["weekday"] = scheduled.dt.day_of_week
    feats["hour"] = scheduled.dt.hour
    feats["confirmed"] = raw["confirmed"].astype(int)
    feats["prior_appointments"] = prior
    feats["prior_no_show_rate"] = smoothed_rate(raw["prior_no_shows"].to_numpy(), prior, PRIOR_NO_SHOW_RATE)
    feats["prior_cancellation_rate"] = smoothed_rate(
        raw["prior_cancellations"].to_numpy(), prior, PRIOR_CANCELLATION_RATE
    )
    # Missing for new patients: XGBoost learns a default branch for NaN, which
    # is better than inventing a sentinel like -1 that has a numeric meaning.
    feats["days_since_last_visit"] = pd.to_numeric(raw["days_since_last_visit"], errors="coerce")
    feats["is_new_patient"] = (prior == 0).astype(int)
    return feats[FEATURE_COLUMNS].astype(float)


def appointments_to_frame(appointments: Iterable[Appointment]) -> pd.DataFrame:
    """Convert validated API objects into the raw frame :func:`build_features` expects."""
    rows = [a.model_dump(include=set(RAW_COLUMNS)) for a in appointments]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def appointments_to_matrix(appointments: Sequence[Appointment]) -> np.ndarray:
    """Serving fast path: compute the same features as :func:`build_features` without pandas.

    Why a second implementation: for the 1-50 rows of an online request,
    building a pandas DataFrame and parsing datetimes costs more than the
    XGBoost call itself. This function computes each feature with plain
    Python/NumPy, in the order of :data:`FEATURE_COLUMNS`.

    The risk of having two implementations is train/serve skew. It is
    contained by ``test_fast_path_matches_pandas_features``, which checks that
    both functions return identical values on thousands of generated rows.

    Args:
        appointments: Validated appointments.

    Returns:
        ``float32`` array of shape ``(len(appointments), len(FEATURE_COLUMNS))``.
    """
    out = np.empty((len(appointments), len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, a in enumerate(appointments):
        prior = float(a.prior_appointments)
        out[i] = (
            (a.scheduled_at - a.booked_at).total_seconds() / 86_400.0,
            a.scheduled_at.weekday(),
            a.scheduled_at.hour,
            1.0 if a.confirmed else 0.0,
            prior,
            (a.prior_no_shows + PRIOR_STRENGTH * PRIOR_NO_SHOW_RATE) / (prior + PRIOR_STRENGTH),
            (a.prior_cancellations + PRIOR_STRENGTH * PRIOR_CANCELLATION_RATE) / (prior + PRIOR_STRENGTH),
            np.nan if a.days_since_last_visit is None else a.days_since_last_visit,
            1.0 if prior == 0 else 0.0,
        )
    return out
