"""Synthetic appointment generator (fixed seed, no real patient data).

Why synthetic: the project is a public demo, so no real patient data may be
used. The generator still has to produce a *learnable* signal, otherwise the
model metrics mean nothing. It works like a small simulation:

1. Each synthetic patient gets hidden traits: ``reliability`` (how likely they
   are to show up) and ``cancel_propensity``.
2. Patients book appointments one after another over ~2 years. Each booking
   has a lead time, a weekday/hour slot and a confirmation status.
3. Each appointment ends as ``attended``, ``no_show`` or ``cancelled``. The
   no-show probability is a logistic function of the hidden reliability plus
   the observable features (lead time, slot, confirmation, ...) plus noise.
4. History counters (``prior_*``, ``days_since_last_visit``) are computed
   *as of booking time* from the patient's earlier appointments only, so the
   training data contains no information from the future.

The model never sees ``reliability`` directly; it has to infer it from the
patient's past no-show rate, which is how it works in a real clinic.

Run it with ``python -m noshow.data --out data/appointments.csv``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = datetime(2024, 1, 1)
END = datetime(2025, 12, 31)

# Clinic opening hours used to draw appointment slots.
OPEN_HOURS = np.arange(7, 19)  # 07:00 .. 18:00 starts
# Mon..Sat; Saturday has fewer slots, Sunday the clinic is closed.
WEEKDAY_WEIGHTS = np.array([0.19, 0.19, 0.18, 0.18, 0.17, 0.09, 0.0])

# Extra no-show log-odds per weekday (Mon=0). Monday/Friday/Saturday are worse.
WEEKDAY_EFFECT = np.array([0.25, 0.0, -0.05, 0.0, 0.3, 0.4, 0.0])


def _sigmoid(x: float) -> float:
    """Logistic function, used to turn log-odds into probabilities."""
    return float(1.0 / (1.0 + np.exp(-x)))


def _hour_effect(hour: int) -> float:
    """Early-morning and end-of-day slots get extra no-show log-odds."""
    if hour <= 8:
        return 0.3
    if hour >= 17:
        return 0.25
    return 0.0


def _draw_slot(rng: np.random.Generator, booked_at: datetime) -> datetime:
    """Draw a scheduled start time after ``booked_at`` (lead time + valid slot)."""
    # Mixture: ~25% short-notice bookings (0-2 days), the rest roughly
    # exponential with a 14-day mean, capped at 90 days.
    short_notice = rng.random() < 0.25
    lead_days = rng.uniform(0.0, 2.0) if short_notice else min(rng.exponential(14.0), 90.0)
    day = (booked_at + timedelta(days=lead_days)).date()
    # Move to a weekday the clinic is open, respecting the weekday weights.
    target_weekday = int(rng.choice(7, p=WEEKDAY_WEIGHTS))
    day += timedelta(days=(target_weekday - day.weekday()) % 7)
    hour = int(rng.choice(OPEN_HOURS))
    minute = int(rng.choice([0, 30]))
    slot = datetime(day.year, day.month, day.day, hour, minute)
    # Same-day bookings can land in the past hour; push them a week forward.
    if slot < booked_at:
        slot += timedelta(days=7)
    return slot


def generate(n_patients: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Generate the synthetic appointment table.

    Args:
        n_patients: Number of synthetic patients to simulate.
        seed: Random seed; the same seed always yields the same table.

    Returns:
        DataFrame with one row per appointment, sorted by ``scheduled_at``.
        ``status`` is ``attended`` / ``no_show`` / ``cancelled`` and
        ``no_show`` is the binary label (1 only for ``no_show``).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    horizon_days = (END - START).days

    for pid in range(n_patients):
        # Hidden traits: never exported, only their effects are observable.
        reliability = rng.normal(0.0, 1.0)
        cancel_propensity = rng.normal(0.0, 0.7)
        # Frequent visitors (chronic follow-up) vs occasional ones.
        mean_gap_days = rng.choice([20.0, 45.0, 90.0], p=[0.3, 0.4, 0.3])

        n_prior = n_no_show = n_cancel = 0
        last_attended: datetime | None = None
        cursor = START + timedelta(days=float(rng.uniform(0, horizon_days * 0.6)))
        cursor = cursor.replace(second=0, microsecond=0)  # whole minutes, like real booking systems

        while True:
            booked_at = cursor + timedelta(days=float(rng.exponential(mean_gap_days)))
            booked_at = booked_at.replace(
                hour=int(rng.integers(8, 20)), minute=int(rng.integers(0, 60)), second=0, microsecond=0
            )
            # Changing the clock time can move the booking a few hours before
            # the previous appointment; a patient books the next visit only
            # after the previous one, so clamp to keep the timeline ordered.
            booked_at = max(booked_at, cursor)
            scheduled_at = _draw_slot(rng, booked_at)
            if scheduled_at > END:
                break
            lead_days = (scheduled_at - booked_at).total_seconds() / 86_400.0
            days_since = (booked_at - last_attended).total_seconds() / 86_400.0 if last_attended else None

            # Confirmation: reliable patients confirm more; long lead times
            # make the reminder less effective.
            confirmed = rng.random() < _sigmoid(0.2 + 0.8 * reliability - 0.015 * lead_days)

            cancel_rate = (n_cancel + 0.2) / (n_prior + 2.0)
            logit = (
                -2.5
                - 0.9 * reliability  # hidden trait, visible via past no-shows
                + 0.35 * np.log1p(lead_days)  # long lead time -> forgets / plans change
                + WEEKDAY_EFFECT[scheduled_at.weekday()]
                + _hour_effect(scheduled_at.hour)
                - 1.2 * confirmed  # strongest protective signal
                + 1.2 * cancel_rate  # habitual cancellers also skip more
                + 0.003 * min(days_since if days_since is not None else 180.0, 365.0)
                + (0.35 if n_prior == 0 else 0.0)  # first visit: weaker commitment
                + rng.normal(0.0, 0.5)  # irreducible noise
            )
            p_cancel = _sigmoid(-2.6 + cancel_propensity + 0.015 * lead_days)

            if rng.random() < p_cancel:
                status = "cancelled"
            elif rng.random() < _sigmoid(logit):
                status = "no_show"
            else:
                status = "attended"

            rows.append(
                {
                    "appointment_id": f"apt-{len(rows):07d}",
                    "patient_id": f"pat-{pid:05d}",
                    "booked_at": booked_at,
                    "scheduled_at": scheduled_at,
                    "confirmed": bool(confirmed),
                    # History strictly before this booking (no leakage).
                    "prior_appointments": n_prior,
                    "prior_no_shows": n_no_show,
                    "prior_cancellations": n_cancel,
                    "days_since_last_visit": days_since,
                    "status": status,
                    "no_show": int(status == "no_show"),
                }
            )

            # Update the running history *after* writing the row.
            n_prior += 1
            n_no_show += status == "no_show"
            n_cancel += status == "cancelled"
            if status == "attended":
                last_attended = scheduled_at
            cursor = scheduled_at

    df = pd.DataFrame(rows)
    return df.sort_values("scheduled_at", kind="stable").reset_index(drop=True)


def main() -> None:
    """CLI entry point: generate the dataset and write it as CSV."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/appointments.csv"))
    parser.add_argument("--patients", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = generate(args.patients, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    scored = df[df["status"] != "cancelled"]
    print(f"wrote {len(df)} appointments ({len(scored)} non-cancelled) to {args.out}")
    print(df["status"].value_counts(normalize=True).round(3).to_string())
    print(f"no-show rate among non-cancelled: {scored['no_show'].mean():.3f}")


if __name__ == "__main__":
    main()
