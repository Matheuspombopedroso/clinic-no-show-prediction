"""Request payloads for the load tests, drawn from the synthetic generator.

Using varied, realistic rows (instead of one payload repeated) keeps the
benchmark honest: every request goes through validation, feature building and
the model with different values. Shared by ``bench/loadtest.py`` and
``bench/locustfile.py``.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from noshow.data import generate

PAYLOAD_FIELDS = [
    "appointment_id",
    "booked_at",
    "scheduled_at",
    "confirmed",
    "prior_appointments",
    "prior_no_shows",
    "prior_cancellations",
    "days_since_last_visit",
]


@lru_cache(maxsize=1)
def appointment_pool(size: int = 2000, seed: int = 123) -> list[dict[str, Any]]:
    """Return ``size`` JSON-ready appointment dicts (deterministic for a seed)."""
    df = generate(n_patients=300, seed=seed).head(size)
    pool: list[dict[str, Any]] = []
    for row in df[PAYLOAD_FIELDS].to_dict("records"):
        row["booked_at"] = row["booked_at"].isoformat()
        row["scheduled_at"] = row["scheduled_at"].isoformat()
        days = row["days_since_last_visit"]
        # NaN is not valid JSON; the API expects null for new patients.
        row["days_since_last_visit"] = None if days != days else float(days)
        pool.append(row)
    return pool


def single_payload(rng: random.Random) -> dict[str, Any]:
    """One appointment for ``POST /predict``."""
    return rng.choice(appointment_pool())


def batch_payload(rng: random.Random, size: int) -> dict[str, Any]:
    """``size`` appointments for ``POST /predict/batch``."""
    return {"appointments": rng.sample(appointment_pool(), size)}
