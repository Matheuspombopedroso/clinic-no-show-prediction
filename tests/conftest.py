"""Shared pytest fixtures.

Most tests use a tiny model trained in a few seconds on freshly generated
synthetic data, so the suite does not depend on artifacts that are
git-ignored. A deterministic fake model is also provided for tests that need
exact, known probabilities (e.g. risk-level mapping through the API).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noshow.data import generate
from noshow.features import FEATURE_COLUMNS
from noshow.model import NoShowModel
from noshow.schemas import Appointment
from noshow.service import PredictionService


class FakeModel:
    """Returns a fixed probability per row, chosen by ``prior_appointments``.

    ``prior_appointments`` = 0 -> 0.05, 1 -> 0.30, >=2 -> 0.80, which makes it
    easy to hit each risk level from a test.
    """

    version = "test-0.0.0"

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[1] == len(FEATURE_COLUMNS)
        prior = features[:, FEATURE_COLUMNS.index("prior_appointments")]
        return np.select([prior == 0, prior == 1], [0.05, 0.30], default=0.80)


@pytest.fixture
def fake_service() -> PredictionService:
    return PredictionService(FakeModel())


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    return generate(n_patients=400, seed=7)


@pytest.fixture(scope="session")
def trained_artifact(tmp_path_factory: pytest.TempPathFactory, small_dataset: pd.DataFrame) -> Path:
    """Train a real (small) model end-to-end and return its artifact dir."""
    from noshow.train import train

    root = tmp_path_factory.mktemp("artifacts")
    csv = root / "appointments.csv"
    small_dataset.to_csv(csv, index=False)
    train(csv, version="9.9.9", out_root=root)
    return root / "9.9.9"


@pytest.fixture(scope="session")
def real_model(trained_artifact: Path) -> NoShowModel:
    return NoShowModel.load(trained_artifact)


def make_appointment(**overrides: object) -> Appointment:
    """Build a valid appointment; keyword arguments override defaults."""
    data: dict[str, object] = {
        "appointment_id": "apt-1",
        "booked_at": "2025-03-03T09:00:00",
        "scheduled_at": "2025-03-10T14:30:00",
        "confirmed": False,
        "prior_appointments": 4,
        "prior_no_shows": 1,
        "prior_cancellations": 0,
        "days_since_last_visit": 30.0,
    }
    data.update(overrides)
    return Appointment.model_validate(data)
