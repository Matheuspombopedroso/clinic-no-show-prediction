"""HTTP API tests against the plain FastAPI app (no Ray needed)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from noshow.api import create_app
from noshow.service import PredictionService


def _payload(appointment_id: str = "apt-1", prior: int = 4, **extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "appointment_id": appointment_id,
        "booked_at": "2025-03-03T09:00:00",
        "scheduled_at": "2025-03-10T14:30:00",
        "confirmed": False,
        "prior_appointments": prior,
        "prior_no_shows": 0,
        "prior_cancellations": 0,
        "days_since_last_visit": 30,
    }
    body.update(extra)
    return body


@pytest.fixture
def client(fake_service: PredictionService) -> TestClient:
    """TestClient for the plain FastAPI app wired to the fake service."""
    return TestClient(create_app(fake_service))


def test_health(client: TestClient) -> None:
    """/health answers 200 and reports the served model version."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": "test-0.0.0"}


def test_predict_single_response_shape(client: TestClient) -> None:
    """Predict single response shape."""
    resp = client.post("/predict", json=_payload(prior=1))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "appointment_id",
        "no_show_probability",
        "risk_level",
        "recommended_action",
        "model_version",
        "inference_time_ms",
    }
    assert body["appointment_id"] == "apt-1"
    assert body["no_show_probability"] == pytest.approx(0.30)
    assert body["risk_level"] == "medium"
    assert body["model_version"] == "test-0.0.0"
    assert body["inference_time_ms"] >= 0


def test_batch_preserves_order_and_levels(client: TestClient) -> None:
    """Batch preserves order and levels."""
    payload = {"appointments": [_payload("a", prior=0), _payload("b", prior=1), _payload("c", prior=2)]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert [p["appointment_id"] for p in preds] == ["a", "b", "c"]
    assert [p["risk_level"] for p in preds] == ["low", "medium", "high"]
    assert resp.json()["model_version"] == "test-0.0.0"


@pytest.mark.parametrize(
    "bad",
    [
        {"scheduled_at": "2025-03-01T09:00:00"},  # before booked_at
        {"prior_no_shows": 10},  # more no-shows than appointments
        {"prior_appointments": -1},
        {"booked_at": "not-a-date"},
    ],
)
def test_invalid_payload_returns_422(client: TestClient, bad: dict[str, object]) -> None:
    """Invalid payload returns 422."""
    assert client.post("/predict", json=_payload(**bad)).status_code == 422


def test_empty_batch_is_rejected(client: TestClient) -> None:
    """Empty batch is rejected."""
    assert client.post("/predict/batch", json={"appointments": []}).status_code == 422


def test_real_model_end_to_end(real_model) -> None:  # type: ignore[no-untyped-def]
    """A real trained model served through the API returns a valid probability and its version."""
    client = TestClient(create_app(PredictionService(real_model)))
    body = client.post("/predict", json=_payload()).json()
    assert 0.0 <= body["no_show_probability"] <= 1.0
    assert body["model_version"] == "9.9.9"
