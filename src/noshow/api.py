"""Plain FastAPI app with the same routes as the Ray Serve deployment.

Why this exists next to :mod:`noshow.serve_app`:

* the API tests use FastAPI's ``TestClient`` against this app, so they run in
  CI in about a second without starting a Ray cluster;
* it is a simple single-process fallback (``uvicorn noshow.api:app``) that is
  useful for debugging and as the "no Ray" baseline.

The route bodies are deliberately tiny: all logic lives in
:class:`~noshow.service.PredictionService`, which the Serve deployment uses too.
"""

from __future__ import annotations

import time

from fastapi import FastAPI

from noshow.config import ServeSettings
from noshow.model import NoShowModel
from noshow.schemas import Appointment, BatchRequest, BatchResponse, Prediction
from noshow.service import PredictionService

API_TITLE = "Clinic No-Show Prediction"
API_DESCRIPTION = (
    "Demo service with synthetic data. Returns the probability that a patient "
    "misses an appointment. Not a medical device; makes no clinical decisions."
)


def batch_response(service: PredictionService, request: BatchRequest) -> BatchResponse:
    """Score a whole batch with one vectorised model call (shared helper)."""
    start = time.perf_counter()
    predictions = service.predict(request.appointments, started_at=start)
    return BatchResponse(
        predictions=predictions,
        model_version=service.model_version,
        inference_time_ms=round((time.perf_counter() - start) * 1000.0, 3),
    )


def create_app(service: PredictionService) -> FastAPI:
    """Build a FastAPI app bound to an already-constructed service.

    Dependency injection by argument keeps tests simple: they pass a service
    wrapping either the real model or a deterministic fake.
    """
    app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness/readiness probe; also reports the served model version."""
        return {"status": "ok", "model_version": service.model_version}

    @app.post("/predict", response_model=Prediction)
    def predict(appointment: Appointment) -> Prediction:
        """Score a single appointment."""
        return service.predict([appointment])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest) -> BatchResponse:
        """Score many appointments in one call."""
        return batch_response(service, request)

    return app


def _default_app() -> FastAPI:
    """Load the model from ``MODEL_DIR`` and build the app (for ``uvicorn``)."""
    settings = ServeSettings.from_env()
    model = NoShowModel.load(settings.model_dir, n_threads=settings.model_threads)
    return create_app(PredictionService(model))


def __getattr__(name: str) -> FastAPI:
    """Lazily build ``app`` so importing this module never requires a model file."""
    if name == "app":
        return _default_app()
    raise AttributeError(name)
