"""Ray Serve deployment: FastAPI ingress, dynamic batching, N replicas.

What Ray Serve adds on top of :class:`~noshow.service.PredictionService`:

* **Replicas** - ``num_replicas`` copies of :class:`NoShowDeployment`, each a
  separate process holding its own copy of the model. The Serve HTTP proxy
  load-balances requests across them. This is the horizontal scaling knob.
* **Dynamic request batching** - ``@serve.batch`` collects single ``/predict``
  calls that arrive within ``batch_wait_timeout_s`` (up to ``max_batch_size``)
  and scores them with one vectorised model call. Per-request Python overhead
  (pandas frame, XGBoost call) is paid once per batch instead of per request.
* **FastAPI ingress** - ``@serve.ingress`` mounts a normal FastAPI app, so
  request validation, OpenAPI docs and response models work unchanged.

``/predict/batch`` does *not* go through ``@serve.batch``: the client already
sent many rows, so it is scored directly in one call.

Start it with ``python -m noshow.serve_app`` (reads env vars, see
``.env.example``) or ``serve run serve_config.yaml``.
"""

# NOTE: no ``from __future__ import annotations`` here on purpose. Ray Serve
# rebuilds the ingress methods for FastAPI, and with postponed (string)
# annotations FastAPI can no longer resolve ``Appointment`` as the request
# body model, so every POST would fail with 422.

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from ray import serve
from ray.serve import Application

from noshow.api import API_DESCRIPTION, API_TITLE, batch_response
from noshow.config import ServeSettings
from noshow.model import NoShowModel
from noshow.schemas import Appointment, BatchRequest, BatchResponse, Prediction
from noshow.service import PredictionService

fastapi_app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version="0.1.0")


@serve.deployment
@serve.ingress(fastapi_app)
class NoShowDeployment:
    """One model replica. Ray creates ``num_replicas`` instances of this class."""

    def __init__(
        self,
        model_dir: str,
        model_threads: int = 1,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.005,
    ) -> None:
        """Load the model once per replica and configure the batcher.

        Args:
            model_dir: Artifact directory (``model.json`` + ``metadata.json``).
            model_threads: XGBoost threads per replica.
            max_batch_size: Max requests merged into one model call.
            batch_wait_timeout_s: Max time a request waits for a batch to fill.
        """
        model = NoShowModel.load(Path(model_dir), n_threads=model_threads)
        self.service = PredictionService(model)
        # ``@serve.batch`` parameters can be changed at runtime; setting them
        # here lets them come from configuration instead of being hard-coded
        # in the decorator.
        self._predict_batched.set_max_batch_size(max_batch_size)
        self._predict_batched.set_batch_wait_timeout_s(batch_wait_timeout_s)

    @fastapi_app.get("/health")
    def health(self) -> dict[str, str]:
        """Liveness/readiness probe; also reports the served model version."""
        return {"status": "ok", "model_version": self.service.model_version}

    @fastapi_app.post("/predict", response_model=Prediction)
    async def predict(self, appointment: Appointment) -> Prediction:
        """Score one appointment; transparently merged with concurrent calls."""
        # Timestamp taken before queueing so inference_time_ms includes the
        # time this request spent waiting for its batch to be formed.
        return await self._predict_batched((appointment, time.perf_counter()))

    @fastapi_app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(self, request: BatchRequest) -> BatchResponse:
        """Score many appointments in one vectorised call (no extra batching)."""
        return batch_response(self.service, request)

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.005)
    async def _predict_batched(self, items: list[tuple[Appointment, float]]) -> list[Prediction]:
        """Called by Serve with a *list* of queued requests; returns one result each."""
        appointments = [appointment for appointment, _ in items]
        start_times = [started for _, started in items]
        return self.service.predict_with_start_times(appointments, start_times)


def build_app(settings: ServeSettings) -> Application:
    """Bind the deployment with the replica count and batching settings."""
    return NoShowDeployment.options(
        num_replicas=settings.num_replicas,
        # A replica must accept at least a full batch of in-flight requests,
        # otherwise the batcher can never collect max_batch_size items.
        max_ongoing_requests=max(settings.max_ongoing_requests, settings.max_batch_size),
        # One logical CPU reserved per replica, matching model_threads=1. Ray
        # refuses to schedule more replicas than the machine has CPUs for.
        ray_actor_options={"num_cpus": 1},
        logging_config=settings.logging_config(),
    ).bind(
        model_dir=str(settings.model_dir.resolve()),
        model_threads=settings.model_threads,
        max_batch_size=settings.max_batch_size,
        batch_wait_timeout_s=settings.batch_wait_timeout_s,
    )


def app_builder(args: dict[str, Any]) -> Application:
    """Entry point for ``serve run`` / ``serve_config.yaml``.

    ``args`` come from the YAML ``args:`` block or ``key=value`` CLI pairs and
    override the environment-based defaults.
    """
    base = ServeSettings.from_env()
    overrides = {
        "model_dir": Path(args["model_dir"]) if "model_dir" in args else base.model_dir,
        "num_replicas": int(args.get("num_replicas", base.num_replicas)),
        "max_batch_size": int(args.get("max_batch_size", base.max_batch_size)),
        "batch_wait_timeout_s": float(args.get("batch_wait_timeout_s", base.batch_wait_timeout_s)),
        "max_ongoing_requests": int(args.get("max_ongoing_requests", base.max_ongoing_requests)),
        "model_threads": int(args.get("model_threads", base.model_threads)),
    }
    return build_app(ServeSettings(**{**base.__dict__, **overrides}))


def main() -> None:
    """Start Serve with settings from the environment and block forever."""
    settings = ServeSettings.from_env()
    serve.start(
        http_options={"host": settings.host, "port": settings.port},
        logging_config=settings.logging_config(),  # also silences the proxy access log
    )
    serve.run(build_app(settings), name="no-show", route_prefix="/", blocking=True)


if __name__ == "__main__":
    main()
