"""End-to-end test against a real local Ray Serve deployment.

Marked ``integration`` and skipped by default (starting Ray takes ~10-30 s).
Run with: ``pytest -m integration``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from bench.payloads import appointment_pool

pytestmark = pytest.mark.integration

PORT = 8011
URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def serve_url(trained_artifact: Path):  # type: ignore[no-untyped-def]
    """Start a local 2-replica Ray Serve app for this module and yield its URL."""
    ray = pytest.importorskip("ray")
    from ray import serve

    from noshow.config import ServeSettings
    from noshow.serve_app import build_app

    ray.init(include_dashboard=False, logging_level="ERROR")
    serve.start(http_options={"host": "127.0.0.1", "port": PORT})
    settings = ServeSettings(model_dir=trained_artifact, num_replicas=2, port=PORT)
    serve.run(build_app(settings), name="no-show-test", route_prefix="/")
    yield URL
    serve.shutdown()
    ray.shutdown()


def test_single_and_batch(serve_url: str) -> None:
    """/predict and /predict/batch work through a real Serve deployment and keep ids in order."""
    pool = appointment_pool()
    single = httpx.post(f"{serve_url}/predict", json=pool[0], timeout=30).json()
    assert single["appointment_id"] == pool[0]["appointment_id"]
    assert single["model_version"] == "9.9.9"
    assert single["risk_level"] in {"low", "medium", "high"}

    batch = httpx.post(f"{serve_url}/predict/batch", json={"appointments": pool[:20]}, timeout=30).json()
    assert [p["appointment_id"] for p in batch["predictions"]] == [a["appointment_id"] for a in pool[:20]]


def test_concurrent_singles_are_batched_correctly(serve_url: str) -> None:
    """Concurrent /predict calls go through @serve.batch; each must get its own answer."""
    pool = appointment_pool()[:64]

    async def fire() -> list[dict[str, object]]:
        async with httpx.AsyncClient(base_url=serve_url, timeout=30) as client:
            responses = await asyncio.gather(*(client.post("/predict", json=a) for a in pool))
        return [r.json() for r in responses]

    results = asyncio.run(fire())
    assert [r["appointment_id"] for r in results] == [a["appointment_id"] for a in pool]

    # Same appointments scored via /predict/batch must give identical probabilities.
    batch = httpx.post(f"{serve_url}/predict/batch", json={"appointments": pool}, timeout=30).json()
    by_id = {p["appointment_id"]: p["no_show_probability"] for p in batch["predictions"]}
    for r in results:
        assert r["no_show_probability"] == pytest.approx(by_id[r["appointment_id"]], abs=1e-6)
