"""Locust load test (run headless) for ``/predict`` and ``/predict/batch``.

Select the scenario with the user class name, for example::

    locust -f bench/locustfile.py --headless -u 32 -r 32 -t 30s \
        --host http://127.0.0.1:8000 --csv results/locust_single SingleUser

Users have no wait time (closed loop, like ``bench/loadtest.py``), so the
reported throughput is the maximum the server sustains for that number of
concurrent users. Locust writes p50/p95/p99 to ``*_stats.csv``.
"""

from __future__ import annotations

import os
import random

from locust import FastHttpUser, task

from bench.payloads import appointment_pool, batch_payload, single_payload

BATCH_SIZE = int(os.getenv("LOCUST_BATCH_SIZE", "50"))
appointment_pool()  # build payloads once, at import, not during the test


class SingleUser(FastHttpUser):
    """Sends one appointment per request (exercises ``@serve.batch``)."""

    def on_start(self) -> None:
        self.rng = random.Random(id(self))

    @task
    def predict(self) -> None:
        self.client.post("/predict", json=single_payload(self.rng), name="/predict")


class BatchUser(FastHttpUser):
    """Sends ``LOCUST_BATCH_SIZE`` appointments per request."""

    def on_start(self) -> None:
        self.rng = random.Random(id(self))

    @task
    def predict_batch(self) -> None:
        self.client.post("/predict/batch", json=batch_payload(self.rng, BATCH_SIZE), name="/predict/batch")
