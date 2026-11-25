"""Closed-loop async load generator reporting p50/p95/p99 latency and throughput.

``concurrency`` virtual clients each send a request, wait for the response and
immediately send the next one, for ``duration`` seconds (after a short warm-up
that is not recorded). Latency is measured client-side, end to end (HTTP
round trip through the Ray Serve proxy). The server-reported
``inference_time_ms`` is collected too, so the two can be compared.

Usage (server must already be running)::

    python -m bench.loadtest --mode single --concurrency 32 --duration 20
    python -m bench.loadtest --mode batch --batch-size 50 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import numpy as np

from bench.payloads import appointment_pool, batch_payload, single_payload


@dataclass
class LoadResult:
    """Summary of one load-test run."""

    mode: str
    concurrency: int
    batch_size: int
    duration_s: float
    requests: int
    errors: int
    requests_per_s: float
    predictions_per_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_server_inference_ms: float


async def _worker(
    client: httpx.AsyncClient,
    mode: str,
    batch_size: int,
    stop_at: float,
    record_after: float,
    latencies: list[float],
    server_ms: list[float],
    errors: list[int],
    seed: int,
) -> None:
    """One virtual client: send requests back-to-back until ``stop_at``."""
    rng = random.Random(seed)
    path = "/predict" if mode == "single" else "/predict/batch"
    while (now := time.perf_counter()) < stop_at:
        body = single_payload(rng) if mode == "single" else batch_payload(rng, batch_size)
        t0 = time.perf_counter()
        try:
            resp = await client.post(path, json=body)
            ok = resp.status_code == 200
        except httpx.HTTPError:
            ok = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if now < record_after:  # warm-up request: not recorded
            continue
        if not ok:
            errors.append(1)
            continue
        latencies.append(elapsed_ms)
        server_ms.append(float(resp.json()["inference_time_ms"]))


async def run_load(
    base_url: str,
    mode: str,
    concurrency: int,
    duration_s: float,
    batch_size: int = 50,
    warmup_s: float = 3.0,
) -> LoadResult:
    """Run one closed-loop load test and return its summary."""
    appointment_pool()  # build payloads before the clock starts
    latencies: list[float] = []
    server_ms: list[float] = []
    errors: list[int] = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0, limits=limits) as client:
        start = time.perf_counter()
        record_after = start + warmup_s
        stop_at = record_after + duration_s
        await asyncio.gather(
            *(
                _worker(client, mode, batch_size, stop_at, record_after, latencies, server_ms, errors, i)
                for i in range(concurrency)
            )
        )
    n = len(latencies)
    rows_per_request = 1 if mode == "single" else batch_size
    pct = np.percentile(latencies, [50, 95, 99]) if n else [float("nan")] * 3
    return LoadResult(
        mode=mode,
        concurrency=concurrency,
        batch_size=rows_per_request,
        duration_s=duration_s,
        requests=n,
        errors=len(errors),
        requests_per_s=round(n / duration_s, 1),
        predictions_per_s=round(n * rows_per_request / duration_s, 1),
        p50_ms=round(float(pct[0]), 2),
        p95_ms=round(float(pct[1]), 2),
        p99_ms=round(float(pct[2]), 2),
        mean_server_inference_ms=round(statistics.fmean(server_ms), 3) if server_ms else float("nan"),
    )


def main() -> None:
    """CLI entry point; prints the result as JSON."""
    parser = argparse.ArgumentParser(description="Closed-loop load test for the no-show API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    result: dict[str, Any] = asdict(
        asyncio.run(run_load(args.url, args.mode, args.concurrency, args.duration, args.batch_size))
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
