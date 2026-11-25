"""Benchmark matrix for the Ray Serve deployment (plus a no-Ray baseline).

Deployment configurations (``--configs``), written ``<replicas>x<max_batch_size>``:

* ``1x1``  - 1 replica, dynamic batching effectively OFF (batches of 1)
* ``1x32`` - 1 replica, dynamic batching ON
* ``2x32`` - 2 replicas
* ``4x32`` - 4 replicas (oversubscribed on a 4-core laptop that also runs
  the load generator; kept to show what happens)

Scenarios run against each configuration:

* ``probe``    - ``/predict`` with 1 client: fixed per-request cost, no queueing
* ``single``   - ``/predict`` with 32 clients (exercises ``@serve.batch``)
* ``batch50``  - ``/predict/batch`` with 50 rows, 8 clients
* ``batch500`` - ``/predict/batch`` with 500 rows, 4 clients (replica-heavy work)
* Locust ``SingleUser`` / ``BatchUser`` (optional, ``--locust``)

Measurement hygiene, learned from a first run whose numbers were dominated by
the harness rather than the service:

* the load generator runs as a **separate process** (``bench.loadtest`` /
  ``locust`` subprocesses), never inside the Ray driver;
* per-request access logs are **off** (they cost CPU and are shipped to the
  driver);
* system CPU utilisation is sampled right before each configuration, because
  the benchmark machine is a shared laptop.

``--uvicorn-baseline`` additionally runs the same scenarios against the plain
FastAPI app under a single uvicorn process (no Ray), to show the cost and
benefit of the Serve layer on this machine.

Usage::

    python -m bench.run_benchmarks --configs 1x1 1x32 2x32 4x32 --locust --uvicorn-baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
import ray
from ray import serve

from noshow.config import ServeSettings
from noshow.serve_app import build_app

RESULTS_DIR = Path("results")

# name -> (endpoint mode, clients, rows per request)
SCENARIOS: dict[str, tuple[str, int, int]] = {
    "probe": ("single", 1, 1),
    "single": ("single", 32, 1),
    "batch50": ("batch", 8, 50),
    "batch500": ("batch", 4, 500),
}


def wait_healthy(url: str, timeout_s: float = 120.0) -> None:
    """Poll ``/health`` until it returns 200 or the timeout expires."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"service at {url} did not become healthy")


def run_loadtest(url: str, mode: str, clients: int, rows: int, duration_s: float) -> dict[str, Any]:
    """Run ``bench.loadtest`` in a separate process and return its JSON result."""
    cmd = [
        sys.executable, "-m", "bench.loadtest", "--url", url, "--mode", mode,
        "--concurrency", str(clients), "--duration", str(duration_s), "--batch-size", str(rows),
    ]  # fmt: skip
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def run_locust(url: str, user_class: str, users: int, duration_s: int, tag: str) -> dict[str, Any]:
    """Run Locust headless as a subprocess and parse its aggregated stats CSV."""
    prefix = RESULTS_DIR / f"locust_{tag}"
    cmd = [
        sys.executable, "-m", "locust", "-f", "bench/locustfile.py", "--headless",
        "-u", str(users), "-r", str(users), "-t", f"{duration_s}s",
        "--host", url, "--csv", str(prefix), "--only-summary", user_class,
    ]  # fmt: skip
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    with open(f"{prefix}_stats.csv", newline="") as fh:
        agg = {row["Name"]: row for row in csv.DictReader(fh)}["Aggregated"]
    return {
        "requests": int(agg["Request Count"]),
        "errors": int(agg["Failure Count"]),
        "requests_per_s": round(float(agg["Requests/s"]), 1),
        "p50_ms": float(agg["50%"]),
        "p95_ms": float(agg["95%"]),
        "p99_ms": float(agg["99%"]),
        "mean_server_inference_ms": None,  # locust does not parse response bodies
    }


def run_scenarios(url: str, config: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run every scenario against an already-running server."""
    rows: list[dict[str, Any]] = []
    for name, (mode, clients, size) in SCENARIOS.items():
        res = run_loadtest(url, mode, clients, size, args.duration)
        rows.append({"config": config, "tool": "loadtest", "scenario": name, **res})
        print(json.dumps(rows[-1]), flush=True)
    if args.locust:
        for name, user_class, clients, size in (
            ("single", "SingleUser", 32, 1),
            ("batch50", "BatchUser", 8, 50),
        ):
            res = run_locust(url, user_class, clients, int(args.duration), f"{config}_{name}")
            res.update(mode="single" if size == 1 else "batch", concurrency=clients, batch_size=size)
            res["predictions_per_s"] = round(res["requests_per_s"] * size, 1)
            rows.append({"config": config, "tool": "locust", "scenario": name, **res})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def cpu_busy_percent(seconds: float = 3.0) -> float:
    """System-wide CPU utilisation over a short window (other apps included)."""
    return float(psutil.cpu_percent(interval=seconds))


def uvicorn_baseline(args: argparse.Namespace, port: int) -> list[dict[str, Any]]:
    """Run the scenarios against ``uvicorn noshow.api:app`` (one process, no Ray)."""
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "MODEL_THREADS": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "noshow.api:app", "--port", str(port), "--log-level", "warning",
         "--no-access-log"],
        env=env,
    )  # fmt: skip
    try:
        wait_healthy(url)
        print(f"--- uvicorn baseline (cpu busy before: {cpu_busy_percent():.0f}%)", flush=True)
        return run_scenarios(url, "uvicorn-1proc", args)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def environment_info() -> dict[str, Any]:
    """Software facts recorded with every result (hardware is documented by hand)."""
    return {
        "python": platform.python_version(),
        "ray": ray.__version__,
        "os": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }


def print_table(results: list[dict[str, Any]]) -> None:
    """Print all results as one Markdown table."""
    print(
        "\n| config | tool | scenario | clients | rows/req | req/s | predictions/s "
        "| p50 ms | p95 ms | p99 ms | server ms (mean) | errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['config']} | {r['tool']} | {r['scenario']} | {r['concurrency']} | {r['batch_size']} "
            f"| {r['requests_per_s']} | {r['predictions_per_s']} | {r['p50_ms']} | {r['p95_ms']} "
            f"| {r['p99_ms']} | {r['mean_server_inference_ms']} | {r['errors']} |"
        )


def main() -> None:
    """Run the matrix and save/print the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=["1x1", "1x32", "2x32", "4x32"])
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--locust", action="store_true", help="also run Locust headless")
    parser.add_argument("--uvicorn-baseline", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    base = ServeSettings.from_env()
    url = f"http://{base.host}:{base.port}"
    results: list[dict[str, Any]] = []
    cpu_before: dict[str, float] = {}

    # log_to_driver=False: replica/proxy logs are not streamed into this process.
    ray.init(include_dashboard=False, logging_level="ERROR", log_to_driver=False)
    serve.start(http_options={"host": base.host, "port": base.port}, logging_config=base.logging_config())
    for config in args.configs:
        replicas, batch = (int(x) for x in config.split("x"))
        settings = replace(base, num_replicas=replicas, max_batch_size=batch)
        serve.run(build_app(settings), name="no-show", route_prefix="/")
        wait_healthy(url)
        time.sleep(2.0)  # let replica start-up CPU settle before sampling
        cpu_before[config] = cpu_busy_percent()
        print(f"--- {config} deployed (cpu busy before: {cpu_before[config]:.0f}%)", flush=True)
        results.extend(run_scenarios(url, config, args))
        serve.delete("no-show")
    serve.shutdown()
    ray.shutdown()

    if args.uvicorn_baseline:
        cpu_before["uvicorn-1proc"] = cpu_busy_percent()
        results.extend(uvicorn_baseline(args, base.port + 1))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"benchmark_{stamp}.json"
    payload = {
        "environment": environment_info(),
        "settings": {**asdict(base), "model_dir": str(base.model_dir)},
        "args": vars(args),
        "cpu_busy_percent_before_config": cpu_before,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print_table(results)
    print(f"\ncpu busy before each config (%): {cpu_before}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
