# 5. Load testing and benchmarks

**Code:** [`bench/loadtest.py`](../bench/loadtest.py),
[`bench/locustfile.py`](../bench/locustfile.py),
[`bench/run_benchmarks.py`](../bench/run_benchmarks.py),
[`bench/microbench.py`](../bench/microbench.py),
[`bench/payloads.py`](../bench/payloads.py) ·
**Raw results:** [benchmarks.md](benchmarks.md)

## What problem it solves

"Low latency" and "scales horizontally" are claims, and claims need
measurements. This part measures p50/p95/p99 latency and throughput for:

- `/predict` (one appointment per request, through `@serve.batch`) vs
  `/predict/batch` (50 or 500 appointments per request);
- 1 vs 2 vs 4 Ray Serve replicas;
- dynamic batching off (`max_batch_size=1`) vs on (`32`);
- Ray Serve vs a single uvicorn process running the same FastAPI routes;

using two independent load generators, plus an in-process micro-benchmark of
the inference optimisation.

## How it works

### Payloads

`bench/payloads.py` draws 2,000 realistic appointments from the synthetic
generator (seed 123). Each request picks a random one, or a random sample of
50 for `/predict/batch`, so every request runs validation, feature
computation and the model on different values.

### `bench/loadtest.py` (async closed loop)

`C` virtual clients (asyncio + httpx) each send a request, wait for the
response and immediately send the next, for `duration` seconds after a 3 s
warm-up that is not recorded. It reports p50/p95/p99 of client-side latency,
requests/s, predictions/s (requests × rows per request), errors, and the mean
server-reported `inference_time_ms`.

### Locust (headless)

`bench/locustfile.py` defines `SingleUser` and `BatchUser`. Both use
`FastHttpUser` with no wait time, so the setup is the same closed loop as the
async script and the two tools can be compared.

### `bench/run_benchmarks.py` (the matrix)

Configurations are written `<replicas>x<max_batch_size>`: `1x1`, `1x32`,
`2x32` and `4x32`. For each one, the runner:

1. deploys the Serve app and waits for `/health`;
2. samples system CPU utilisation;
3. runs four `bench.loadtest` scenarios, each **in a separate process**:
   - `probe`: `/predict` with 1 client;
   - `single`: `/predict` with 32 clients;
   - `batch50`: 50 rows per request, 8 clients;
   - `batch500`: 500 rows per request, 4 clients;
4. runs two Locust scenarios (`single`, `batch50`);
5. deletes the app.

With `--uvicorn-baseline`, it then repeats the same scenarios against
`uvicorn noshow.api:app` in a single process. Results are printed as a
Markdown table and saved to `results/benchmark_<timestamp>.json`.

```bash
SERVE_PORT=8710 python -m bench.run_benchmarks --configs 1x1 1x32 2x32 4x32 --duration 15 --locust --uvicorn-baseline
python -m bench.microbench --repeat 300
```

### Lessons from the first run

The first complete run measured the harness more than the service. Two
things were wrong:

- the async client ran inside the Ray driver process;
- access logs for every request were streamed to that same process.

An aborted attempt before it also still built features with pandas, which
is why the NumPy fast path exists.

The runner now launches clients as subprocesses, starts Ray with
`log_to_driver=False`, and turns access logs off (`ACCESS_LOG=false`). The
first run's output is kept, clearly labelled, in
[benchmarks.md](benchmarks.md).

## How to read the numbers

- **Client-side latency** covers the whole HTTP round trip through the Serve
  proxy, the hop to a replica, the batch queue, the model, and the way back.
  `inference_time_ms` covers only the part inside the replica (see
  [04-serving.md](04-serving.md)). The gap between the two is the cost of
  the proxy and the transport.
- **Closed loop**: throughput is what the server sustains with `C`
  concurrent clients. In a closed loop, latency grows with `C` once the
  server is saturated (Little's law: latency ≈ C / throughput).
- **Same machine**: the load generator, the Ray proxy and the replicas share
  one laptop CPU, and other applications were running during the benchmark.
  Absolute numbers are therefore pessimistic and noisy. Compare rows with
  each other, not with numbers from server hardware.

The measured results and what they show are in [benchmarks.md](benchmarks.md)
and the README table.
