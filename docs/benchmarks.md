# Benchmarks (raw results)

Every number in the README comes from the runs below. They were run on the
machine described here, with the commands shown, and are reproduced without
rounding or editing. JSON files with the same data are in
[`results/`](../results).

## Machine

| | |
|---|---|
| CPU | 11th Gen Intel Core i7-1185G7 @ 3.00GHz, 4 cores / 8 logical processors |
| RAM | 39.7 GB |
| OS | Windows 11 Pro (10.0.26200), native (no Docker, no WSL) |
| Python | 3.13.1 (CI and the Docker image use 3.11/3.12) |
| Ray | 2.60.1 |
| XGBoost / FastAPI / Locust | 3.4.1 / 0.165.0 / 2.47.0 |

**Conditions.** This was a shared laptop with other applications running
(browser, IDE, other Python jobs). The runner sampled system-wide CPU
utilisation for 3 s before each configuration, *before* any load was
applied. The values are printed with each run and range from 23% to 62%.
The load generator, the Ray Serve proxy, the controller and the replicas all
ran on the same 4-core CPU. Treat absolute numbers as pessimistic and noisy,
and compare rows within a run.

## 1. Inference micro-benchmark (no HTTP, no Ray)

```bash
python -m bench.microbench --repeat 300
```

Median wall time per call, 1 XGBoost thread:

| rows | pandas features+model (ms) | fast features+model (ms) | of which model only (ms) |
|---|---|---|---|
| 1 | 12.713 | 1.419 | 1.647 |
| 32 | 13.342 | 1.911 | 1.786 |
| 50 | 12.798 | 2.343 | 2.082 |

The pandas feature path cost about 11 ms regardless of the row count. The
NumPy fast path removed almost all of it. The model call itself is about
1.4–2.1 ms for 1–50 rows. Some single-row figures put "model only" above
"fast features+model", which is noise of about 0.2 ms on a busy machine.

## 2. Ray Serve matrix: final runs (two passes)

```bash
SERVE_PORT=8710 python -m bench.run_benchmarks --configs 1x1 1x32 2x32 4x32 --duration 15 --locust --uvicorn-baseline      # run twice
```

- `config` = `<replicas>x<max_batch_size>`. `1x1` means dynamic batching is
  effectively off. `uvicorn-1proc` is the plain FastAPI app in one uvicorn
  process, without Ray.
- Scenarios: `probe` = `/predict` with 1 client; `single` = `/predict` with
  32 clients; `batch50` / `batch500` = `/predict/batch` with 50 / 500 rows
  per request.
- Each loadtest scenario ran for 15 s after a 3 s warm-up. Locust ran for
  15 s with all users spawned at once.
- Latencies are client-side. `server ms` is the mean `inference_time_ms`
  reported by the replica.

### Pass 1

| config | tool | scenario | clients | rows/req | req/s | predictions/s | p50 ms | p95 ms | p99 ms | server ms (mean) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1 | loadtest | probe | 1 | 1 | 59.1 | 59.1 | 16.51 | 21.18 | 29.54 | 1.675 | 0 |
| 1x1 | loadtest | single | 32 | 1 | 72.1 | 72.1 | 267.13 | 1240.38 | 2112.37 | 2.465 | 0 |
| 1x1 | loadtest | batch50 | 8 | 50 | 113.3 | 5663.3 | 65.41 | 100.03 | 175.48 | 3.852 | 0 |
| 1x1 | loadtest | batch500 | 4 | 500 | 39.3 | 19666.7 | 90.89 | 138.37 | 367.37 | 20.326 | 0 |
| 1x1 | locust | single | 32 | 1 | 84.8 | 84.8 | 360.0 | 480.0 | 510.0 | None | 0 |
| 1x1 | locust | batch50 | 8 | 50 | 103.5 | 5175.0 | 74.0 | 110.0 | 130.0 | None | 0 |
| 1x32 | loadtest | probe | 1 | 1 | 31.5 | 31.5 | 31.57 | 33.91 | 39.44 | 15.847 | 0 |
| 1x32 | loadtest | single | 32 | 1 | 69.9 | 69.9 | 276.6 | 1448.17 | 2115.75 | 10.534 | 0 |
| 1x32 | loadtest | batch50 | 8 | 50 | 128.7 | 6436.7 | 60.58 | 77.01 | 109.34 | 3.489 | 0 |
| 1x32 | loadtest | batch500 | 4 | 500 | 44.3 | 22133.3 | 83.08 | 115.96 | 141.59 | 17.415 | 0 |
| 1x32 | locust | single | 32 | 1 | 97.8 | 97.8 | 330.0 | 350.0 | 360.0 | None | 0 |
| 1x32 | locust | batch50 | 8 | 50 | 128.8 | 6440.0 | 61.0 | 73.0 | 82.0 | None | 0 |
| 2x32 | loadtest | probe | 1 | 1 | 31.7 | 31.7 | 31.48 | 32.91 | 36.65 | 18.425 | 0 |
| 2x32 | loadtest | single | 32 | 1 | 79.9 | 79.9 | 306.82 | 1186.54 | 2004.77 | 12.449 | 0 |
| 2x32 | loadtest | batch50 | 8 | 50 | 117.7 | 5883.3 | 55.4 | 138.02 | 271.59 | 2.836 | 0 |
| 2x32 | loadtest | batch500 | 4 | 500 | 58.2 | 29100.0 | 59.15 | 85.97 | 357.98 | 16.69 | 0 |
| 2x32 | locust | single | 32 | 1 | 108.4 | 108.4 | 290.0 | 350.0 | 410.0 | None | 0 |
| 2x32 | locust | batch50 | 8 | 50 | 157.6 | 7880.0 | 50.0 | 61.0 | 66.0 | None | 0 |
| 4x32 | loadtest | probe | 1 | 1 | 31.7 | 31.7 | 31.33 | 33.04 | 36.9 | 16.85 | 0 |
| 4x32 | loadtest | single | 32 | 1 | 69.5 | 69.5 | 257.63 | 1404.77 | 2094.5 | 14.849 | 0 |
| 4x32 | loadtest | batch50 | 8 | 50 | 99.7 | 4983.3 | 63.27 | 183.47 | 316.19 | 3.064 | 0 |
| 4x32 | loadtest | batch500 | 4 | 500 | 55.1 | 27533.3 | 66.08 | 97.86 | 117.75 | 19.277 | 0 |
| 4x32 | locust | single | 32 | 1 | 93.0 | 93.0 | 340.0 | 410.0 | 430.0 | None | 0 |
| 4x32 | locust | batch50 | 8 | 50 | 124.7 | 6235.0 | 60.0 | 92.0 | 110.0 | None | 0 |
| uvicorn-1proc | loadtest | probe | 1 | 1 | 190.7 | 190.7 | 4.82 | 7.81 | 12.97 | 0.98 | 0 |
| uvicorn-1proc | loadtest | single | 32 | 1 | 144.3 | 144.3 | 137.43 | 672.99 | 1032.54 | 1.016 | 0 |
| uvicorn-1proc | loadtest | batch50 | 8 | 50 | 171.6 | 8580.0 | 30.55 | 87.2 | 211.91 | 2.323 | 0 |
| uvicorn-1proc | loadtest | batch500 | 4 | 500 | 86.7 | 43333.3 | 40.24 | 80.52 | 107.48 | 13.262 | 0 |
| uvicorn-1proc | locust | single | 32 | 1 | 733.8 | 733.8 | 41.0 | 55.0 | 66.0 | None | 0 |
| uvicorn-1proc | locust | batch50 | 8 | 50 | 452.2 | 22610.0 | 17.0 | 24.0 | 29.0 | None | 0 |

cpu busy before each config (%): {'1x1': 62.4, '1x32': 52.9, '2x32': 59.7, '4x32': 38.8, 'uvicorn-1proc': 62.6}
saved results\benchmark_20261125-181432.json

### Pass 2

| config | tool | scenario | clients | rows/req | req/s | predictions/s | p50 ms | p95 ms | p99 ms | server ms (mean) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x1 | loadtest | probe | 1 | 1 | 61.4 | 61.4 | 16.27 | 18.94 | 21.3 | 1.646 | 0 |
| 1x1 | loadtest | single | 32 | 1 | 103.3 | 103.3 | 196.48 | 914.45 | 1427.94 | 2.12 | 0 |
| 1x1 | loadtest | batch50 | 8 | 50 | 141.7 | 7083.3 | 54.23 | 73.7 | 97.51 | 3.251 | 0 |
| 1x1 | loadtest | batch500 | 4 | 500 | 55.5 | 27766.7 | 65.54 | 84.4 | 348.4 | 13.188 | 0 |
| 1x1 | locust | single | 32 | 1 | 129.4 | 129.4 | 240.0 | 300.0 | 340.0 | None | 0 |
| 1x1 | locust | batch50 | 8 | 50 | 149.9 | 7495.0 | 52.0 | 65.0 | 71.0 | None | 0 |
| 1x32 | loadtest | probe | 1 | 1 | 40.2 | 40.2 | 30.62 | 32.2 | 32.99 | 14.853 | 0 |
| 1x32 | loadtest | single | 32 | 1 | 102.1 | 102.1 | 179.45 | 934.47 | 1591.48 | 8.937 | 0 |
| 1x32 | loadtest | batch50 | 8 | 50 | 142.5 | 7123.3 | 54.63 | 69.52 | 88.78 | 3.261 | 0 |
| 1x32 | loadtest | batch500 | 4 | 500 | 54.3 | 27166.7 | 64.08 | 98.2 | 359.83 | 13.427 | 0 |
| 1x32 | locust | single | 32 | 1 | 100.5 | 100.5 | 310.0 | 390.0 | 420.0 | None | 0 |
| 1x32 | locust | batch50 | 8 | 50 | 135.5 | 6775.0 | 55.0 | 81.0 | 98.0 | None | 0 |
| 2x32 | loadtest | probe | 1 | 1 | 41.6 | 41.6 | 30.05 | 32.29 | 33.47 | 14.143 | 0 |
| 2x32 | loadtest | single | 32 | 1 | 93.6 | 93.6 | 266.88 | 919.31 | 1640.7 | 10.044 | 0 |
| 2x32 | loadtest | batch50 | 8 | 50 | 139.9 | 6993.3 | 49.09 | 107.83 | 189.46 | 2.7 | 0 |
| 2x32 | loadtest | batch500 | 4 | 500 | 65.9 | 32966.7 | 51.48 | 70.15 | 347.07 | 13.906 | 0 |
| 2x32 | locust | single | 32 | 1 | 127.5 | 127.5 | 240.0 | 290.0 | 400.0 | None | 0 |
| 2x32 | locust | batch50 | 8 | 50 | 150.2 | 7510.0 | 51.0 | 68.0 | 81.0 | None | 0 |
| 4x32 | loadtest | probe | 1 | 1 | 31.7 | 31.7 | 31.48 | 32.8 | 33.96 | 17.649 | 0 |
| 4x32 | loadtest | single | 32 | 1 | 90.1 | 90.1 | 192.7 | 1062.43 | 1632.03 | 14.31 | 0 |
| 4x32 | loadtest | batch50 | 8 | 50 | 113.1 | 5653.3 | 60.05 | 140.3 | 235.88 | 3.019 | 0 |
| 4x32 | loadtest | batch500 | 4 | 500 | 64.0 | 32000.0 | 56.89 | 77.7 | 90.19 | 16.76 | 0 |
| 4x32 | locust | single | 32 | 1 | 101.6 | 101.6 | 280.0 | 490.0 | 610.0 | None | 0 |
| 4x32 | locust | batch50 | 8 | 50 | 145.4 | 7270.0 | 53.0 | 74.0 | 90.0 | None | 0 |
| uvicorn-1proc | loadtest | probe | 1 | 1 | 267.3 | 267.3 | 3.43 | 5.47 | 6.58 | 0.698 | 0 |
| uvicorn-1proc | loadtest | single | 32 | 1 | 141.0 | 141.0 | 139.22 | 662.81 | 1035.09 | 1.002 | 0 |
| uvicorn-1proc | loadtest | batch50 | 8 | 50 | 162.1 | 8103.3 | 28.71 | 72.9 | 135.4 | 2.251 | 0 |
| uvicorn-1proc | loadtest | batch500 | 4 | 500 | 91.3 | 45633.3 | 38.86 | 66.18 | 101.33 | 12.108 | 0 |
| uvicorn-1proc | locust | single | 32 | 1 | 810.0 | 810.0 | 38.0 | 49.0 | 62.0 | None | 0 |
| uvicorn-1proc | locust | batch50 | 8 | 50 | 443.6 | 22180.0 | 17.0 | 23.0 | 29.0 | None | 0 |

cpu busy before each config (%): {'1x1': 62.2, '1x32': 23.4, '2x32': 25.3, '4x32': 39.7, 'uvicorn-1proc': 41.5}
saved results\benchmark_20261125-182526.json

## 3. First run (discarded as the headline, kept for transparency)

The first complete run used the same service code, including the NumPy
fast path, but had two problems in the harness:

- per-request access logs were on and streamed to the driver process;
- the async load client ran inside the Ray driver process.

It measured mostly the harness. For example, `/predict` at 1 replica with 32
clients gave 38.0 req/s with p50 529.76 ms. It also seemed to show 4
replicas as slower than 1. Raw output:

```
--- 1 replica(s) deployed
{"tool": "loadtest", "replicas": 1, "mode": "single", "concurrency": 32, "batch_size": 1, "duration_s": 20.0, "requests": 760, "errors": 0, "requests_per_s": 38.0, "predictions_per_s": 38.0, "p50_ms": 529.76, "p95_ms": 2422.37, "p99_ms": 3745.25, "mean_server_inference_ms": 12.466}
{"tool": "locust", "scenario": "SingleUser", "users": 32, "duration_s": 20, "requests": 1065, "failures": 0, "requests_per_s": 75.1, "p50_ms": 420.0, "p95_ms": 560.0, "p99_ms": 1200.0, "replicas": 1}
{"tool": "loadtest", "replicas": 1, "mode": "batch", "concurrency": 8, "batch_size": 50, "duration_s": 20.0, "requests": 1030, "errors": 0, "requests_per_s": 51.5, "predictions_per_s": 2575.0, "p50_ms": 144.36, "p95_ms": 265.65, "p99_ms": 379.38, "mean_server_inference_ms": 8.701}
{"tool": "locust", "scenario": "BatchUser", "users": 8, "duration_s": 20, "requests": 266, "failures": 0, "requests_per_s": 29.3, "p50_ms": 220.0, "p95_ms": 540.0, "p99_ms": 1100.0, "replicas": 1}
--- 2 replica(s) deployed
{"tool": "loadtest", "replicas": 2, "mode": "single", "concurrency": 32, "batch_size": 1, "duration_s": 20.0, "requests": 940, "errors": 0, "requests_per_s": 47.0, "predictions_per_s": 47.0, "p50_ms": 398.74, "p95_ms": 2086.2, "p99_ms": 3411.99, "mean_server_inference_ms": 14.352}
{"tool": "locust", "scenario": "SingleUser", "users": 32, "duration_s": 20, "requests": 1238, "failures": 0, "requests_per_s": 81.6, "p50_ms": 360.0, "p95_ms": 640.0, "p99_ms": 750.0, "replicas": 2}
{"tool": "loadtest", "replicas": 2, "mode": "batch", "concurrency": 8, "batch_size": 50, "duration_s": 20.0, "requests": 1134, "errors": 0, "requests_per_s": 56.7, "predictions_per_s": 2835.0, "p50_ms": 99.82, "p95_ms": 360.03, "p99_ms": 704.58, "mean_server_inference_ms": 5.758}
{"tool": "locust", "scenario": "BatchUser", "users": 8, "duration_s": 20, "requests": 458, "failures": 0, "requests_per_s": 46.8, "p50_ms": 150.0, "p95_ms": 260.0, "p99_ms": 870.0, "replicas": 2}
--- 4 replica(s) deployed
{"tool": "loadtest", "replicas": 4, "mode": "single", "concurrency": 32, "batch_size": 1, "duration_s": 20.0, "requests": 441, "errors": 0, "requests_per_s": 22.1, "predictions_per_s": 22.1, "p50_ms": 851.12, "p95_ms": 4043.2, "p99_ms": 6824.06, "mean_server_inference_ms": 21.154}
{"tool": "locust", "scenario": "SingleUser", "users": 32, "duration_s": 20, "requests": 552, "failures": 0, "requests_per_s": 49.9, "p50_ms": 570.0, "p95_ms": 1000.0, "p99_ms": 1600.0, "replicas": 4}
{"tool": "loadtest", "replicas": 4, "mode": "batch", "concurrency": 8, "batch_size": 50, "duration_s": 20.0, "requests": 516, "errors": 0, "requests_per_s": 25.8, "predictions_per_s": 1290.0, "p50_ms": 211.16, "p95_ms": 761.84, "p99_ms": 1402.13, "mean_server_inference_ms": 12.065}
{"tool": "locust", "scenario": "BatchUser", "users": 8, "duration_s": 20, "requests": 841, "failures": 0, "requests_per_s": 63.7, "p50_ms": 120.0, "p95_ms": 190.0, "p99_ms": 250.0, "replicas": 4}
[33m(raylet)[0m Task ServeController.graceful_shutdown failed. There are infinite retries remaining, so the task will be retried. Error: The actor is dead because it was killed by `ray.kill`.

| tool | replicas | endpoint | clients | req/s | predictions/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| loadtest | 1 | /predict | 32 | 38.0 | 38.0 | 529.76 | 2422.37 | 3745.25 |
| locust | 1 | /predict | 32 | 75.1 | 75.1 | 420.0 | 560.0 | 1200.0 |
| loadtest | 1 | /predict/batch | 8 | 51.5 | 2575.0 | 144.36 | 265.65 | 379.38 |
| locust | 1 | /predict/batch | 8 | 29.3 | 1465.0 | 220.0 | 540.0 | 1100.0 |
| loadtest | 2 | /predict | 32 | 47.0 | 47.0 | 398.74 | 2086.2 | 3411.99 |
| locust | 2 | /predict | 32 | 81.6 | 81.6 | 360.0 | 640.0 | 750.0 |
| loadtest | 2 | /predict/batch | 8 | 56.7 | 2835.0 | 99.82 | 360.03 | 704.58 |
| locust | 2 | /predict/batch | 8 | 46.8 | 2340.0 | 150.0 | 260.0 | 870.0 |
| loadtest | 4 | /predict | 32 | 22.1 | 22.1 | 851.12 | 4043.2 | 6824.06 |
| locust | 4 | /predict | 32 | 49.9 | 49.9 | 570.0 | 1000.0 | 1600.0 |
| loadtest | 4 | /predict/batch | 8 | 25.8 | 1290.0 | 211.16 | 761.84 | 1402.13 |
| locust | 4 | /predict/batch | 8 | 63.7 | 3185.0 | 120.0 | 190.0 | 250.0 |

saved results\benchmark_20261125-160147.json
```

An earlier attempt before that one was aborted, and no numbers from it are
used. It ran on port 8000, which another local process was also hitting. The
generator had a bug that produced invalid payloads (negative
`days_since_last_visit`). Features were still built with pandas, and the replicas reported a mean of
155.426 ms server-side time per `/predict` request (32 clients, batch
queueing included). All three were fixed before
the run above.

## What the numbers show

All figures below are taken from the Pass 1 / Pass 2 tables above.

1. **Each request through the Serve proxy costs about 15 ms on this
   machine.** With one client (`probe`, `1x1`), client p50 was 16.51 / 16.27
   ms, while the replica reported 1.675 / 1.646 ms. The plain uvicorn app
   answered the same probe in 4.82 / 3.43 ms.
2. **Single-request throughput is capped by the proxy path, not by the
   model.** For `/predict` with 32 clients, every Ray Serve configuration
   stayed between 69.5 and 129.4 req/s (both tools, both passes), and adding
   replicas did not raise that ceiling. Locust against the single uvicorn
   process reached 733.8 / 810.0 req/s. The replicas were not the
   bottleneck: they reported 2–15 ms of server time per request.
3. **Replicas help when the replica does most of the work.** For
   `/predict/batch` with 500 rows (4 clients), going from 1 to 2 replicas
   (`1x32` → `2x32`) raised throughput from 44.3 to 58.2 req/s (+31%) in
   pass 1 and from 54.3 to 65.9 req/s (+21%) in pass 2. That is 22,133.3 →
   29,100.0 and 27,166.7 → 32,966.7 predictions/s. With 4 replicas (`4x32`) the
   numbers were 55.1 and 64.0 req/s, no better than 2. The 4 physical cores
   were shared by 4 replicas, the proxy, the Ray control processes, the load
   generator and other applications.
4. **Dynamic batching gave no consistent throughput gain here and added
   latency to lone requests.** At 1 replica, `/predict` with 32 clients went
   from 72.1 → 69.9 req/s (loadtest) and 84.8 → 97.8 (Locust) in pass 1, and
   from 103.3 → 102.1 and 129.4 → 100.5 in pass 2 (`1x1` → `1x32`). This is
   expected given finding 2: batching saves model calls, but the model call
   (about 2 ms) was not the bottleneck. With one client, batching raised p50
   from 16.51 to 31.57 ms (pass 1) and from 16.27 to 30.62 ms (pass 2). The
   replica-side time for a lone request was 14.1–18.4 ms (all `x32` probes), although
   `batch_wait_timeout_s` is 5 ms. That fits Windows' default timer
   granularity of about 15.6 ms, but this explanation was not verified, and
   behaviour on Linux may differ.
5. **Batching on the client side is the most effective lever.** Sending 500
   appointments per request gave up to 32,966.7 predictions/s through Ray
   Serve, against at most 129.4 predictions/s with one appointment per
   request.
6. **The async load client tops out at about 140 req/s.** Against uvicorn,
   `bench.loadtest` measured 141.0–144.3 req/s for `single`, while Locust
   measured 733.8–810.0. Above that rate, Locust's numbers are the
   trustworthy ones. For the Ray Serve rows, both tools reported similar
   ranges, so the server was the limit there.
7. **Run-to-run noise is large.** The same scenario differed by as much as 43%
   between passes (`1x1` single loadtest: 72.1 vs 103.3 req/s), in line with the varying background CPU load. Draw conclusions
   only from effects that appear in both passes.
