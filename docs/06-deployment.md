# 6. Deployment: Docker, compose and CI

**Files:** [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml),
[`serve_config.yaml`](../serve_config.yaml), [`.env.example`](../.env.example),
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

## What problem it solves

Anyone should be able to start the service with one command and get the same
model, without installing Python packages locally or downloading a model
binary.

## Docker image

The image (`python:3.12-slim`) installs the package and then runs the two
pipeline steps **at build time**:

```dockerfile
RUN python -m noshow.data --out data/appointments.csv \
 && python -m noshow.train --data data/appointments.csv --version 1.0.0
```

Because the generator uses a fixed seed, every build produces the same data
and an equivalent model. No model file is committed to git (`artifacts/` is
ignored). In a real system this step would be replaced by pulling a
registered model version from a model registry. See "Future work" in the
README.

The container runs `python -m noshow.serve_app`, which reads the env vars
documented in `.env.example`.

## docker compose

```bash
docker compose up --build
curl localhost:8000/health
```

Settings explained in `docker-compose.yml`:

- `NUM_REPLICAS`, `MAX_BATCH_SIZE`, `BATCH_WAIT_TIMEOUT_S`,
  `MAX_ONGOING_REQUESTS`, `MODEL_THREADS`: the scaling and batching knobs
  described in [04-serving.md](04-serving.md).
- `shm_size: 1gb`: Ray uses `/dev/shm` for its object store. Docker's 64 MB
  default makes Ray fall back to slower storage.
- `healthcheck`: polls `/health`, with a 30 s start period for Ray start-up
  and model loading.

> **Not verified on the development machine.** Docker Desktop is installed on the
> Windows laptop used to build and benchmark this project, but its engine did
> only returned errors (HTTP 502 and 500 from the Docker API), so the Dockerfile
> and compose file were written but **not run** there. All benchmark numbers
> come from a native Windows run (see [benchmarks.md](benchmarks.md)).

## Ray Serve config file

`serve_config.yaml` is the declarative alternative to env vars
(`serve run serve_config.yaml`, or `serve deploy` against a running Ray
cluster). Its `args:` block is passed to `noshow.serve_app:app_builder`.
`serve run serve_config.yaml` was run from the repository root on the
development machine. It started 2 replicas and answered `/health` and
`/predict` correctly.

## CI

GitHub Actions runs on every push and pull request, on Python 3.11 and 3.12:

1. `pip install -e ".[dev]"`
2. `ruff check .` and `ruff format --check .`
3. `pytest -q`: unit and API tests. The `integration` tests, which start
   Ray Serve, are excluded by default through `addopts` in `pyproject.toml`.
   Run them locally with `pytest -m integration`.
