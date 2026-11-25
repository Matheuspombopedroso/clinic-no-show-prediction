# Image for the Ray Serve no-show service.
# The model is generated and trained at build time from the fixed-seed
# synthetic data, so the image is self-contained and no model binary is
# committed to git.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package and its dependencies (pyproject.toml + src/). Kept
# simple for a demo; a production image would install pinned dependencies in
# a separate layer first so code-only changes rebuild faster.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Build-time data generation + training (deterministic: fixed seed).
RUN python -m noshow.data --out data/appointments.csv \
 && python -m noshow.train --data data/appointments.csv --version 1.0.0

COPY serve_config.yaml ./

# Serve settings (overridable from docker-compose / `docker run -e`).
ENV MODEL_DIR=artifacts/no-show-xgb/1.0.0 \
    SERVE_HOST=0.0.0.0 \
    SERVE_PORT=8000 \
    NUM_REPLICAS=2

EXPOSE 8000
CMD ["python", "-m", "noshow.serve_app"]
