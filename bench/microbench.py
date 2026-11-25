"""In-process micro-benchmark: pandas feature path vs NumPy fast path.

Measures, without HTTP or Ray, how long one call to the model pipeline takes
for 1, 32 and 50 appointments, comparing:

* ``pandas``: ``build_features(appointments_to_frame(...))`` + model
* ``fast``  : ``appointments_to_matrix(...)`` + model (what the service uses)

This isolates the inference optimisation from network and proxy overhead.

Usage: ``python -m bench.microbench --repeat 300``
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from bench.payloads import appointment_pool
from noshow.config import ServeSettings
from noshow.features import appointments_to_frame, appointments_to_matrix, build_features
from noshow.model import NoShowModel
from noshow.schemas import Appointment


def _median_ms(fn: Callable[[], object], repeat: int) -> float:
    """Median wall time of ``fn`` in milliseconds over ``repeat`` calls."""
    fn()  # warm-up
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    """Print a Markdown table of median latencies per path and input size."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=300)
    args = parser.parse_args()

    settings = ServeSettings.from_env()
    model = NoShowModel.load(Path(settings.model_dir), n_threads=1)
    pool = [Appointment.model_validate(p) for p in appointment_pool()]

    print("| rows | pandas features+model (ms) | fast features+model (ms) | of which model only (ms) |")
    print("|---|---|---|---|")
    for n in (1, 32, 50):
        rows = pool[:n]
        matrix = appointments_to_matrix(rows)
        pandas_ms = _median_ms(
            lambda rows=rows: model.predict_proba(build_features(appointments_to_frame(rows))), args.repeat
        )
        fast_ms = _median_ms(lambda rows=rows: model.predict_proba(appointments_to_matrix(rows)), args.repeat)
        model_ms = _median_ms(lambda matrix=matrix: model.predict_proba(matrix), args.repeat)
        print(f"| {n} | {pandas_ms:.3f} | {fast_ms:.3f} | {model_ms:.3f} |")


if __name__ == "__main__":
    main()
