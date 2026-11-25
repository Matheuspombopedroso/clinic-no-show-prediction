"""Runtime configuration read from environment variables.

Everything that changes between environments (where the model lives, how many
replicas to run, how aggressive request batching is) is read here, in one
place, so the rest of the code receives plain values instead of reading
``os.environ`` all over the codebase. See ``.env.example`` for the documented
list of variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Default artifact location produced by ``python -m noshow.train``.
DEFAULT_MODEL_DIR = Path("artifacts") / "no-show-xgb" / "1.0.0"


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to ``default``."""
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class ServeSettings:
    """Settings for the Ray Serve deployment.

    Attributes:
        model_dir: Directory containing ``model.json`` and ``metadata.json``.
        num_replicas: Number of model replicas (horizontal scaling knob).
        max_batch_size: Upper bound of requests merged by ``@serve.batch``.
        batch_wait_timeout_s: How long the first request of a batch waits for
            others to arrive before the batch is flushed anyway.
        max_ongoing_requests: Concurrent requests a single replica accepts.
            Must be >= ``max_batch_size`` or batches can never fill up.
        model_threads: Threads XGBoost may use per replica. Kept at 1 so that
            scaling is done with replicas and the 1-vs-N comparison is fair.
        host: HTTP bind host for the Serve proxy.
        port: HTTP port for the Serve proxy.
        access_log: Log one line per request in the proxy and replicas. Off by
            default: at hundreds of requests per second the logging itself
            (formatting + shipping logs to the driver) costs measurable CPU.
    """

    model_dir: Path = DEFAULT_MODEL_DIR
    num_replicas: int = 1
    max_batch_size: int = 32
    batch_wait_timeout_s: float = 0.005
    max_ongoing_requests: int = 64
    model_threads: int = 1
    host: str = "127.0.0.1"
    port: int = 8000
    access_log: bool = False

    @classmethod
    def from_env(cls) -> ServeSettings:
        """Build settings from environment variables (see ``.env.example``)."""
        return cls(
            model_dir=Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))),
            num_replicas=_env_int("NUM_REPLICAS", cls.num_replicas),
            max_batch_size=_env_int("MAX_BATCH_SIZE", cls.max_batch_size),
            batch_wait_timeout_s=_env_float("BATCH_WAIT_TIMEOUT_S", cls.batch_wait_timeout_s),
            max_ongoing_requests=_env_int("MAX_ONGOING_REQUESTS", cls.max_ongoing_requests),
            model_threads=_env_int("MODEL_THREADS", cls.model_threads),
            host=os.getenv("SERVE_HOST", cls.host),
            port=_env_int("SERVE_PORT", cls.port),
            access_log=os.getenv("ACCESS_LOG", "false").lower() in ("1", "true", "yes"),
        )

    def logging_config(self) -> dict[str, object]:
        """Ray Serve ``LoggingConfig`` (as a dict) derived from these settings."""
        return {"enable_access_log": self.access_log, "log_level": "INFO" if self.access_log else "WARNING"}
