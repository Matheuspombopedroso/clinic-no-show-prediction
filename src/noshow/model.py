"""Versioned model artifact: save, load and predict.

An artifact is a directory such as ``artifacts/no-show-xgb/1.0.0/`` with:

* ``model.json``    - the XGBoost booster in XGBoost's portable JSON format
* ``metadata.json`` - version, feature list, training metrics, data seed

Storing the feature list next to the weights lets the server refuse to start
if the code's :data:`~noshow.features.FEATURE_COLUMNS` no longer match what the
model was trained on (a silent and very common production bug).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from noshow.features import FEATURE_COLUMNS

MODEL_FILE = "model.json"
METADATA_FILE = "metadata.json"


def save_artifact(booster: xgb.Booster, metadata: dict[str, Any], out_dir: Path) -> Path:
    """Write ``model.json`` + ``metadata.json`` into ``out_dir``.

    Args:
        booster: Trained XGBoost booster.
        metadata: JSON-serialisable dict; must contain ``version``.
        out_dir: Target directory (created if missing).

    Returns:
        The artifact directory.
    """
    if "version" not in metadata:
        raise ValueError("metadata must include a 'version'")
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return out_dir


@dataclass
class NoShowModel:
    """Loaded model plus its metadata, ready for low-latency inference."""

    booster: xgb.Booster
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        """Model version string stored at training time (e.g. ``1.0.0``)."""
        return str(self.metadata["version"])

    @classmethod
    def load(cls, model_dir: Path, n_threads: int = 1) -> NoShowModel:
        """Load an artifact directory written by :func:`save_artifact`.

        Args:
            model_dir: Directory with ``model.json`` and ``metadata.json``.
            n_threads: XGBoost threads for prediction. 1 by default so that a
                replica uses one core and scaling happens through replicas.

        Raises:
            ValueError: If the stored feature list differs from the code's.
        """
        metadata = json.loads((model_dir / METADATA_FILE).read_text())
        if metadata.get("features") != FEATURE_COLUMNS:
            raise ValueError(
                "feature mismatch between model artifact and code: "
                f"{metadata.get('features')} != {FEATURE_COLUMNS}"
            )
        booster = xgb.Booster()
        booster.load_model(model_dir / MODEL_FILE)
        booster.set_param({"nthread": n_threads})
        return cls(booster=booster, metadata=metadata)

    def predict_proba(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return the no-show probability for each feature row.

        Args:
            features: Matrix whose columns follow :data:`FEATURE_COLUMNS`
                (NumPy array from the serving fast path, or a DataFrame).

        ``inplace_predict`` skips building a ``DMatrix``, which matters for
        latency when the input is only a handful of rows.
        """
        if isinstance(features, pd.DataFrame):
            features = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        return np.asarray(self.booster.inplace_predict(features), dtype=float)
