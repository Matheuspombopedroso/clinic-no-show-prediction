"""Tests for training, the artifact format and model behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from noshow.model import METADATA_FILE, NoShowModel
from noshow.service import PredictionService
from noshow.train import time_split
from tests.conftest import make_appointment


def test_time_split_has_no_overlap() -> None:
    df = pd.DataFrame({"scheduled_at": pd.date_range("2025-01-01", periods=100, freq="D")[::-1]})
    train, test = time_split(df, 0.2)
    assert len(test) == 20
    assert train["scheduled_at"].max() <= test["scheduled_at"].min()


def test_artifact_contains_version_and_metrics(trained_artifact: Path) -> None:
    meta = json.loads((trained_artifact / METADATA_FILE).read_text())
    assert meta["version"] == "9.9.9"
    for key in ("roc_auc", "pr_auc", "brier"):
        assert 0.0 <= meta["test_metrics"][key] <= 1.0


def test_model_beats_chance_on_synthetic_data(trained_artifact: Path) -> None:
    meta = json.loads((trained_artifact / METADATA_FILE).read_text())
    assert meta["test_metrics"]["roc_auc"] > 0.6


def test_feature_mismatch_is_refused(trained_artifact: Path, tmp_path: Path) -> None:
    for name in ("model.json", METADATA_FILE):
        (tmp_path / name).write_bytes((trained_artifact / name).read_bytes())
    meta = json.loads((tmp_path / METADATA_FILE).read_text())
    meta["features"] = [*meta["features"], "unexpected"]
    (tmp_path / METADATA_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature mismatch"):
        NoShowModel.load(tmp_path)


def test_confirmation_lowers_risk(real_model: NoShowModel) -> None:
    """Sanity check: the model learned the strongest synthetic signal."""
    service = PredictionService(real_model)
    unconfirmed, confirmed = service.predict_probabilities(
        [make_appointment(confirmed=False), make_appointment(confirmed=True)]
    )
    assert confirmed < unconfirmed
