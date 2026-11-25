"""Unit tests for the documented risk thresholds."""

from __future__ import annotations

import math

import pytest

from noshow.risk import HIGH_THRESHOLD, MEDIUM_THRESHOLD, RECOMMENDED_ACTIONS, recommended_action, risk_level


def test_documented_threshold_values() -> None:
    # If these change, README and docs/04-serving.md must change too.
    assert MEDIUM_THRESHOLD == 0.20
    assert HIGH_THRESHOLD == 0.45


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.1999, "low"),
        (0.20, "medium"),  # lower bound inclusive
        (0.4499, "medium"),
        (0.45, "high"),  # lower bound inclusive
        (1.0, "high"),
    ],
)
def test_risk_level_boundaries(probability: float, expected: str) -> None:
    assert risk_level(probability) == expected


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan])
def test_out_of_range_probability_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        risk_level(bad)


def test_every_level_has_an_action() -> None:
    for level in ("low", "medium", "high"):
        assert recommended_action(level) == RECOMMENDED_ACTIONS[level]
        assert recommended_action(level)
