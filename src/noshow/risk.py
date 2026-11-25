"""Business rules that turn a probability into a risk level and an action.

The model only outputs a number. A clinic needs a decision: "do nothing",
"ask for confirmation" or "call the patient". Keeping these rules in their own
module makes them easy to read, test and change without retraining.

Thresholds (documented in the README and docs/04-serving.md):

* ``low``    : probability <  0.20  -> standard automatic reminder
* ``medium`` : 0.20 <= probability < 0.45 -> request explicit confirmation
* ``high``   : probability >= 0.45 -> phone call + consider waitlist/overbooking

Lower bounds are inclusive: exactly 0.20 is ``medium``, exactly 0.45 is
``high``. The values are a demo choice tied to the synthetic base rate
(roughly 1 in 5 appointments is a no-show), not a clinical recommendation; in a
real clinic they would be tuned to the cost of a missed slot versus the cost
of contacting the patient.
"""

from __future__ import annotations

from noshow.schemas import RiskLevel

MEDIUM_THRESHOLD = 0.20
HIGH_THRESHOLD = 0.45

RECOMMENDED_ACTIONS: dict[str, str] = {
    "low": "Send the standard automatic reminder.",
    "medium": "Request explicit confirmation (SMS/WhatsApp) 48h before the appointment.",
    "high": "Call the patient to confirm and consider offering the slot to the waitlist.",
}


def risk_level(probability: float) -> RiskLevel:
    """Map a no-show probability to ``low`` / ``medium`` / ``high``.

    Args:
        probability: Model output in ``[0, 1]``.

    Returns:
        The risk bucket. Lower bounds are inclusive.

    Raises:
        ValueError: If ``probability`` is outside ``[0, 1]`` (or NaN).
    """
    if not 0.0 <= probability <= 1.0:  # also False for NaN
        raise ValueError(f"probability must be in [0, 1], got {probability!r}")
    if probability >= HIGH_THRESHOLD:
        return "high"
    if probability >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def recommended_action(level: RiskLevel) -> str:
    """Return the operational action the clinic should take for a risk level."""
    return RECOMMENDED_ACTIONS[level]
