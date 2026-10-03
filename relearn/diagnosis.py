"""Person 1 -> Person 2 diagnosis interface.

Only the three core fields are part of the stable contract. Extra keys
in the input are accepted and ignored. No ML code lives here.
"""

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .taxonomy import MISCONCEPTIONS

REQUIRED_FIELDS = ("misconception_id", "misconception", "confidence")


class DiagnosisError(ValueError):
    """Raised when a diagnosis payload violates the contract."""


@dataclass(frozen=True)
class Diagnosis:
    misconception_id: str
    misconception: str
    confidence: float


def parse_diagnosis(data: Mapping[str, Any]) -> Diagnosis:
    """Validate Person 1's output dict and return an immutable Diagnosis."""
    if not isinstance(data, Mapping):
        raise DiagnosisError(
            f"diagnosis must be a dict, got {type(data).__name__}"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise DiagnosisError(f"missing required field(s): {', '.join(missing)}")

    mid = data["misconception_id"]
    if not isinstance(mid, str) or mid not in MISCONCEPTIONS:
        raise DiagnosisError(
            f"unknown misconception_id {mid!r}; valid IDs: "
            f"{', '.join(MISCONCEPTIONS)}"
        )

    name = data["misconception"]
    expected = MISCONCEPTIONS[mid]
    if not isinstance(name, str) or name != expected:
        raise DiagnosisError(
            f"misconception {name!r} does not match {mid}; expected {expected!r}"
        )

    conf = data["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        raise DiagnosisError(
            f"confidence must be a number, got {type(conf).__name__} ({conf!r})"
        )
    if not math.isfinite(conf):
        raise DiagnosisError(f"confidence must be finite, got {conf!r}")
    if not 0.0 <= conf <= 1.0:
        raise DiagnosisError(f"confidence must be within 0.0-1.0, got {conf!r}")

    return Diagnosis(mid, name, float(conf))
