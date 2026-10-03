"""Intervention knowledge base: loading, validation and lookup.

Content lives in data/interventions.json. This module only loads and
validates it; it contains no adaptive logic and no ML code.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .taxonomy import MISCONCEPTIONS

DEFAULT_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "interventions.json"

TEXT_FIELDS = ("explanation", "example", "hint", "common_error")
REQUIRED_FIELDS = ("misconception_id", "misconception") + TEXT_FIELDS + (
    "follow_up_questions",
)


class InterventionError(ValueError):
    """Raised when the knowledge base or an entry is malformed."""


class InterventionNotFoundError(InterventionError):
    """Raised when no intervention exists for a requested misconception ID."""


@dataclass(frozen=True)
class Intervention:
    misconception_id: str
    misconception: str
    explanation: str
    example: str
    hint: str
    common_error: str
    follow_up_questions: Tuple[str, ...]


def _non_empty_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise InterventionError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise InterventionError(f"{label} must not be empty")
    return value


def parse_intervention(entry: Mapping[str, Any]) -> Intervention:
    """Validate one knowledge-base entry and return an Intervention.

    Extra fields are ignored.
    """
    if not isinstance(entry, Mapping):
        raise InterventionError(
            f"intervention entry must be an object, got {type(entry).__name__}"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise InterventionError(
            f"intervention entry missing required field(s): {', '.join(missing)}"
        )

    mid = entry["misconception_id"]
    if not isinstance(mid, str) or mid not in MISCONCEPTIONS:
        raise InterventionError(
            f"unknown misconception_id {mid!r}; valid IDs: "
            f"{', '.join(MISCONCEPTIONS)}"
        )

    name = entry["misconception"]
    expected = MISCONCEPTIONS[mid]
    if not isinstance(name, str) or name != expected:
        raise InterventionError(
            f"{mid}: misconception {name!r} does not match canonical name "
            f"{expected!r}"
        )

    texts = {f: _non_empty_str(entry[f], f"{mid}: {f}") for f in TEXT_FIELDS}

    questions = entry["follow_up_questions"]
    if not isinstance(questions, (list, tuple)):
        raise InterventionError(
            f"{mid}: follow_up_questions must be a list, "
            f"got {type(questions).__name__}"
        )
    if not questions:
        raise InterventionError(f"{mid}: follow_up_questions must not be empty")
    checked = tuple(
        _non_empty_str(q, f"{mid}: follow_up_questions[{i}]")
        for i, q in enumerate(questions)
    )

    return Intervention(mid, name, follow_up_questions=checked, **texts)


def load_interventions(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Intervention]:
    """Load and validate the whole knowledge base, keyed by misconception ID."""
    kb_path = Path(path) if path is not None else DEFAULT_KB_PATH
    try:
        with open(kb_path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise InterventionError(f"knowledge base file not found: {kb_path}")
    except json.JSONDecodeError as e:
        raise InterventionError(f"knowledge base is not valid JSON ({kb_path}): {e}")

    if not isinstance(raw, list):
        raise InterventionError(
            f"knowledge base must be a JSON list of entries, got {type(raw).__name__}"
        )

    result: Dict[str, Intervention] = {}
    for i, entry in enumerate(raw):
        try:
            item = parse_intervention(entry)
        except InterventionError as e:
            raise InterventionError(f"entry {i}: {e}") from None
        if item.misconception_id in result:
            raise InterventionError(
                f"entry {i}: duplicate misconception_id {item.misconception_id!r}"
            )
        result[item.misconception_id] = item
    return result


def get_intervention(
    misconception_id: str, path: Optional[Union[str, Path]] = None
) -> Intervention:
    """Return the intervention for a misconception ID.

    Raises InterventionNotFoundError if the ID is not a canonical ID or has
    no content in the knowledge base yet.
    """
    if not isinstance(misconception_id, str) or misconception_id not in MISCONCEPTIONS:
        raise InterventionNotFoundError(
            f"unknown misconception_id {misconception_id!r}; valid IDs: "
            f"{', '.join(MISCONCEPTIONS)}"
        )
    kb = load_interventions(path)
    if misconception_id not in kb:
        raise InterventionNotFoundError(
            f"no intervention content for {misconception_id} "
            f"({MISCONCEPTIONS[misconception_id]}) in the knowledge base"
        )
    return kb[misconception_id]
