"""Recursive leakage checks for decision-time benchmark payloads."""

from collections.abc import Mapping, Sequence
from typing import cast

_FORBIDDEN_FRAGMENTS = (
    "criterion",
    "demonstrated_mastery",
    "label_rationale",
    "rubric",
    "reviewer",
    "adjudication",
    "evidence_pattern",
    "misconception_id",
    "transfer_case_id",
    "false_public_mastery",
    "supported_mastery",
    "honest_non_mastery",
    "underconfidence",
)


def find_public_payload_violations(value: object, path: str = "$") -> tuple[str, ...]:
    """Return locations containing evaluator-only names or string values."""

    violations: list[str] = []
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, child in mapping.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            lowered = key_text.casefold()
            if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
                violations.append(child_path)
            violations.extend(find_public_payload_violations(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[object], value)
        for index, child in enumerate(sequence):
            violations.extend(find_public_payload_violations(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(fragment in lowered for fragment in _FORBIDDEN_FRAGMENTS):
            violations.append(path)
    return tuple(violations)


def assert_public_payload_safe(value: object) -> None:
    """Reject a public payload containing evaluator-only data or references."""

    violations = find_public_payload_violations(value)
    if violations:
        joined = ", ".join(violations)
        raise ValueError(f"Evaluator-only content appeared in public payload: {joined}")
