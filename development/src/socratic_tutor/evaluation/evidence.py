"""Deterministic evidence classification for authored development tasks."""

from socratic_tutor.contracts import (
    Evidence,
    EvidenceCategory,
    StudentSubmission,
    TaskDefinition,
)


def _matching_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(marker for marker in markers if marker.casefold() in text)


def classify_evidence(task: TaskDefinition, submission: StudentSubmission) -> Evidence:
    """Classify one response using transparent, task-authored string rules."""

    normalized = " ".join(submission.response_text.casefold().split())
    if not normalized:
        return Evidence(
            category=EvidenceCategory.EMPTY,
            confidence=1.0,
            rationale="No response was provided.",
        )

    rules = task.evidence_rules
    correct_output = _matching_markers(normalized, rules.correct_output_markers)
    correct_explanation = _matching_markers(normalized, rules.correct_explanation_markers)
    misconception_output = _matching_markers(normalized, rules.misconception_output_markers)
    misconception_explanation = _matching_markers(
        normalized, rules.misconception_explanation_markers
    )

    supports_correct = bool(correct_output and correct_explanation)
    supports_misconception = bool(misconception_output or misconception_explanation)
    observed = tuple(
        dict.fromkeys(
            (
                *correct_output,
                *correct_explanation,
                *misconception_output,
                *misconception_explanation,
            )
        )
    )

    if supports_correct and supports_misconception:
        category = EvidenceCategory.CONFLICTING
        confidence = 0.8
        rationale = "The response contains both correct and misconception signals."
    elif supports_correct:
        category = EvidenceCategory.CORRECT
        confidence = 0.9
        rationale = "The predicted output and explanation both match the reviewed rules."
    elif supports_misconception:
        category = EvidenceCategory.MISCONCEPTION
        confidence = 0.9
        rationale = "The response matches the authored mutable-copy misconception."
    else:
        category = EvidenceCategory.UNCERTAIN
        confidence = 0.5
        rationale = "The response does not provide enough recognised evidence to classify."

    return Evidence(
        category=category,
        confidence=confidence,
        rationale=rationale,
        observed_signals=observed,
    )
