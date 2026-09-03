"""Deterministic evidence classification for authored development tasks."""

from socratic_tutor.contracts import (
    Evidence,
    EvidenceCategory,
    StudentSubmission,
    TaskDefinition,
)


def _matching_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(marker for marker in markers if marker.casefold() in text)


def _classify_follow_up(
    task: TaskDefinition,
    submission: StudentSubmission,
    question_prompt: str,
) -> Evidence | None:
    rule = next(
        (
            candidate
            for candidate in task.follow_up_evidence_rules
            if any(
                marker.casefold() in question_prompt.casefold()
                for marker in candidate.question_prompt_markers
            )
        ),
        None,
    )
    if rule is None:
        return None

    normalized = " ".join(submission.response_text.casefold().split())
    correct = _matching_markers(normalized, rule.required_correct_markers)
    misconception = _matching_markers(normalized, rule.misconception_markers)
    observed = tuple(dict.fromkeys((*correct, *misconception)))

    if correct and misconception:
        return Evidence(
            category=EvidenceCategory.CONFLICTING,
            confidence=0.8,
            rationale="The follow up answer contains both correct and incorrect signals.",
            observed_signals=observed,
        )
    if len(correct) == len(rule.required_correct_markers) and not misconception:
        return Evidence(
            category=EvidenceCategory.CORRECT,
            confidence=0.9,
            rationale=rule.rationale_if_correct,
            observed_signals=observed,
        )
    if misconception:
        return Evidence(
            category=EvidenceCategory.MISCONCEPTION,
            confidence=0.9,
            rationale=rule.rationale_if_misconception,
            observed_signals=observed,
        )
    return Evidence(
        category=EvidenceCategory.UNCERTAIN,
        confidence=0.5,
        rationale="The response does not provide enough recognised evidence for this follow up question.",
        observed_signals=observed,
    )


def classify_evidence(
    task: TaskDefinition,
    submission: StudentSubmission,
    *,
    observation_number: int = 0,
    question_prompt: str | None = None,
) -> Evidence:
    """Classify one response using transparent, task-authored string rules."""

    normalized = " ".join(submission.response_text.casefold().split())
    if not normalized:
        return Evidence(
            category=EvidenceCategory.EMPTY,
            confidence=1.0,
            rationale="No response was provided.",
        )

    if question_prompt is not None and question_prompt != task.initial_prompt:
        follow_up = _classify_follow_up(task, submission, question_prompt)
        if follow_up is not None:
            return follow_up
        return Evidence(
            category=EvidenceCategory.UNCERTAIN,
            confidence=0.5,
            rationale="The response does not match the checks for the current tutor question.",
        )

    rules = task.evidence_rules
    correct_output = _matching_markers(normalized, rules.correct_output_markers)
    correct_explanation = _matching_markers(normalized, rules.correct_explanation_markers)
    misconception_output = _matching_markers(normalized, rules.misconception_output_markers)
    misconception_explanation = _matching_markers(
        normalized, rules.misconception_explanation_markers
    )

    has_correct_output = bool(correct_output)
    has_correct_explanation = bool(correct_explanation)
    supports_correct = bool(has_correct_output and has_correct_explanation)
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
    elif has_correct_output:
        if has_correct_explanation:
            category = EvidenceCategory.CORRECT
            confidence = 0.9
            rationale = "The predicted output and explanation both match the reviewed rules."
        else:
            category = EvidenceCategory.INCOMPLETE
            confidence = 0.8
            rationale = (
                "The predicted output matches the reviewed rules, but the explanation is missing."
            )
    elif supports_misconception:
        category = EvidenceCategory.MISCONCEPTION
        confidence = 0.9
        rationale = "The response matches an authored misconception rule for this task."
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
