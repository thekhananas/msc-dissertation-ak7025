"""Deterministic acceptance tests for the first tutoring turn."""

import pytest
from pydantic import ValidationError

from socratic_tutor.contracts import (
    EvidenceCategory,
    StudentSubmission,
    TrackerState,
    TutorAction,
)
from socratic_tutor.guardrails import check_prompt
from socratic_tutor.tasks import load_task
from socratic_tutor.tracking import initial_tracker_state, update_tracker
from socratic_tutor.turn import run_turn


@pytest.mark.parametrize(
    ("response", "category", "mastery", "action"),
    [
        (
            "It prints [1, 2, 3] because alias and numbers reference the same list.",
            EvidenceCategory.CORRECT,
            0.7,
            TutorAction.TRANSFER,
        ),
        (
            "It prints [1, 2] because alias is an independent copy.",
            EvidenceCategory.MISCONCEPTION,
            0.3,
            TutorAction.HINT,
        ),
        (
            "It prints [1, 2, 3], but alias is a separate list copy.",
            EvidenceCategory.CONFLICTING,
            0.4,
            TutorAction.CLARIFY,
        ),
        (
            "I think append changes something but I cannot explain which value is printed.",
            EvidenceCategory.UNCERTAIN,
            0.5,
            TutorAction.PROBE,
        ),
        ("   ", EvidenceCategory.EMPTY, 0.5, TutorAction.ENCOURAGE),
    ],
)
def test_run_turn_covers_each_evidence_path(
    response: str,
    category: EvidenceCategory,
    mastery: float,
    action: TutorAction,
) -> None:
    task = load_task()
    initial = initial_tracker_state(task.concept)

    result = run_turn(task, StudentSubmission(response_text=response), initial)

    assert result.evidence.category is category
    assert result.tracker_before == initial
    assert result.tracker_after.mastery_probability == mastery
    assert result.tracker_after.observations == 1
    assert result.decision.action is action
    assert result.next_prompt
    assert result.guardrail.safe is True


def test_tracker_is_bounded_after_repeated_evidence() -> None:
    task = load_task()
    state = initial_tracker_state(task.concept)
    correct = run_turn(
        task,
        StudentSubmission(response_text="[1, 2, 3], because both names reference the same list"),
        state,
    ).evidence

    for _ in range(20):
        state = update_tracker(state, correct)

    assert state.mastery_probability == 1.0


def test_repeated_evidence_advances_prompt_variants() -> None:
    task = load_task()
    state = initial_tracker_state(task.concept)
    submission = StudentSubmission(
        response_text="[1, 2, 3], because both names reference the same list"
    )
    prompts: list[str] = []

    for _ in range(3):
        result = run_turn(task, submission, state)
        prompts.append(result.next_prompt)
        state = result.tracker_after

    assert len(set(prompts)) == 3


def test_task_public_view_does_not_expose_evaluator_rules() -> None:
    payload = load_task().public_view().model_dump()

    assert "evidence_rules" not in payload


def test_unknown_task_id_fails_explicitly() -> None:
    with pytest.raises(KeyError, match="Unknown task"):
        load_task("not-a-task")


def test_contracts_reject_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StudentSubmission.model_validate({"response_text": "answer", "hidden_truth": True})


def test_tracker_contract_rejects_non_finite_probability() -> None:
    with pytest.raises(ValidationError):
        TrackerState(concept="lists", mastery_probability=float("nan"), observations=0)


def test_guardrail_replaces_direct_answer_leakage() -> None:
    result = check_prompt("The answer is [1, 2, 3].")

    assert result.safe is False
    assert result.violations
    assert "[1, 2, 3]" not in result.output_prompt


def test_turn_rejects_mismatched_tracker_concept() -> None:
    with pytest.raises(ValueError, match="concept"):
        run_turn(
            load_task(),
            StudentSubmission(response_text="unsure"),
            TrackerState(concept="loops", mastery_probability=0.5, observations=0),
        )
