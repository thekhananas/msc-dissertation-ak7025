"""Deterministic acceptance tests for the first tutoring turn."""

import pytest
from pydantic import ValidationError

from socratic_tutor.contracts import (
    Evidence,
    EvidenceCategory,
    StudentSubmission,
    TrackerState,
    TutorAction,
)
from socratic_tutor.guardrails import check_prompt
from socratic_tutor.policies import choose_action
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


def test_incorrect_evidence_does_not_claim_a_specific_misconception() -> None:
    task = load_task()
    evidence = Evidence(
        category=EvidenceCategory.INCORRECT,
        confidence=1.0,
        rationale="All executable checks failed.",
    )

    tracker = update_tracker(initial_tracker_state(task.concept), evidence)
    decision = choose_action(evidence)

    assert tracker.mastery_probability == 0.3
    assert tracker.last_evidence is EvidenceCategory.INCORRECT
    assert decision.action is TutorAction.HINT
    assert "assuming a misconception" in decision.rationale


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


def test_repeated_uncertainty_moves_from_probe_to_clarify() -> None:
    task = load_task()
    state = initial_tracker_state(task.concept)
    submission = StudentSubmission(response_text="I am not sure.")
    actions: list[TutorAction] = []

    for _ in range(4):
        result = run_turn(task, submission, state)
        actions.append(result.decision.action)
        state = result.tracker_after

    assert actions == [
        TutorAction.PROBE,
        TutorAction.PROBE,
        TutorAction.PROBE,
        TutorAction.CLARIFY,
    ]


def test_matching_output_without_explanation_requests_clarification() -> None:
    task = load_task("none-versus-falsy")
    result = run_turn(
        task,
        StudentSubmission(response_text="score=0\nmissing"),
        initial_tracker_state(task.concept),
    )

    assert result.evidence.category is EvidenceCategory.INCOMPLETE
    assert result.tracker_after.mastery_probability == 0.5
    assert result.decision.action is TutorAction.CLARIFY
    assert "explanation is missing" in result.evidence.rationale
    assert "identified both printed lines" in result.next_prompt


def test_follow_up_answer_is_checked_against_its_current_question() -> None:
    task = load_task("none-versus-falsy")
    state = initial_tracker_state(task.concept)
    incomplete = run_turn(
        task,
        StudentSubmission(response_text="score=0\nmissing"),
        state,
    )
    follow_up = run_turn(
        task,
        StudentSubmission(response_text="False for zero and true for None."),
        incomplete.tracker_after,
        question_prompt="For the first call only, what Boolean value does score is None produce when score is zero?",
    )

    assert follow_up.evidence.category is EvidenceCategory.CONFLICTING
    assert follow_up.decision.action is TutorAction.CLARIFY


@pytest.mark.parametrize(
    ("response", "category", "mastery", "action", "prompt_fragment"),
    [
        (
            "It prints score=0 and then missing because the condition is an identity check.",
            EvidenceCategory.CORRECT,
            0.7,
            TutorAction.TRANSFER,
            "empty string",
        ),
        (
            "It prints missing twice because both are falsy, like using if not score.",
            EvidenceCategory.MISCONCEPTION,
            0.3,
            TutorAction.HINT,
            "general truthiness",
        ),
    ],
)
def test_none_versus_falsy_task_has_answer_dependent_paths(
    response: str,
    category: EvidenceCategory,
    mastery: float,
    action: TutorAction,
    prompt_fragment: str,
) -> None:
    task = load_task("none-versus-falsy")

    result = run_turn(
        task,
        StudentSubmission(response_text=response),
        initial_tracker_state(task.concept),
    )

    assert result.evidence.category is category
    assert result.tracker_after.mastery_probability == mastery
    assert result.decision.action is action
    assert prompt_fragment in result.next_prompt


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
