"""Side-effect-free composition of one deterministic tutoring turn."""

from socratic_tutor.contracts import StudentSubmission, TaskDefinition, TrackerState, TurnResult
from socratic_tutor.evaluation import classify_evidence
from socratic_tutor.generation import generate_prompt
from socratic_tutor.guardrails import check_prompt
from socratic_tutor.policies import choose_action
from socratic_tutor.tracking import update_tracker


def run_turn(
    task: TaskDefinition,
    submission: StudentSubmission,
    tracker: TrackerState,
) -> TurnResult:
    """Produce a complete turn result without I/O or hidden mutable state."""

    if tracker.concept != task.concept:
        raise ValueError("Tracker concept must match the task concept")

    evidence = classify_evidence(task, submission)
    tracker_after = update_tracker(tracker, evidence)
    decision = choose_action(evidence)
    candidate_prompt = generate_prompt(decision, tracker_after.observations)
    guardrail = check_prompt(candidate_prompt)
    return TurnResult(
        evidence=evidence,
        tracker_before=tracker,
        tracker_after=tracker_after,
        decision=decision,
        next_prompt=guardrail.output_prompt,
        guardrail=guardrail,
    )
