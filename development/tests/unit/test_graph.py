"""LangGraph boundary tests for the bounded tutoring turn."""

from socratic_tutor.contracts import StudentSubmission
from socratic_tutor.graph import run_graph_turn
from socratic_tutor.tasks import load_task
from socratic_tutor.tracking import initial_tracker_state
from socratic_tutor.turn import run_turn


def test_graph_matches_the_pure_turn_contract() -> None:
    task = load_task()
    tracker = initial_tracker_state(task.concept)
    submission = StudentSubmission(
        response_text="It prints [1, 2, 3] because both names reference the same list."
    )

    assert run_graph_turn(task, submission, tracker) == run_turn(task, submission, tracker)
