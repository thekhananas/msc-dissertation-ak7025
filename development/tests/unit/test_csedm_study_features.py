"""Leakage tests for CSEDM learner-history features."""

from socratic_tutor.csedm_study.features import (
    PredictionTarget,
    build_fold_features,
    parse_events,
)


def test_features_exclude_later_events_and_each_training_rows_own_label() -> None:
    events = parse_events(
        (
            _event("learner-a", order=1, event_type="Submit", correct="TRUE"),
            _event("learner-a", order=2, event_type="X-HintRequest", correct="FALSE"),
            _event("learner-a", order=10, event_type="Submit", correct="FALSE"),
        )
    )
    training = (
        PredictionTarget("learner-a", "target-problem", 5, True),
        PredictionTarget("learner-b", "target-problem", 5, False),
    )
    test = (PredictionTarget("learner-c", "target-problem", 5, False),)

    training_features, test_features = build_fold_features(training, test, events)

    assert training_features[0].prior_event_count == 2.0
    assert training_features[0].prior_submission_count == 1.0
    assert training_features[0].prior_correctness_rate == 1.0
    assert training_features[0].prior_hint_request_rate == 0.5
    assert training_features[0].training_learner_target_problem_success_rate == 0.0
    assert training_features[1].training_learner_target_problem_success_rate == 1.0
    assert test_features[0].training_learner_target_problem_success_rate == 0.5


def _event(
    learner_id: str,
    *,
    order: int,
    event_type: str,
    correct: str,
) -> dict[str, str]:
    return {
        "EventType": event_type,
        "EventID": f"event-{order}",
        "Order": str(order),
        "SubjectID": learner_id,
        "ToolInstances": "ITAP; Python",
        "CodeStateID": f"code-{order}",
        "ServerTimestamp": "2016-01-01T00:00:00",
        "ProblemID": "prior-problem",
        "Correct": correct,
    }
