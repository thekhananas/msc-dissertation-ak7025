"""Hand-calculated check of CSEDM uncertainty-based error review."""

import numpy as np

from socratic_tutor.csedm_study.evaluation import (
    PredictionRecord,
    learner_bootstrap_differences,
    triage_summary,
)


def test_uncertainty_triage_matches_hand_calculated_random_contrast() -> None:
    records = (
        _record(0, "a", "learner-a", True, 0.49),
        _record(0, "b", "learner-b", False, 0.51),
        _record(0, "c", "learner-a", True, 0.90),
        _record(0, "d", "learner-b", False, 0.10),
        _record(1, "e", "learner-c", True, 0.48),
        _record(1, "f", "learner-d", True, 0.52),
        _record(1, "g", "learner-c", True, 0.85),
        _record(1, "h", "learner-d", True, 0.10),
    )

    summary = triage_summary(records, 0.5)
    first = learner_bootstrap_differences(
        records,
        summary.selected_target_ids,
        budget_fraction=0.5,
        repetitions=100,
        seed=17,
    )
    second = learner_bootstrap_differences(
        records,
        summary.selected_target_ids,
        budget_fraction=0.5,
        repetitions=100,
        seed=17,
    )

    assert summary.eligible_count == 8
    assert summary.selected_count == 4
    assert summary.error_count == 4
    assert summary.selected_error_count == 3
    assert summary.expected_random_selected_error_count == 2.0
    assert summary.captured_error_recall == 0.75
    assert summary.expected_random_captured_error_recall == 0.5
    assert summary.captured_error_recall_difference == 0.25
    assert np.array_equal(first, second)


def _record(
    fold_id: int,
    target_id: str,
    learner_id: str,
    first_correct: bool,
    probability_correct: float,
) -> PredictionRecord:
    return PredictionRecord(
        model_id="logistic_learner_history",
        fold_id=fold_id,
        target_id=target_id,
        learner_id=learner_id,
        problem_id="problem-a",
        first_correct=first_correct,
        probability_correct=probability_correct,
    )
