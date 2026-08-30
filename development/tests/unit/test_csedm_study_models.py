"""Leakage test for the frozen CSEDM prediction models."""

from socratic_tutor.csedm_study.analysis_plan import LogisticModelSpecification
from socratic_tutor.csedm_study.models import (
    LOGISTIC_LEARNER_HISTORY,
    MODEL_IDS,
    TRAINING_LABEL_PREVALENCE,
    TRAINING_PROBLEM_FREQUENCY,
    ModelInput,
    predict_fold,
    with_label,
)


def test_test_labels_cannot_change_any_model_probability() -> None:
    training = tuple(
        _row(
            target_id=f"training-{index}",
            learner_id=f"training-learner-{index}",
            problem_id="problem-a" if index < 4 else "problem-b",
            first_correct=index in {1, 2, 3, 5},
            feature_value=float(index - 3),
        )
        for index in range(6)
    )
    test = (
        _row("test-a", "test-learner-a", "problem-a", False, -0.5),
        _row("test-b", "test-learner-b", "problem-b", True, 0.5),
        _row("test-c", "test-learner-c", "unseen-problem", False, 1.5),
    )
    flipped_test = tuple(with_label(row, not row.first_correct) for row in test)

    original = predict_fold(training, test, _specification())
    flipped = predict_fold(training, flipped_test, _specification())

    assert tuple(original.predictions) == MODEL_IDS
    for model_id in MODEL_IDS:
        assert [row.probability_correct for row in original.predictions[model_id]] == [
            row.probability_correct for row in flipped.predictions[model_id]
        ]
    assert {row.probability_correct for row in original.predictions[TRAINING_LABEL_PREVALENCE]} == {
        2 / 3
    }
    assert [
        row.probability_correct for row in original.predictions[TRAINING_PROBLEM_FREQUENCY]
    ] == [0.75, 0.5, 2 / 3]
    assert original.problem_frequency_fallback_count == 1
    assert original.logistic_iterations > 0
    assert len(original.predictions[LOGISTIC_LEARNER_HISTORY]) == len(test)


def _row(
    target_id: str,
    learner_id: str,
    problem_id: str,
    first_correct: bool,
    feature_value: float,
) -> ModelInput:
    return ModelInput(
        fold_id=0,
        target_id=target_id,
        learner_id=learner_id,
        problem_id=problem_id,
        first_correct=first_correct,
        feature_values=(feature_value,) * 12,
        source_record_hash="a" * 64,
    )


def _specification() -> LogisticModelSpecification:
    return LogisticModelSpecification(
        implementation="sklearn.linear_model.LogisticRegression",
        solver="lbfgs",
        penalty="l2",
        regularisation_c=1.0,
        fit_intercept=True,
        class_weight="none",
        maximum_iterations=1000,
        tolerance=1e-8,
        random_seed=9052901,
        classification_threshold=0.5,
        numeric_scaling="training_fold_standard_score",
        missing_value_handling="training_fold_median_plus_indicator",
    )
