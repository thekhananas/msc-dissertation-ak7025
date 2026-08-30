# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Frozen prediction models for the CSEDM companion study."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from socratic_tutor.csedm_study.analysis_plan import LogisticModelSpecification
from socratic_tutor.csedm_study.features import CSEDMAdapterError

TRAINING_LABEL_PREVALENCE = "training_label_prevalence"
TRAINING_PROBLEM_FREQUENCY = "training_problem_frequency"
LOGISTIC_LEARNER_HISTORY = "logistic_learner_history"
MODEL_IDS = (
    TRAINING_LABEL_PREVALENCE,
    TRAINING_PROBLEM_FREQUENCY,
    LOGISTIC_LEARNER_HISTORY,
)


@dataclass(frozen=True, slots=True)
class ModelInput:
    """One restricted feature row consumed by a prediction model."""

    fold_id: int
    target_id: str
    learner_id: str
    problem_id: str
    first_correct: bool
    feature_values: tuple[float, ...]
    source_record_hash: str


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """One out-of-fold probability and its derived decision values."""

    model_id: str
    fold_id: int
    target_id: str
    learner_id: str
    problem_id: str
    first_correct: bool
    probability_correct: float
    predicted_correct: bool
    uncertainty: float
    source_record_hash: str


@dataclass(frozen=True, slots=True)
class FoldPredictionResult:
    """Predictions and aggregate fitting details for one official fold."""

    predictions: dict[str, tuple[ModelPrediction, ...]]
    training_prevalence: float
    problem_frequency_fallback_count: int
    logistic_iterations: int


def predict_fold(
    training_rows: tuple[ModelInput, ...],
    test_rows: tuple[ModelInput, ...],
    specification: LogisticModelSpecification,
) -> FoldPredictionResult:
    """Fit the three frozen models and predict one untouched test fold."""

    _validate_fold_rows(training_rows, test_rows)
    training_prevalence = sum(row.first_correct for row in training_rows) / len(training_rows)
    problem_counts = Counter(row.problem_id for row in training_rows)
    problem_successes = Counter(row.problem_id for row in training_rows if row.first_correct)
    problem_probabilities = {
        problem_id: problem_successes[problem_id] / count
        for problem_id, count in problem_counts.items()
    }
    problem_fallback_count = sum(row.problem_id not in problem_probabilities for row in test_rows)

    logistic_probabilities, logistic_iterations = _logistic_probabilities(
        training_rows,
        test_rows,
        specification,
    )
    probabilities = {
        TRAINING_LABEL_PREVALENCE: tuple(training_prevalence for _ in test_rows),
        TRAINING_PROBLEM_FREQUENCY: tuple(
            problem_probabilities.get(row.problem_id, training_prevalence) for row in test_rows
        ),
        LOGISTIC_LEARNER_HISTORY: logistic_probabilities,
    }
    predictions = {
        model_id: tuple(
            _prediction(model_id, row, probability, specification.classification_threshold)
            for row, probability in zip(test_rows, model_probabilities, strict=True)
        )
        for model_id, model_probabilities in probabilities.items()
    }
    return FoldPredictionResult(
        predictions=predictions,
        training_prevalence=training_prevalence,
        problem_frequency_fallback_count=problem_fallback_count,
        logistic_iterations=logistic_iterations,
    )


def with_label(row: ModelInput, first_correct: bool) -> ModelInput:
    """Return a test fixture with a changed outcome and unchanged model inputs."""

    return replace(row, first_correct=first_correct)


def _logistic_probabilities(
    training_rows: tuple[ModelInput, ...],
    test_rows: tuple[ModelInput, ...],
    specification: LogisticModelSpecification,
) -> tuple[tuple[float, ...], int]:
    labels = np.asarray([row.first_correct for row in training_rows], dtype=np.int8)
    if len(np.unique(labels)) != 2:
        raise CSEDMAdapterError("Each training fold must contain both outcome classes")
    feature_count = len(training_rows[0].feature_values)
    all_rows = (*training_rows, *test_rows)
    if feature_count == 0 or any(len(row.feature_values) != feature_count for row in all_rows):
        raise CSEDMAdapterError("CSEDM model rows have inconsistent feature dimensions")
    training_features = np.asarray(
        [row.feature_values for row in training_rows],
        dtype=np.float64,
    )
    test_features = np.asarray([row.feature_values for row in test_rows], dtype=np.float64)
    if not np.isfinite(training_features).all() or not np.isfinite(test_features).all():
        raise CSEDMAdapterError("CSEDM model features must be finite")

    model = LogisticRegression(
        C=specification.regularisation_c,
        solver=specification.solver,
        penalty=specification.penalty,
        fit_intercept=specification.fit_intercept,
        class_weight=None,
        max_iter=specification.maximum_iterations,
        tol=specification.tolerance,
        random_state=specification.random_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(training_features, labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise CSEDMAdapterError("The frozen logistic model did not converge")
    probabilities = tuple(float(value) for value in model.predict_proba(test_features)[:, 1])
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise CSEDMAdapterError("The frozen logistic model returned an invalid probability")
    return probabilities, int(model.n_iter_[0])


def _prediction(
    model_id: str,
    row: ModelInput,
    probability: float,
    threshold: float,
) -> ModelPrediction:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise CSEDMAdapterError(f"{model_id} returned an invalid probability")
    return ModelPrediction(
        model_id=model_id,
        fold_id=row.fold_id,
        target_id=row.target_id,
        learner_id=row.learner_id,
        problem_id=row.problem_id,
        first_correct=row.first_correct,
        probability_correct=probability,
        predicted_correct=probability >= threshold,
        uncertainty=abs(probability - 0.5),
        source_record_hash=row.source_record_hash,
    )


def _validate_fold_rows(
    training_rows: tuple[ModelInput, ...],
    test_rows: tuple[ModelInput, ...],
) -> None:
    if not training_rows or not test_rows:
        raise CSEDMAdapterError("Each fold requires non-empty training and test rows")
    fold_ids = {row.fold_id for row in (*training_rows, *test_rows)}
    if len(fold_ids) != 1:
        raise CSEDMAdapterError("Model rows from different folds cannot be mixed")
    training_learners = {row.learner_id for row in training_rows}
    test_learners = {row.learner_id for row in test_rows}
    if training_learners & test_learners:
        raise CSEDMAdapterError("Model training and test learners must remain separate")
