"""Leakage-safe learner-history features for the CSEDM companion study."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from socratic_tutor.csedm_study.analysis_plan import PRIMARY_FEATURES
from socratic_tutor.csedm_study.source import CsvRow


class CSEDMAdapterError(ValueError):
    """CSEDM rows cannot be transformed under the frozen leakage rules."""


@dataclass(frozen=True, slots=True)
class LearnerEvent:
    """One event available to a prediction only when its order is earlier."""

    order: int
    learner_id: str
    problem_id: str
    event_type: str
    correct: bool


@dataclass(frozen=True, slots=True)
class PredictionTarget:
    """One first-attempt outcome and its prediction-time cut-off."""

    learner_id: str
    problem_id: str
    start_order: int
    first_correct: bool


@dataclass(frozen=True, slots=True)
class RawFeatures:
    """Six prespecified features before fold-specific imputation and scaling."""

    prior_event_count: float
    prior_submission_count: float
    prior_correctness_rate: float | None
    prior_hint_request_rate: float | None
    prior_distinct_problem_count: float
    training_learner_target_problem_success_rate: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in PRIMARY_FEATURES}


@dataclass(frozen=True, slots=True)
class FeatureTransform:
    """Training-fold values used to impute and standardise one feature."""

    median: float
    mean: float
    scale: float


def parse_events(rows: tuple[CsvRow, ...]) -> dict[str, tuple[LearnerEvent, ...]]:
    """Parse and order source events by learner."""

    grouped: defaultdict[str, list[LearnerEvent]] = defaultdict(list)
    seen_orders: set[int] = set()
    for row in rows:
        order = _positive_integer(row["Order"], "MainTable.Order")
        if order in seen_orders:
            raise CSEDMAdapterError(f"MainTable.Order is duplicated: {order}")
        seen_orders.add(order)
        event = LearnerEvent(
            order=order,
            learner_id=_required(row["SubjectID"], "MainTable.SubjectID"),
            problem_id=_required(row["ProblemID"], "MainTable.ProblemID"),
            event_type=_required(row["EventType"], "MainTable.EventType"),
            correct=_boolean(row["Correct"], "MainTable.Correct"),
        )
        grouped[event.learner_id].append(event)
    return {
        learner_id: tuple(sorted(events, key=lambda event: event.order))
        for learner_id, events in sorted(grouped.items())
    }


def parse_targets(rows: tuple[CsvRow, ...]) -> tuple[PredictionTarget, ...]:
    """Parse prediction targets without exposing post-target outcomes as features."""

    return tuple(
        PredictionTarget(
            learner_id=_required(row["SubjectID"], "Predict.SubjectID"),
            problem_id=_required(row["ProblemID"], "Predict.ProblemID"),
            start_order=_positive_integer(row["StartOrder"], "Predict.StartOrder"),
            first_correct=_boolean(row["FirstCorrect"], "Predict.FirstCorrect"),
        )
        for row in rows
    )


def build_fold_features(
    training_targets: tuple[PredictionTarget, ...],
    test_targets: tuple[PredictionTarget, ...],
    events_by_learner: dict[str, tuple[LearnerEvent, ...]],
) -> tuple[tuple[RawFeatures, ...], tuple[RawFeatures, ...]]:
    """Build raw features using training-only and pre-cut-off information."""

    problem_total = Counter(target.problem_id for target in training_targets)
    problem_success = Counter(
        target.problem_id for target in training_targets if target.first_correct
    )
    training = tuple(
        _raw_features(
            target,
            events_by_learner,
            problem_success_count=problem_success[target.problem_id] - int(target.first_correct),
            problem_target_count=problem_total[target.problem_id] - 1,
        )
        for target in training_targets
    )
    test = tuple(
        _raw_features(
            target,
            events_by_learner,
            problem_success_count=problem_success[target.problem_id],
            problem_target_count=problem_total[target.problem_id],
        )
        for target in test_targets
    )
    return training, test


def fit_feature_transforms(rows: tuple[RawFeatures, ...]) -> dict[str, FeatureTransform]:
    """Fit medians and standard scores from training-fold rows only."""

    if not rows:
        raise CSEDMAdapterError("A training fold cannot be empty")
    transforms: dict[str, FeatureTransform] = {}
    for name in PRIMARY_FEATURES:
        observed = [value for row in rows if (value := row.as_dict()[name]) is not None]
        if not observed:
            raise CSEDMAdapterError(f"Training fold has no observed values for {name}")
        median = float(statistics.median(observed))
        numeric: list[float] = []
        for row in rows:
            value = row.as_dict()[name]
            numeric.append(median if value is None else value)
        mean = math.fsum(numeric) / len(numeric)
        variance = math.fsum((value - mean) ** 2 for value in numeric) / len(numeric)
        scale = math.sqrt(variance)
        transforms[name] = FeatureTransform(
            median=median,
            mean=mean,
            scale=scale if scale > 0.0 else 1.0,
        )
    return transforms


def transform_features(
    row: RawFeatures,
    transforms: dict[str, FeatureTransform],
) -> tuple[dict[str, float], dict[str, bool]]:
    """Impute and standardise one row using frozen training-fold values."""

    values: dict[str, float] = {}
    missing: dict[str, bool] = {}
    raw = row.as_dict()
    for name in PRIMARY_FEATURES:
        transform = transforms[name]
        raw_value = raw[name]
        missing[name] = raw_value is None
        value = transform.median if raw_value is None else raw_value
        values[name] = (value - transform.mean) / transform.scale
    return values, missing


def _raw_features(
    target: PredictionTarget,
    events_by_learner: dict[str, tuple[LearnerEvent, ...]],
    *,
    problem_success_count: int,
    problem_target_count: int,
) -> RawFeatures:
    prior_events = tuple(
        event
        for event in events_by_learner.get(target.learner_id, ())
        if event.order < target.start_order
    )
    submissions = tuple(event for event in prior_events if event.event_type == "Submit")
    hints = sum(event.event_type == "X-HintRequest" for event in prior_events)
    return RawFeatures(
        prior_event_count=float(len(prior_events)),
        prior_submission_count=float(len(submissions)),
        prior_correctness_rate=(
            sum(event.correct for event in submissions) / len(submissions) if submissions else None
        ),
        prior_hint_request_rate=hints / len(prior_events) if prior_events else None,
        prior_distinct_problem_count=float(len({event.problem_id for event in prior_events})),
        training_learner_target_problem_success_rate=(
            problem_success_count / problem_target_count if problem_target_count > 0 else None
        ),
    )


def _required(value: str, field: str) -> str:
    if not value:
        raise CSEDMAdapterError(f"{field} cannot be empty")
    return value


def _positive_integer(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise CSEDMAdapterError(f"{field} must be an integer") from error
    if parsed <= 0:
        raise CSEDMAdapterError(f"{field} must be positive")
    return parsed


def _boolean(value: str, field: str) -> bool:
    normalised = value.upper()
    if normalised not in {"TRUE", "FALSE"}:
        raise CSEDMAdapterError(f"{field} must be TRUE or FALSE")
    return normalised == "TRUE"
