"""Statistical calculations for CSEDM uncertainty-based error review."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from socratic_tutor.csedm_study.features import CSEDMAdapterError


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One restricted out-of-fold prediction used in analysis."""

    model_id: str
    fold_id: int
    target_id: str
    learner_id: str
    first_correct: bool
    probability_correct: float

    @property
    def predicted_correct(self) -> bool:
        return self.probability_correct >= 0.5

    @property
    def is_error(self) -> bool:
        return self.predicted_correct != self.first_correct

    @property
    def uncertainty(self) -> float:
        return abs(self.probability_correct - 0.5)


@dataclass(frozen=True, slots=True)
class TriageSummary:
    """Observed and random-expected error capture at one review budget."""

    eligible_count: int
    selected_count: int
    error_count: int
    selected_error_count: int
    expected_random_selected_error_count: float
    captured_error_recall: float
    expected_random_captured_error_recall: float
    captured_error_recall_difference: float
    selected_target_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Descriptive prediction quality for one model."""

    prediction_count: int
    positive_count: int
    predicted_positive_count: int
    error_count: int
    accuracy: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """Error among the most confident predictions retained automatically."""

    requested_coverage: float
    retained_count: int
    actual_coverage: float
    retained_error_count: int
    retained_error_rate: float


def triage_summary(
    records: tuple[PredictionRecord, ...],
    budget_fraction: float,
) -> TriageSummary:
    """Compare uncertainty-ranked error capture with exact random expectation."""

    _validate_records(records)
    if not 0.0 < budget_fraction < 1.0:
        raise CSEDMAdapterError("Review budget fraction must be between zero and one")
    by_fold = _by_fold(records)
    selected: set[str] = set()
    expected_errors = 0.0
    for fold_records in by_fold.values():
        selected_count = math.floor(budget_fraction * len(fold_records))
        selected.update(
            row.target_id
            for row in sorted(fold_records, key=lambda row: (row.uncertainty, row.target_id))[
                :selected_count
            ]
        )
        fold_errors = sum(row.is_error for row in fold_records)
        expected_errors += selected_count * fold_errors / len(fold_records)

    error_count = sum(row.is_error for row in records)
    if error_count == 0:
        raise CSEDMAdapterError("Captured-error recall is undefined when no predictions fail")
    selected_errors = sum(row.is_error and row.target_id in selected for row in records)
    captured = selected_errors / error_count
    expected = expected_errors / error_count
    return TriageSummary(
        eligible_count=len(records),
        selected_count=len(selected),
        error_count=error_count,
        selected_error_count=selected_errors,
        expected_random_selected_error_count=expected_errors,
        captured_error_recall=captured,
        expected_random_captured_error_recall=expected,
        captured_error_recall_difference=captured - expected,
        selected_target_ids=frozenset(selected),
    )


def learner_bootstrap_differences(
    records: tuple[PredictionRecord, ...],
    selected_target_ids: frozenset[str],
    *,
    budget_fraction: float,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Resample learners while preserving the original selected predictions."""

    _validate_records(records)
    if repetitions < 1:
        raise CSEDMAdapterError("Bootstrap repetitions must be positive")
    fold_selection_probabilities = {
        fold_id: math.floor(budget_fraction * len(fold_records)) / len(fold_records)
        for fold_id, fold_records in _by_fold(records).items()
    }
    learners: defaultdict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        learners[record.learner_id].append(record)
    learner_values = tuple(
        (
            sum(row.is_error for row in rows),
            sum(row.is_error and row.target_id in selected_target_ids for row in rows),
            math.fsum(
                float(row.is_error) * fold_selection_probabilities[row.fold_id] for row in rows
            ),
        )
        for _, rows in sorted(learners.items())
    )
    values = np.asarray(learner_values, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(
        0,
        len(learner_values),
        size=(repetitions, len(learner_values)),
    )
    totals = np.sum(values[indices], axis=1)
    if np.any(totals[:, 0] == 0.0):
        raise CSEDMAdapterError("A learner bootstrap sample contained no prediction errors")
    return (totals[:, 1] - totals[:, 2]) / totals[:, 0]


def prediction_metrics(
    records: tuple[PredictionRecord, ...],
    *,
    probability_floor: float,
    calibration_bins: int,
) -> PredictionMetrics:
    """Calculate the four prespecified descriptive prediction metrics."""

    _validate_records(records)
    probabilities = np.asarray([row.probability_correct for row in records], dtype=np.float64)
    outcomes = np.asarray([row.first_correct for row in records], dtype=np.float64)
    clipped = np.clip(probabilities, probability_floor, 1.0 - probability_floor)
    log_loss = -float(
        np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
    )
    error_count = sum(row.is_error for row in records)
    return PredictionMetrics(
        prediction_count=len(records),
        positive_count=sum(row.first_correct for row in records),
        predicted_positive_count=sum(row.predicted_correct for row in records),
        error_count=error_count,
        accuracy=1.0 - error_count / len(records),
        brier_score=float(np.mean((probabilities - outcomes) ** 2)),
        log_loss=log_loss,
        expected_calibration_error=_calibration_error(
            probabilities,
            outcomes,
            bin_count=calibration_bins,
        ),
    )


def risk_coverage_curve(
    records: tuple[PredictionRecord, ...],
    fractions: tuple[float, ...],
) -> tuple[RiskCoveragePoint, ...]:
    """Retain the most confident predictions within each official fold."""

    _validate_records(records)
    by_fold = _by_fold(records)
    points: list[RiskCoveragePoint] = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise CSEDMAdapterError("Risk-coverage fractions must be in (0, 1]")
        retained: list[PredictionRecord] = []
        for fold_records in by_fold.values():
            count = max(1, math.floor(fraction * len(fold_records)))
            retained.extend(
                sorted(fold_records, key=lambda row: (-row.uncertainty, row.target_id))[:count]
            )
        errors = sum(row.is_error for row in retained)
        points.append(
            RiskCoveragePoint(
                requested_coverage=fraction,
                retained_count=len(retained),
                actual_coverage=len(retained) / len(records),
                retained_error_count=errors,
                retained_error_rate=errors / len(retained),
            )
        )
    return tuple(points)


def _by_fold(
    records: tuple[PredictionRecord, ...],
) -> dict[int, tuple[PredictionRecord, ...]]:
    grouped: defaultdict[int, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.fold_id].append(record)
    return {
        fold_id: tuple(sorted(rows, key=lambda row: row.target_id))
        for fold_id, rows in sorted(grouped.items())
    }


def _validate_records(records: tuple[PredictionRecord, ...]) -> None:
    if not records:
        raise CSEDMAdapterError("CSEDM analysis requires at least one prediction")
    if len({row.target_id for row in records}) != len(records):
        raise CSEDMAdapterError("CSEDM analysis predictions must have unique target IDs")
    if any(
        not math.isfinite(row.probability_correct) or not 0.0 <= row.probability_correct <= 1.0
        for row in records
    ):
        raise CSEDMAdapterError("CSEDM analysis probabilities must be finite values in [0, 1]")


def _calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bin_count: int,
) -> float:
    if bin_count < 1:
        raise CSEDMAdapterError("Calibration bin count must be positive")
    bin_indices = np.minimum((probabilities * bin_count).astype(np.int64), bin_count - 1)
    error = 0.0
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        count = int(np.count_nonzero(mask))
        if count:
            error += (
                count
                / len(probabilities)
                * abs(float(np.mean(probabilities[mask])) - float(np.mean(outcomes[mask])))
            )
    return error
