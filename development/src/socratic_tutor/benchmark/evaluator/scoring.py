"""Transparent per-sample losses and scoring-only benchmark baselines."""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.criterion import (
    CriterionExecutionStatus,
    CriterionMissingReason,
    CriterionRecord,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.public import (
    EXPECTED_CONDITIONS,
    BenchmarkCondition,
    BenchmarkSampleKey,
    ConditionCommitSet,
    DecisionPredictionRecord,
    ProbeExecutionSummary,
    evidence_from_test_counts,
)
from socratic_tutor.contracts import ContractModel, Evidence, TrackerState, TutorAction

type TrackerUpdate = Callable[[TrackerState, Evidence], TrackerState]


class ScoringError(ValueError):
    """Scoring inputs are incomplete, unmatched, or inconsistent with frozen rules."""


class CalibrationStatus(StrEnum):
    """Run-level interpretation of the current tracker score."""

    CALIBRATED = "calibrated"
    UNCALIBRATED_SCORE = "uncalibrated_score"


class PrimaryMetric(StrEnum):
    """The loss contrast selected before held-out criterion access."""

    PAIRED_BRIER_DIFFERENCE = "paired_brier_difference"
    PAIRED_CLASSIFICATION_ERROR_DIFFERENCE = "paired_classification_error_difference"


_PRIMARY_METRIC_BY_STATUS: dict[CalibrationStatus, PrimaryMetric] = {
    CalibrationStatus.CALIBRATED: PrimaryMetric.PAIRED_BRIER_DIFFERENCE,
    CalibrationStatus.UNCALIBRATED_SCORE: (PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE),
}


class CalibrationDecision(ContractModel):
    """Content-addressed pre-held-out decision selecting probability or score analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.calibration_decision.v1"] = "benchmark.calibration_decision.v1"
    status: CalibrationStatus
    primary_metric: PrimaryMetric
    policy_threshold: float = Field(ge=0.0, le=1.0)
    calibration_task_family_hash: Sha256
    calibration_data_hash: Sha256
    tracker_version: str = Field(min_length=1)
    diagnostics_hash: Sha256
    decision_owner: str = Field(min_length=1)
    decided_at_utc: datetime
    decision_hash: Sha256

    @model_validator(mode="after")
    def validate_decision(self) -> "CalibrationDecision":
        _require_utc(self.decided_at_utc, "Calibration decision time")
        if self.primary_metric is not _PRIMARY_METRIC_BY_STATUS[self.status]:
            raise ValueError("Calibration status and primary metric do not match")
        if self.decision_hash != calibration_decision_hash(self):
            raise ValueError("Calibration decision hash does not match its content")
        return self


def calibration_decision_hash(decision: CalibrationDecision) -> Sha256:
    """Hash all calibration decision fields except the self digest."""

    return model_content_hash(decision, exclude={"decision_hash"})


def create_calibration_decision(
    *,
    status: CalibrationStatus,
    policy_threshold: float,
    calibration_task_family_hash: Sha256,
    calibration_data_hash: Sha256,
    tracker_version: str,
    diagnostics_hash: Sha256,
    decision_owner: str,
    decided_at_utc: datetime,
) -> CalibrationDecision:
    """Create a calibration decision while deriving the only valid primary metric."""

    content = {
        "schema_version": 1,
        "schema_id": "benchmark.calibration_decision.v1",
        "status": status,
        "primary_metric": _PRIMARY_METRIC_BY_STATUS[status],
        "policy_threshold": policy_threshold,
        "calibration_task_family_hash": calibration_task_family_hash,
        "calibration_data_hash": calibration_data_hash,
        "tracker_version": tracker_version,
        "diagnostics_hash": diagnostics_hash,
        "decision_owner": decision_owner,
        "decided_at_utc": decided_at_utc,
    }
    draft = CalibrationDecision.model_construct(
        _fields_set=set(content),
        **content,
        decision_hash="0" * 64,
    )
    return CalibrationDecision.model_validate(
        {**content, "decision_hash": calibration_decision_hash(draft)}
    )


class BaselineFitRole(StrEnum):
    """Only non-held-out roles allowed to define scoring baselines."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"


class ConstantPrevalenceFit(ContractModel):
    """Calibration/development-only fit for the constant scoring baseline."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.constant_prevalence_fit.v1"] = (
        "benchmark.constant_prevalence_fit.v1"
    )
    fit_role: BaselineFitRole
    fit_split_hash: Sha256
    baseline_version: str = Field(min_length=1)
    positive_count: int = Field(ge=0)
    sample_count: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    fitted_at_utc: datetime
    fit_hash: Sha256

    @model_validator(mode="after")
    def validate_fit(self) -> "ConstantPrevalenceFit":
        _require_utc(self.fitted_at_utc, "Constant-baseline fit time")
        if self.positive_count > self.sample_count:
            raise ValueError("Positive count cannot exceed fitted sample count")
        if self.score != self.positive_count / self.sample_count:
            raise ValueError("Constant score must equal fitted criterion prevalence")
        if self.fit_hash != constant_prevalence_fit_hash(self):
            raise ValueError("Constant-baseline fit hash does not match its content")
        return self


def constant_prevalence_fit_hash(fit: ConstantPrevalenceFit) -> Sha256:
    """Hash all fitted baseline fields except the self digest."""

    return model_content_hash(fit, exclude={"fit_hash"})


def fit_constant_prevalence(
    outcomes: Sequence[bool],
    *,
    fit_role: BaselineFitRole,
    fit_split_hash: Sha256,
    baseline_version: str,
    fitted_at_utc: datetime,
) -> ConstantPrevalenceFit:
    """Fit criterion prevalence using only caller-declared calibration/development data."""

    frozen_outcomes = tuple(outcomes)
    if not frozen_outcomes:
        raise ScoringError("Constant prevalence requires at least one fit outcome")
    positive_count = sum(frozen_outcomes)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.constant_prevalence_fit.v1",
        "fit_role": fit_role,
        "fit_split_hash": fit_split_hash,
        "baseline_version": baseline_version,
        "positive_count": positive_count,
        "sample_count": len(frozen_outcomes),
        "score": positive_count / len(frozen_outcomes),
        "fitted_at_utc": fitted_at_utc,
    }
    draft = ConstantPrevalenceFit.model_construct(
        _fields_set=set(content),
        **content,
        fit_hash="0" * 64,
    )
    return ConstantPrevalenceFit.model_validate(
        {**content, "fit_hash": constant_prevalence_fit_hash(draft)}
    )


class ScoringBaseline(StrEnum):
    """Baselines evaluated after commitments without becoming tracker conditions."""

    CONSTANT_PREVALENCE = "constant_prevalence"
    PROBE_ONLY = "probe_only"


EXPECTED_BASELINES = tuple(ScoringBaseline)


class BaselineResult(ContractModel):
    """One scoring-only prediction bound to frozen input and calibration semantics."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.baseline_result.v1"] = "benchmark.baseline_result.v1"
    key: BenchmarkSampleKey
    baseline: ScoringBaseline
    fit_role: BaselineFitRole
    fit_split_hash: Sha256
    baseline_version: str = Field(min_length=1)
    input_hash: Sha256
    score: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    binary_decision: bool
    calibration_decision_hash: Sha256
    created_at_utc: datetime
    record_hash: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> "BaselineResult":
        _require_utc(self.created_at_utc, "Baseline creation time")
        if self.binary_decision is not (self.score >= self.policy_threshold):
            raise ValueError("Baseline decision does not match the frozen threshold")
        if self.record_hash != baseline_result_hash(self):
            raise ValueError("Baseline result hash does not match its content")
        return self


def baseline_result_hash(result: BaselineResult) -> Sha256:
    """Hash every baseline result field except its self digest."""

    return model_content_hash(result, exclude={"record_hash"})


def create_constant_baseline_result(
    key: BenchmarkSampleKey,
    *,
    fit: ConstantPrevalenceFit,
    calibration: CalibrationDecision,
    created_at_utc: datetime,
) -> BaselineResult:
    """Apply one frozen prevalence fit to a held-out sample without reading its label."""

    if fit.fitted_at_utc > calibration.decided_at_utc:
        raise ScoringError("Constant prevalence fit must precede the calibration decision")
    return _create_baseline_result(
        key=key,
        baseline=ScoringBaseline.CONSTANT_PREVALENCE,
        fit_role=fit.fit_role,
        fit_split_hash=fit.fit_split_hash,
        baseline_version=fit.baseline_version,
        input_hash=fit.fit_hash,
        score=fit.score,
        calibration=calibration,
        created_at_utc=created_at_utc,
    )


def create_probe_only_baseline_result(
    key: BenchmarkSampleKey,
    *,
    fit_role: BaselineFitRole,
    fit_split_hash: Sha256,
    baseline_version: str,
    initial_tracker_state: TrackerState,
    probe_evidence: ProbeExecutionSummary,
    tracker_update: TrackerUpdate,
    calibration: CalibrationDecision,
    created_at_utc: datetime,
) -> BaselineResult:
    """Score the valid probe from the initial prior with no public-dialogue input."""

    if probe_evidence.case_id != key.case_id:
        raise ScoringError("Probe-only evidence belongs to another benchmark case")
    if initial_tracker_state.concept != probe_evidence.target_concept:
        raise ScoringError("Probe-only prior targets another concept")
    evidence = evidence_from_test_counts(
        passed=probe_evidence.passed,
        failed=probe_evidence.failed,
        source="probe-only baseline",
    )
    tracker_after = tracker_update(initial_tracker_state, evidence)
    if tracker_after.concept != initial_tracker_state.concept:
        raise ScoringError("Probe-only tracker update changed the target concept")
    input_hash = canonical_sha256(
        {
            "schema_id": "benchmark.probe_only_input.v1",
            "key": key,
            "initial_tracker_state": initial_tracker_state,
            "probe_evidence": probe_evidence,
            "derived_evidence": evidence,
            "tracker_after": tracker_after,
            "baseline_version": baseline_version,
            "fit_split_hash": fit_split_hash,
            "fit_role": fit_role,
        }
    )
    return _create_baseline_result(
        key=key,
        baseline=ScoringBaseline.PROBE_ONLY,
        fit_role=fit_role,
        fit_split_hash=fit_split_hash,
        baseline_version=baseline_version,
        input_hash=input_hash,
        score=tracker_after.mastery_probability,
        calibration=calibration,
        created_at_utc=created_at_utc,
    )


def _create_baseline_result(
    *,
    key: BenchmarkSampleKey,
    baseline: ScoringBaseline,
    fit_role: BaselineFitRole,
    fit_split_hash: Sha256,
    baseline_version: str,
    input_hash: Sha256,
    score: float,
    calibration: CalibrationDecision,
    created_at_utc: datetime,
) -> BaselineResult:
    if created_at_utc < calibration.decided_at_utc:
        raise ScoringError("Baseline result cannot precede the calibration decision")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.baseline_result.v1",
        "key": key,
        "baseline": baseline,
        "fit_role": fit_role,
        "fit_split_hash": fit_split_hash,
        "baseline_version": baseline_version,
        "input_hash": input_hash,
        "score": score,
        "policy_threshold": calibration.policy_threshold,
        "binary_decision": score >= calibration.policy_threshold,
        "calibration_decision_hash": calibration.decision_hash,
        "created_at_utc": created_at_utc,
    }
    draft = BaselineResult.model_construct(
        _fields_set=set(content),
        **content,
        record_hash="0" * 64,
    )
    return BaselineResult.model_validate({**content, "record_hash": baseline_result_hash(draft)})


class RepeatEstimator(StrEnum):
    """Four committed conditions followed by two scoring-only baselines."""

    DIALOGUE_ONLY = "dialogue_only"
    PROBE_INFORMED = "probe_informed"
    UNRELATED_PROBE = "unrelated_probe"
    CORRUPTED_PROBE = "corrupted_probe"
    CONSTANT_PREVALENCE = "constant_prevalence"
    PROBE_ONLY = "probe_only"


EXPECTED_ESTIMATORS = tuple(RepeatEstimator)


class RepeatLossName(StrEnum):
    """Supported per-sample binary losses."""

    BRIER = "brier"
    CLASSIFICATION_ERROR = "classification_error"


class ControlRole(StrEnum):
    """Prespecified role of each estimator in the paired analysis."""

    REFERENCE = "reference"
    VALID_EVIDENCE = "valid_evidence"
    UNRELATED_CONTROL = "unrelated_control"
    CORRUPTED_CONTROL = "corrupted_control"
    CONSTANT_BASELINE = "constant_baseline"
    PROBE_ONLY_BASELINE = "probe_only_baseline"


_ESTIMATOR_BY_CONDITION: dict[BenchmarkCondition, RepeatEstimator] = {
    BenchmarkCondition.DIALOGUE_ONLY: RepeatEstimator.DIALOGUE_ONLY,
    BenchmarkCondition.PROBE_INFORMED: RepeatEstimator.PROBE_INFORMED,
    BenchmarkCondition.UNRELATED_PROBE: RepeatEstimator.UNRELATED_PROBE,
    BenchmarkCondition.CORRUPTED_PROBE: RepeatEstimator.CORRUPTED_PROBE,
}
_ESTIMATOR_BY_BASELINE: dict[ScoringBaseline, RepeatEstimator] = {
    ScoringBaseline.CONSTANT_PREVALENCE: RepeatEstimator.CONSTANT_PREVALENCE,
    ScoringBaseline.PROBE_ONLY: RepeatEstimator.PROBE_ONLY,
}
_CONTROL_ROLE: dict[RepeatEstimator, ControlRole] = {
    RepeatEstimator.DIALOGUE_ONLY: ControlRole.REFERENCE,
    RepeatEstimator.PROBE_INFORMED: ControlRole.VALID_EVIDENCE,
    RepeatEstimator.UNRELATED_PROBE: ControlRole.UNRELATED_CONTROL,
    RepeatEstimator.CORRUPTED_PROBE: ControlRole.CORRUPTED_CONTROL,
    RepeatEstimator.CONSTANT_PREVALENCE: ControlRole.CONSTANT_BASELINE,
    RepeatEstimator.PROBE_ONLY: ControlRole.PROBE_ONLY_BASELINE,
}


class RepeatMetric(ContractModel):
    """One estimator scored against one sample's common held-out outcome."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.repeat_metric.v1"] = "benchmark.repeat_metric.v1"
    key: BenchmarkSampleKey
    estimator: RepeatEstimator
    estimator_version: str = Field(min_length=1)
    source_record_hash: Sha256
    criterion_record_hash: Sha256
    criterion_demonstrated_performance: bool | None
    score: float = Field(ge=0.0, le=1.0)
    binary_decision: bool
    directive: TutorAction | None
    selected_loss_name: RepeatLossName
    loss_value: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_loss: float | None = Field(default=None, ge=0.0, le=1.0)
    classification_error: float | None = Field(default=None, ge=0.0, le=1.0)
    unsafe_advancement: bool | None
    false_confidence_acceptance: bool | None
    false_confidence_detection: bool | None
    action_disagreement_with_dialogue: bool | None
    control_role: ControlRole
    evidence_source_hash: Sha256 | None
    eligible: bool
    missing_reason: CriterionMissingReason | None
    latency_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    retries: int | None = Field(default=None, ge=0)
    sandbox_time_ms: int | None = Field(default=None, ge=0)
    execution_failure: bool
    cost_usd_micros: int | None = Field(default=None, ge=0)
    calibration_decision_hash: Sha256
    scorer_version: str = Field(min_length=1)
    created_at_utc: datetime
    record_hash: Sha256

    @model_validator(mode="after")
    def validate_metric(self) -> "RepeatMetric":
        _require_utc(self.created_at_utc, "Repeat-metric creation time")
        if self.eligible:
            if self.criterion_demonstrated_performance is None or self.missing_reason is not None:
                raise ValueError("Eligible metric requires a criterion outcome")
            if (
                self.loss_value is None
                or self.brier_loss is None
                or self.classification_error is None
            ):
                raise ValueError("Eligible metric requires all prespecified losses")
            expected_selected = (
                self.brier_loss
                if self.selected_loss_name is RepeatLossName.BRIER
                else self.classification_error
            )
            if self.loss_value != expected_selected:
                raise ValueError("Selected loss value does not match selected loss name")
            if self.false_confidence_acceptance is None or self.false_confidence_detection is None:
                raise ValueError("Eligible metric requires safety outcomes")
            if (self.directive is None) is not (self.unsafe_advancement is None):
                raise ValueError("Unsafe advancement is defined only for policy directives")
        elif (
            any(
                value is not None
                for value in (
                    self.criterion_demonstrated_performance,
                    self.loss_value,
                    self.brier_loss,
                    self.classification_error,
                    self.unsafe_advancement,
                    self.false_confidence_acceptance,
                    self.false_confidence_detection,
                )
            )
            or self.missing_reason is None
        ):
            raise ValueError(
                "Ineligible metric must preserve missingness without invented outcomes"
            )
        if self.control_role is not _CONTROL_ROLE[self.estimator]:
            raise ValueError("Estimator has the wrong control role")
        if self.record_hash != repeat_metric_hash(self):
            raise ValueError("Repeat metric hash does not match its content")
        return self


def repeat_metric_hash(metric: RepeatMetric) -> Sha256:
    """Hash every repeat-level metric field except its self digest."""

    return model_content_hash(metric, exclude={"record_hash"})


class RepeatComparisonMetrics(ContractModel):
    """Transient hand-checkable paired effects; case aggregation remains M3.15."""

    primary_valid_effect: float | None
    unrelated_control_effect: float | None
    corrupted_control_effect: float | None
    dialogue_vs_constant_effect: float | None
    dialogue_vs_probe_only_effect: float | None


class BenchmarkScorer:
    """Score one exact committed set against one exact criterion record."""

    def __init__(self, calibration: CalibrationDecision, *, scorer_version: str) -> None:
        if not scorer_version.strip():
            raise ScoringError("Scorer version must be non-empty")
        self._calibration = calibration
        self._scorer_version = scorer_version

    def score(
        self,
        *,
        commit_set: ConditionCommitSet,
        predictions: tuple[DecisionPredictionRecord, ...],
        criterion: CriterionRecord,
        baselines: tuple[BaselineResult, ...],
        created_at_utc: datetime,
    ) -> tuple[RepeatMetric, ...]:
        """Return four committed and two baseline rows with one selected loss each."""

        _validate_scoring_inputs(
            commit_set=commit_set,
            predictions=predictions,
            criterion=criterion,
            baselines=baselines,
            calibration=self._calibration,
            created_at_utc=created_at_utc,
        )
        dialogue_action = predictions[0].decision.action
        rows = tuple(
            self._score_prediction(
                prediction,
                criterion=criterion,
                dialogue_action=dialogue_action,
                created_at_utc=created_at_utc,
            )
            for prediction in predictions
        ) + tuple(
            self._score_baseline(
                baseline,
                criterion=criterion,
                created_at_utc=created_at_utc,
            )
            for baseline in baselines
        )
        if tuple(row.estimator for row in rows) != EXPECTED_ESTIMATORS:
            raise ScoringError("Scorer did not produce the frozen estimator order")
        return rows

    def _score_prediction(
        self,
        prediction: DecisionPredictionRecord,
        *,
        criterion: CriterionRecord,
        dialogue_action: TutorAction,
        created_at_utc: datetime,
    ) -> RepeatMetric:
        estimator = _ESTIMATOR_BY_CONDITION[prediction.condition]
        return self._metric(
            key=criterion.key,
            estimator=estimator,
            estimator_version=f"{prediction.tracker_version}+{prediction.policy_version}",
            source_record_hash=prediction.record_hash,
            criterion=criterion,
            score=prediction.tracker_mastery_probability,
            directive=prediction.decision.action,
            action_disagreement=prediction.decision.action != dialogue_action,
            evidence_source_hash=prediction.additional_evidence_hash,
            created_at_utc=created_at_utc,
        )

    def _score_baseline(
        self,
        baseline: BaselineResult,
        *,
        criterion: CriterionRecord,
        created_at_utc: datetime,
    ) -> RepeatMetric:
        return self._metric(
            key=criterion.key,
            estimator=_ESTIMATOR_BY_BASELINE[baseline.baseline],
            estimator_version=baseline.baseline_version,
            source_record_hash=baseline.record_hash,
            criterion=criterion,
            score=baseline.score,
            directive=None,
            action_disagreement=None,
            evidence_source_hash=baseline.input_hash,
            created_at_utc=created_at_utc,
        )

    def _metric(
        self,
        *,
        key: BenchmarkSampleKey,
        estimator: RepeatEstimator,
        estimator_version: str,
        source_record_hash: Sha256,
        criterion: CriterionRecord,
        score: float,
        directive: TutorAction | None,
        action_disagreement: bool | None,
        evidence_source_hash: Sha256 | None,
        created_at_utc: datetime,
    ) -> RepeatMetric:
        threshold = self._calibration.policy_threshold
        binary_decision = score >= threshold
        outcome = criterion.demonstrated_performance
        eligible = outcome is not None
        brier = brier_loss(score, outcome) if outcome is not None else None
        classification = (
            classification_error(score, outcome, threshold=threshold)
            if outcome is not None
            else None
        )
        selected_name = (
            RepeatLossName.BRIER
            if self._calibration.primary_metric is PrimaryMetric.PAIRED_BRIER_DIFFERENCE
            else RepeatLossName.CLASSIFICATION_ERROR
        )
        selected_value = brier if selected_name is RepeatLossName.BRIER else classification
        false_acceptance = (not outcome and binary_decision) if outcome is not None else None
        false_detection = (not outcome and not binary_decision) if outcome is not None else None
        unsafe_advancement = (
            (not outcome and directive == TutorAction.TRANSFER)
            if outcome is not None and directive is not None
            else None
        )
        content = {
            "schema_version": 1,
            "schema_id": "benchmark.repeat_metric.v1",
            "key": key,
            "estimator": estimator,
            "estimator_version": estimator_version,
            "source_record_hash": source_record_hash,
            "criterion_record_hash": criterion.record_hash,
            "criterion_demonstrated_performance": outcome,
            "score": score,
            "binary_decision": binary_decision,
            "directive": directive,
            "selected_loss_name": selected_name,
            "loss_value": selected_value,
            "brier_loss": brier,
            "classification_error": classification,
            "unsafe_advancement": unsafe_advancement,
            "false_confidence_acceptance": false_acceptance,
            "false_confidence_detection": false_detection,
            "action_disagreement_with_dialogue": action_disagreement,
            "control_role": _CONTROL_ROLE[estimator],
            "evidence_source_hash": evidence_source_hash,
            "eligible": eligible,
            "missing_reason": criterion.missing_reason,
            "latency_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "retries": None,
            "sandbox_time_ms": None,
            "execution_failure": (
                criterion.execution.status is not CriterionExecutionStatus.COMPLETED
            ),
            "cost_usd_micros": None,
            "calibration_decision_hash": self._calibration.decision_hash,
            "scorer_version": self._scorer_version,
            "created_at_utc": created_at_utc,
        }
        draft = RepeatMetric.model_construct(
            _fields_set=set(content),
            **content,
            record_hash="0" * 64,
        )
        return RepeatMetric.model_validate({**content, "record_hash": repeat_metric_hash(draft)})


def brier_loss(score: float, outcome: bool) -> float:
    """Return squared error for one bounded score and binary outcome."""

    if not 0.0 <= score <= 1.0:
        raise ScoringError("Brier score input must be within [0, 1]")
    return (score - float(outcome)) ** 2


def classification_error(score: float, outcome: bool, *, threshold: float) -> float:
    """Return zero-one error under the frozen simple-policy threshold."""

    if not 0.0 <= score <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ScoringError("Classification score and threshold must be within [0, 1]")
    return float((score >= threshold) is not outcome)


def repeat_comparisons(metrics: tuple[RepeatMetric, ...]) -> RepeatComparisonMetrics:
    """Compute hand-checkable repeat effects in the prespecified favourable direction."""

    if tuple(metric.estimator for metric in metrics) != EXPECTED_ESTIMATORS:
        raise ScoringError("Repeat comparisons require the exact estimator set and order")
    if any(metric.key != metrics[0].key for metric in metrics[1:]):
        raise ScoringError("Repeat comparisons require one shared sample identity")
    by_estimator = {metric.estimator: metric for metric in metrics}
    dialogue = by_estimator[RepeatEstimator.DIALOGUE_ONLY]
    return RepeatComparisonMetrics(
        primary_valid_effect=_loss_difference(
            dialogue,
            by_estimator[RepeatEstimator.PROBE_INFORMED],
        ),
        unrelated_control_effect=_loss_difference(
            dialogue,
            by_estimator[RepeatEstimator.UNRELATED_PROBE],
        ),
        corrupted_control_effect=_loss_difference(
            dialogue,
            by_estimator[RepeatEstimator.CORRUPTED_PROBE],
        ),
        dialogue_vs_constant_effect=_loss_difference(
            dialogue,
            by_estimator[RepeatEstimator.CONSTANT_PREVALENCE],
        ),
        dialogue_vs_probe_only_effect=_loss_difference(
            dialogue,
            by_estimator[RepeatEstimator.PROBE_ONLY],
        ),
    )


def _loss_difference(left: RepeatMetric, right: RepeatMetric) -> float | None:
    if left.loss_value is None or right.loss_value is None:
        return None
    return left.loss_value - right.loss_value


def _validate_scoring_inputs(
    *,
    commit_set: ConditionCommitSet,
    predictions: tuple[DecisionPredictionRecord, ...],
    criterion: CriterionRecord,
    baselines: tuple[BaselineResult, ...],
    calibration: CalibrationDecision,
    created_at_utc: datetime,
) -> None:
    _require_utc(created_at_utc, "Scoring time")
    if created_at_utc < criterion.revealed_at_utc:
        raise ScoringError("Scoring cannot precede criterion reveal")
    if criterion.key != commit_set.key or criterion.condition_set_hash != commit_set.seal_hash:
        raise ScoringError("Criterion record does not match the committed condition set")
    if tuple(record.condition for record in predictions) != EXPECTED_CONDITIONS:
        raise ScoringError("Scoring requires the exact ordered prediction set")
    if tuple(record.record_hash for record in predictions) != commit_set.prediction_hashes:
        raise ScoringError("Prediction hashes do not match the committed condition set")
    if any(BenchmarkSampleKey.from_record(record) != commit_set.key for record in predictions):
        raise ScoringError("Prediction identity does not match the committed condition set")
    if calibration.tracker_version != commit_set.tracker_version:
        raise ScoringError("Calibration decision belongs to another tracker version")
    if calibration.decided_at_utc >= min(record.committed_at_utc for record in predictions):
        raise ScoringError("Calibration decision must precede held-out predictions")
    if tuple(result.baseline for result in baselines) != EXPECTED_BASELINES:
        raise ScoringError("Scoring requires the exact ordered scoring-only baselines")
    if any(result.key != commit_set.key for result in baselines):
        raise ScoringError("Baseline identity does not match the committed sample")
    if any(
        result.calibration_decision_hash != calibration.decision_hash
        or result.policy_threshold != calibration.policy_threshold
        for result in baselines
    ):
        raise ScoringError("Baseline does not use the frozen calibration decision")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
