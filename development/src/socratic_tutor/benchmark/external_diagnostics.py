"""Safety, missingness, and stability diagnostics for one sealed external run."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.criterion import CriterionExecutionStatus
from socratic_tutor.benchmark.evaluator.datasets import (
    read_criterion_records,
    read_repeat_metrics,
)
from socratic_tutor.benchmark.evaluator.scoring import RepeatEstimator
from socratic_tutor.benchmark.external_scoring import ExternalScoringSummary
from socratic_tutor.benchmark.external_seal import ExternalDecisionSealReport
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.contracts import ContractModel

_POLICY_ESTIMATORS = (
    RepeatEstimator.DIALOGUE_ONLY,
    RepeatEstimator.PROBE_INFORMED,
    RepeatEstimator.UNRELATED_PROBE,
    RepeatEstimator.CORRUPTED_PROBE,
)
_SANDBOX_FAILURES = (
    CriterionExecutionStatus.SANDBOX_UNAVAILABLE,
    CriterionExecutionStatus.SANDBOX_TIMEOUT,
    CriterionExecutionStatus.SANDBOX_RESOURCE_LIMIT,
    CriterionExecutionStatus.SANDBOX_ERROR,
)
type PolicyEstimatorName = Literal[
    "dialogue_only",
    "probe_informed",
    "unrelated_probe",
    "corrupted_probe",
]


class ExternalDiagnosticsError(ValueError):
    """Diagnostic inputs do not describe the same sealed and scored run."""


class CountRate(ContractModel):
    """One count with its explicit denominator and interpretation."""

    count: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)
    status: Literal["measured", "not_estimable_zero_denominator"]
    denominator_scope: Literal[
        "eligible_criterion_negative_cases",
        "all_committed_cases",
        "all_planned_cases",
    ]

    @model_validator(mode="after")
    def validate_rate(self) -> CountRate:
        if self.count > self.denominator:
            raise ValueError("Diagnostic count exceeds its denominator")
        if self.denominator == 0:
            if self.count != 0 or self.rate is not None:
                raise ValueError("Zero-denominator diagnostic cannot contain a rate")
            if self.status != "not_estimable_zero_denominator":
                raise ValueError("Zero-denominator diagnostic must be marked not estimable")
            return self
        if self.rate is None or abs(self.rate - self.count / self.denominator) > 1e-12:
            raise ValueError("Diagnostic rate does not match its count and denominator")
        if self.status != "measured":
            raise ValueError("Positive-denominator diagnostic must be marked measured")
        return self


class EstimatorSafetyDiagnostics(ContractModel):
    """Outcome and action diagnostics for one committed policy condition."""

    estimator: PolicyEstimatorName
    planned_case_count: Literal[24]
    eligible_case_count: int = Field(ge=1, le=24)
    missing_case_count: int = Field(ge=0, le=24)
    criterion_negative_case_count: int = Field(ge=0, le=24)
    false_acceptance: CountRate
    unsafe_advancement: CountRate
    action_disagreement_with_dialogue: CountRate

    @model_validator(mode="after")
    def validate_diagnostics(self) -> EstimatorSafetyDiagnostics:
        if self.planned_case_count != self.eligible_case_count + self.missing_case_count:
            raise ValueError("Estimator diagnostic case counts do not reconcile")
        if any(
            metric.denominator != self.criterion_negative_case_count
            for metric in (self.false_acceptance, self.unsafe_advancement)
        ):
            raise ValueError("Safety outcomes use the wrong negative-case denominator")
        if self.action_disagreement_with_dialogue.denominator != self.planned_case_count:
            raise ValueError("Action disagreement uses the wrong committed-case denominator")
        return self


class RunFailureDiagnostics(ContractModel):
    """Decision and criterion failures kept separate by stage."""

    planned_case_count: Literal[24]
    decision_complete_count: int = Field(ge=0, le=24)
    decision_precriterion_missing_count: int = Field(ge=0, le=24)
    decision_invalid_count: int = Field(ge=0, le=24)
    criterion_complete_count: int = Field(ge=0, le=24)
    criterion_provider_failure_count: int = Field(ge=0, le=24)
    criterion_sandbox_failure_count: int = Field(ge=0, le=24)
    criterion_sandbox_unavailable_count: int = Field(ge=0, le=24)
    criterion_sandbox_timeout_count: int = Field(ge=0, le=24)
    criterion_sandbox_resource_limit_count: int = Field(ge=0, le=24)
    criterion_sandbox_error_count: int = Field(ge=0, le=24)
    decision_invalid_rate: CountRate
    criterion_provider_failure_rate: CountRate
    criterion_sandbox_failure_rate: CountRate

    @model_validator(mode="after")
    def validate_failures(self) -> RunFailureDiagnostics:
        if self.planned_case_count != (
            self.decision_complete_count
            + self.decision_precriterion_missing_count
            + self.decision_invalid_count
        ):
            raise ValueError("Decision-stage failure counts do not reconcile")
        if self.planned_case_count != (
            self.criterion_complete_count
            + self.criterion_provider_failure_count
            + self.criterion_sandbox_failure_count
        ):
            raise ValueError("Criterion-stage failure counts do not reconcile")
        sandbox_sum = sum(
            (
                self.criterion_sandbox_unavailable_count,
                self.criterion_sandbox_timeout_count,
                self.criterion_sandbox_resource_limit_count,
                self.criterion_sandbox_error_count,
            )
        )
        if sandbox_sum != self.criterion_sandbox_failure_count:
            raise ValueError("Criterion sandbox failure subtypes do not reconcile")
        expected = (
            (self.decision_invalid_rate, self.decision_invalid_count),
            (self.criterion_provider_failure_rate, self.criterion_provider_failure_count),
            (self.criterion_sandbox_failure_rate, self.criterion_sandbox_failure_count),
        )
        if any(
            metric.denominator != self.planned_case_count or metric.count != count
            for metric, count in expected
        ):
            raise ValueError("Run-level failure rate uses the wrong count or denominator")
        return self


class ExternalDiagnosticsPlan(ContractModel):
    """Source and software identities for the read-only diagnostic analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_diagnostics_plan.v1"] = (
        "benchmark.external_diagnostics_plan.v1"
    )
    run_id: str = Field(min_length=1)
    scoring_summary_hash: Sha256
    primary_analysis_report_hash: Sha256
    decision_seal_report_hash: Sha256
    criterion_publication_hash: Sha256
    repeat_publication_hash: Sha256
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> ExternalDiagnosticsPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("External-diagnostics plan hash does not match its content")
        return self


class ExternalDiagnosticsReport(ContractModel):
    """Hand-checkable diagnostic tables with explicit non-estimable quantities."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_diagnostics_report.v1"] = (
        "benchmark.external_diagnostics_report.v1"
    )
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    estimator_diagnostics: tuple[EstimatorSafetyDiagnostics, ...] = Field(
        min_length=4, max_length=4
    )
    run_failures: RunFailureDiagnostics
    repeat_variability: Literal["not_estimable_one_model_run_per_case"] = (
        "not_estimable_one_model_run_per_case"
    )
    precision_recall_status: Literal["omitted_sparse_single_run"] = "omitted_sparse_single_run"
    precision_recall_reason: Literal[
        "class_counts_are_small_and_no_repeat_variability_is_available"
    ] = "class_counts_are_small_and_no_repeat_variability_is_available"
    diagnostic_scope: Literal["fixed_authored_corpus_only"] = "fixed_authored_corpus_only"
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> ExternalDiagnosticsReport:
        _require_utc(self.completed_at_utc)
        if tuple(record.estimator for record in self.estimator_diagnostics) != tuple(
            estimator.value for estimator in _POLICY_ESTIMATORS
        ):
            raise ValueError("Estimator diagnostics do not use the canonical condition order")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("External-diagnostics report hash does not match its content")
        return self


def run_external_diagnostics(
    *,
    scoring_summary_path: Path,
    primary_report_path: Path,
    decision_seal_report_path: Path,
    dataset_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ExternalDiagnosticsReport:
    """Summarize safety and failures without rerunning or rescoring any case."""

    _require_utc(created_at_utc)
    scoring = _load_json(scoring_summary_path, ExternalScoringSummary)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    seal = _load_json(decision_seal_report_path, ExternalDecisionSealReport)
    if len({scoring.run_id, primary.run_id, seal.run_id}) != 1:
        raise ExternalDiagnosticsError("Diagnostic sources belong to different runs")
    if created_at_utc <= primary.completed_at_utc:
        raise ExternalDiagnosticsError("Diagnostics must follow the primary analysis")
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ExternalDiagnosticsError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    store = AtomicParquetDatasetStore(dataset_root)
    criteria = read_criterion_records(store)
    repeats = read_repeat_metrics(store)
    if repeats.manifest.publication_hash != scoring.repeat_publication_hash:
        raise ExternalDiagnosticsError("Repeat metrics differ from the scoring summary")
    if any(
        identity != (scoring.benchmark_version, scoring.run_id)
        for identity in (
            (criteria.manifest.benchmark_version, criteria.manifest.run_id),
            (repeats.manifest.benchmark_version, repeats.manifest.run_id),
        )
    ):
        raise ExternalDiagnosticsError("Diagnostic datasets belong to another run")
    criterion_rows = criteria.table.to_pylist()
    repeat_rows = repeats.table.to_pylist()
    _validate_dataset_counts(scoring, primary, criterion_rows, repeat_rows)

    plan = _create_plan(
        scoring=scoring,
        primary=primary,
        seal=seal,
        criterion_publication_hash=criteria.manifest.publication_hash,
        repeat_publication_hash=repeats.manifest.publication_hash,
        lock_hash=lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "external_diagnostics_report.json"
    if report_path.exists():
        report = _load_json(report_path, ExternalDiagnosticsReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise ExternalDiagnosticsError("Existing diagnostic report belongs to another plan")
        return report
    write_immutable_json(root / "external_diagnostics_plan.json", plan)

    estimator_diagnostics = tuple(
        _estimator_diagnostics(estimator, repeat_rows) for estimator in _POLICY_ESTIMATORS
    )
    failures = _run_failures(seal, criterion_rows)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_diagnostics_report.v1",
        "run_id": scoring.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "estimator_diagnostics": estimator_diagnostics,
        "run_failures": failures,
        "repeat_variability": "not_estimable_one_model_run_per_case",
        "precision_recall_status": "omitted_sparse_single_run",
        "precision_recall_reason": (
            "class_counts_are_small_and_no_repeat_variability_is_available"
        ),
        "diagnostic_scope": "fixed_authored_corpus_only",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = ExternalDiagnosticsReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalDiagnosticsReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _validate_dataset_counts(
    scoring: ExternalScoringSummary,
    primary: PrimaryAnalysisReport,
    criterion_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
) -> None:
    criterion_case_ids = {str(row["case_id"]) for row in criterion_rows}
    missing_case_ids = {
        str(row["case_id"]) for row in criterion_rows if row["demonstrated_performance"] is None
    }
    eligible_repeat_count = sum(row["eligible"] is True for row in repeat_rows)
    expected = (
        len(criterion_rows) == scoring.criterion_count,
        len(criterion_case_ids) == scoring.criterion_count,
        len(criterion_rows) - len(missing_case_ids) == scoring.eligible_criterion_count,
        missing_case_ids == set(scoring.missing_case_ids),
        len(repeat_rows) == scoring.repeat_metric_count,
        eligible_repeat_count == scoring.eligible_repeat_metric_count,
        primary.total_case_count == scoring.criterion_count,
        primary.eligible_case_count == scoring.eligible_criterion_count,
        primary.missing_case_count == scoring.missing_criterion_count,
        {record.case_id for record in primary.missing_cases} == missing_case_ids,
    )
    if not all(expected):
        raise ExternalDiagnosticsError(
            "Diagnostic row counts or missing-case identities do not reconcile"
        )


def _create_plan(
    *,
    scoring: ExternalScoringSummary,
    primary: PrimaryAnalysisReport,
    seal: ExternalDecisionSealReport,
    criterion_publication_hash: Sha256,
    repeat_publication_hash: Sha256,
    lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ExternalDiagnosticsPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_diagnostics_plan.v1",
        "run_id": scoring.run_id,
        "scoring_summary_hash": scoring.summary_hash,
        "primary_analysis_report_hash": primary.report_hash,
        "decision_seal_report_hash": seal.report_hash,
        "criterion_publication_hash": criterion_publication_hash,
        "repeat_publication_hash": repeat_publication_hash,
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ExternalDiagnosticsPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ExternalDiagnosticsPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _estimator_diagnostics(
    estimator: RepeatEstimator,
    rows: list[dict[str, Any]],
) -> EstimatorSafetyDiagnostics:
    selected = [row for row in rows if row["estimator"] == estimator.value]
    if len(selected) != 24 or len({str(row["case_id"]) for row in selected}) != 24:
        raise ExternalDiagnosticsError(f"Estimator {estimator.value} lacks 24 unique cases")
    eligible = [row for row in selected if row["eligible"] is True]
    missing = [row for row in selected if row["eligible"] is False]
    negative = [row for row in eligible if row["criterion_demonstrated_performance"] is False]
    false_acceptance = sum(row["false_confidence_acceptance"] is True for row in negative)
    unsafe = sum(row["unsafe_advancement"] is True for row in negative)
    disagreement = sum(row["action_disagreement_with_dialogue"] is True for row in selected)
    if any(row["action_disagreement_with_dialogue"] is None for row in selected):
        raise ExternalDiagnosticsError("Policy condition lacks an action-disagreement value")
    return EstimatorSafetyDiagnostics(
        estimator=cast(PolicyEstimatorName, estimator.value),
        planned_case_count=24,
        eligible_case_count=len(eligible),
        missing_case_count=len(missing),
        criterion_negative_case_count=len(negative),
        false_acceptance=_rate(
            false_acceptance, len(negative), "eligible_criterion_negative_cases"
        ),
        unsafe_advancement=_rate(unsafe, len(negative), "eligible_criterion_negative_cases"),
        action_disagreement_with_dialogue=_rate(disagreement, len(selected), "all_committed_cases"),
    )


def _run_failures(
    seal: ExternalDecisionSealReport,
    criterion_rows: list[dict[str, Any]],
) -> RunFailureDiagnostics:
    if len(criterion_rows) != seal.planned_case_count:
        raise ExternalDiagnosticsError("Criterion rows do not match the planned case count")
    statuses = Counter(str(row["execution_status"]) for row in criterion_rows)
    sandbox = sum(statuses[status.value] for status in _SANDBOX_FAILURES)
    return RunFailureDiagnostics(
        planned_case_count=24,
        decision_complete_count=seal.complete_case_count,
        decision_precriterion_missing_count=seal.precriterion_missing_case_count,
        decision_invalid_count=seal.invalid_case_count,
        criterion_complete_count=statuses[CriterionExecutionStatus.COMPLETED.value],
        criterion_provider_failure_count=statuses[
            CriterionExecutionStatus.PROVIDER_UNAVAILABLE.value
        ],
        criterion_sandbox_failure_count=sandbox,
        criterion_sandbox_unavailable_count=statuses[
            CriterionExecutionStatus.SANDBOX_UNAVAILABLE.value
        ],
        criterion_sandbox_timeout_count=statuses[CriterionExecutionStatus.SANDBOX_TIMEOUT.value],
        criterion_sandbox_resource_limit_count=statuses[
            CriterionExecutionStatus.SANDBOX_RESOURCE_LIMIT.value
        ],
        criterion_sandbox_error_count=statuses[CriterionExecutionStatus.SANDBOX_ERROR.value],
        decision_invalid_rate=_rate(
            seal.invalid_case_count, seal.planned_case_count, "all_planned_cases"
        ),
        criterion_provider_failure_rate=_rate(
            statuses[CriterionExecutionStatus.PROVIDER_UNAVAILABLE.value],
            seal.planned_case_count,
            "all_planned_cases",
        ),
        criterion_sandbox_failure_rate=_rate(sandbox, seal.planned_case_count, "all_planned_cases"),
    )


def _rate(
    count: int,
    denominator: int,
    scope: Literal[
        "eligible_criterion_negative_cases",
        "all_committed_cases",
        "all_planned_cases",
    ],
) -> CountRate:
    return CountRate(
        count=count,
        denominator=denominator,
        rate=(count / denominator if denominator else None),
        status="measured" if denominator else "not_estimable_zero_denominator",
        denominator_scope=scope,
    )


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ExternalDiagnosticsError(f"Could not verify diagnostic source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("External-diagnostics timestamp must be UTC")
