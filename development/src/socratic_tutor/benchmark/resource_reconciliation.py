"""Operational cost and implementation-burden reconciliation for the external study."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.external_criterion import ExternalCriterionReport
from socratic_tutor.benchmark.external_decision import ExternalDecisionGenerationReport
from socratic_tutor.benchmark.external_seal import (
    ExternalDecisionSealReport,
    ExternalEvidenceExecutionArtifact,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.benchmark.probe_correction_sensitivity import (
    ProbeCorrectionSensitivityReport,
)
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel


class ResourceReconciliationError(ValueError):
    """Resource inputs do not describe the same external benchmark run."""


class ProviderPhaseSummary(ContractModel):
    """Measured provider workload for one generation phase."""

    phase: Literal["decision_generation", "criterion_generation"]
    response_count: int = Field(ge=1)
    responses_with_token_usage: int = Field(ge=0)
    responses_with_latency: int = Field(ge=0)
    responses_with_reported_cost: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    summed_provider_latency_ms: int = Field(ge=0)
    first_response_at_utc: datetime
    last_response_at_utc: datetime
    response_capture_span_seconds: float = Field(ge=0.0)
    provider_ids: tuple[str, ...] = Field(min_length=1)
    model_ids: tuple[str, ...] = Field(min_length=1)
    provider_failure_count: int = Field(ge=0)
    retry_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_summary(self) -> ProviderPhaseSummary:
        _require_utc(self.first_response_at_utc)
        _require_utc(self.last_response_at_utc)
        if self.first_response_at_utc > self.last_response_at_utc:
            raise ValueError("Provider phase starts after it ends")
        if any(
            count > self.response_count
            for count in (
                self.responses_with_token_usage,
                self.responses_with_latency,
                self.responses_with_reported_cost,
            )
        ):
            raise ValueError("Provider metadata count exceeds response count")
        expected_span = (self.last_response_at_utc - self.first_response_at_utc).total_seconds()
        if abs(self.response_capture_span_seconds - expected_span) > 1e-6:
            raise ValueError("Provider response span does not reconcile")
        return self


class SandboxWorkloadSummary(ContractModel):
    """Observed evidence and criterion sandbox workload."""

    evidence_execution_count: int = Field(ge=0)
    evidence_completed_count: int = Field(ge=0)
    evidence_test_pass_count: int = Field(ge=0)
    evidence_test_fail_count: int = Field(ge=0)
    reviewed_test_defect_count: int = Field(ge=0)
    evidence_test_fail_count_after_reviewed_correction: int = Field(ge=0)
    criterion_execution_count: int = Field(ge=0)
    criterion_completed_count: int = Field(ge=0)
    criterion_missing_count: int = Field(ge=0)
    total_execution_count: int = Field(ge=0)
    total_completed_count: int = Field(ge=0)
    execution_time_status: Literal["not_recorded"] = "not_recorded"
    cpu_and_memory_status: Literal["not_recorded"] = "not_recorded"

    @model_validator(mode="after")
    def validate_summary(self) -> SandboxWorkloadSummary:
        if self.evidence_completed_count > self.evidence_execution_count:
            raise ValueError("Completed evidence executions exceed attempts")
        if self.evidence_test_pass_count + self.evidence_test_fail_count != (
            self.evidence_completed_count
        ):
            raise ValueError("Evidence test outcomes do not reconcile")
        if self.reviewed_test_defect_count > self.evidence_test_fail_count:
            raise ValueError("Reviewed defects exceed observed evidence-test failures")
        if self.evidence_test_fail_count_after_reviewed_correction != (
            self.evidence_test_fail_count - self.reviewed_test_defect_count
        ):
            raise ValueError("Corrected evidence-test failure count does not reconcile")
        if self.criterion_completed_count + self.criterion_missing_count != (
            self.criterion_execution_count
        ):
            raise ValueError("Criterion execution outcomes do not reconcile")
        if self.total_execution_count != (
            self.evidence_execution_count + self.criterion_execution_count
        ):
            raise ValueError("Total sandbox execution count does not reconcile")
        if self.total_completed_count != (
            self.evidence_completed_count + self.criterion_completed_count
        ):
            raise ValueError("Total completed execution count does not reconcile")
        return self


class ConditionBurdenSummary(ContractModel):
    """Predictive quality and incremental machinery for one reported condition."""

    condition: Literal[
        "dialogue_only",
        "probe_informed_observed",
        "probe_informed_post_hoc_corrected",
    ]
    evidence_status: Literal["sealed_primary", "post_hoc_correction"]
    eligible_case_count: int = Field(ge=1)
    correct_case_count: int = Field(ge=0)
    accuracy_on_fixed_authored_cases: float = Field(ge=0.0, le=1.0)
    external_model_response_channels_per_case: int = Field(ge=1)
    sandbox_executions_per_case: int = Field(ge=0)
    required_channels: tuple[Literal["public", "evidence"], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary(self) -> ConditionBurdenSummary:
        if self.correct_case_count > self.eligible_case_count:
            raise ValueError("Condition correct count exceeds eligible cases")
        expected = self.correct_case_count / self.eligible_case_count
        if abs(self.accuracy_on_fixed_authored_cases - expected) > 1e-12:
            raise ValueError("Condition accuracy does not reconcile")
        if self.external_model_response_channels_per_case != len(self.required_channels):
            raise ValueError("Condition response-channel burden does not reconcile")
        if self.condition == "dialogue_only" and self.evidence_status != "sealed_primary":
            raise ValueError("Dialogue-only burden must use sealed evidence")
        if self.condition.endswith("corrected") and self.evidence_status != "post_hoc_correction":
            raise ValueError("Corrected burden must be labelled post-hoc")
        return self


class ResourceReconciliationPlan(ContractModel):
    """Source and software identities for one operational reconciliation."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.resource_reconciliation_plan.v1"] = (
        "benchmark.resource_reconciliation_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    generation_report_hash: Sha256
    criterion_report_hash: Sha256
    decision_seal_report_hash: Sha256
    primary_report_hash: Sha256
    correction_report_hash: Sha256
    decision_responses_sha256: Sha256
    criterion_responses_sha256: Sha256
    method: Literal["observed_external_workload_reconciliation_v1"] = (
        "observed_external_workload_reconciliation_v1"
    )
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> ResourceReconciliationPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Resource-reconciliation plan hash does not match")
        return self


class ResourceReconciliationReport(ContractModel):
    """Measured workload, unavailable quantities, and quality-burden comparison."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.resource_reconciliation_report.v1"] = (
        "benchmark.resource_reconciliation_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    provider_phases: tuple[ProviderPhaseSummary, ProviderPhaseSummary]
    total_provider_response_count: int = Field(ge=1)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_summed_provider_latency_ms: int = Field(ge=0)
    provider_failure_count: int = Field(ge=0)
    provider_cost_status: Literal["not_reported_by_provider"] = "not_reported_by_provider"
    provider_cost_usd: None = None
    sandbox: SandboxWorkloadSummary
    condition_burden: tuple[
        ConditionBurdenSummary,
        ConditionBurdenSummary,
        ConditionBurdenSummary,
    ]
    measured_artifact_bytes_excluding_this_report: int = Field(ge=0)
    local_analysis_network_calls: Literal[0] = 0
    local_analysis_sandbox_calls: Literal[0] = 0
    interpretation_scope: Literal["observed_single_run_workload_not_deployment_cost"] = (
        "observed_single_run_workload_not_deployment_cost"
    )
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> ResourceReconciliationReport:
        _require_utc(self.completed_at_utc)
        if {phase.phase for phase in self.provider_phases} != {
            "decision_generation",
            "criterion_generation",
        }:
            raise ValueError("Resource report requires both provider phases")
        if self.total_provider_response_count != sum(
            phase.response_count for phase in self.provider_phases
        ):
            raise ValueError("Total provider responses do not reconcile")
        if self.total_input_tokens != sum(phase.input_tokens for phase in self.provider_phases):
            raise ValueError("Total input tokens do not reconcile")
        if self.total_output_tokens != sum(phase.output_tokens for phase in self.provider_phases):
            raise ValueError("Total output tokens do not reconcile")
        if self.total_summed_provider_latency_ms != sum(
            phase.summed_provider_latency_ms for phase in self.provider_phases
        ):
            raise ValueError("Total provider latency does not reconcile")
        if self.provider_failure_count != sum(
            phase.provider_failure_count for phase in self.provider_phases
        ):
            raise ValueError("Total provider failures do not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Resource-reconciliation report hash does not match")
        return self


def run_resource_reconciliation(
    *,
    generation_report_path: Path,
    criterion_report_path: Path,
    decision_seal_report_path: Path,
    primary_report_path: Path,
    correction_report_path: Path,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    evidence_execution_root: Path,
    measured_artifact_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ResourceReconciliationReport:
    """Reconcile observed quality with external and local execution burden."""

    _require_utc(created_at_utc)
    generation = _load_json(generation_report_path, ExternalDecisionGenerationReport)
    criterion = _load_json(criterion_report_path, ExternalCriterionReport)
    seal = _load_json(decision_seal_report_path, ExternalDecisionSealReport)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    correction = _load_json(correction_report_path, ProbeCorrectionSensitivityReport)
    sources = (generation, criterion, seal, primary, correction)
    if len({source.run_id for source in sources}) != 1:
        raise ResourceReconciliationError("Resource sources belong to different runs")
    if len({source.benchmark_version for source in sources}) != 1:
        raise ResourceReconciliationError("Resource sources use different benchmarks")
    if correction.original.inference != primary.inference:
        raise ResourceReconciliationError("Correction report does not reconstruct the primary")
    if seal.generation_report_hash != generation.report_hash:
        raise ResourceReconciliationError("Decision seal belongs to another generation report")
    if created_at_utc <= max(
        generation.generated_at_utc,
        criterion.completed_at_utc,
        primary.completed_at_utc,
        correction.completed_at_utc,
    ):
        raise ResourceReconciliationError("Resource reconciliation must follow source reports")

    decision_responses = _load_jsonl(decision_responses_path)
    criterion_responses = _load_jsonl(criterion_responses_path)
    if len(decision_responses) != generation.success_count:
        raise ResourceReconciliationError("Decision response count differs from generation report")
    if len(criterion_responses) != criterion.recorded_response_count:
        raise ResourceReconciliationError("Criterion response count differs from criterion report")
    evidence_executions = _load_evidence_executions(evidence_execution_root)
    if len(evidence_executions) != seal.evidence_execution_count:
        raise ResourceReconciliationError("Evidence executions differ from decision seal")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
        decision_responses_hash = file_sha256(decision_responses_path.read_bytes())
        criterion_responses_hash = file_sha256(criterion_responses_path.read_bytes())
    except OSError as error:
        raise ResourceReconciliationError("Could not hash a resource input") from error

    plan = _create_plan(
        generation=generation,
        criterion=criterion,
        seal=seal,
        primary=primary,
        correction=correction,
        decision_responses_hash=decision_responses_hash,
        criterion_responses_hash=criterion_responses_hash,
        pixi_lock_hash=pixi_lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "resource_reconciliation_report.json"
    if report_path.exists():
        report = _load_json(report_path, ResourceReconciliationReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise ResourceReconciliationError("Existing resource report belongs to another plan")
        return report
    write_immutable_json(root / "resource_reconciliation_plan.json", plan)

    decision_phase = _provider_phase(
        "decision_generation",
        decision_responses,
        provider_failure_count=generation.provider_failure_count,
        retry_count=sum((attempt.attempt_count or 1) - 1 for attempt in generation.attempts),
    )
    criterion_phase = _provider_phase(
        "criterion_generation",
        criterion_responses,
        provider_failure_count=criterion.provider_missing_count,
        retry_count=None,
    )
    phases = (decision_phase, criterion_phase)
    sandbox = _sandbox_summary(evidence_executions, criterion, correction)
    burden = _condition_burden(primary, correction)
    artifact_bytes = _tree_size_excluding(measured_artifact_root, root)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.resource_reconciliation_report.v1",
        "benchmark_version": generation.benchmark_version,
        "run_id": generation.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "provider_phases": phases,
        "total_provider_response_count": sum(phase.response_count for phase in phases),
        "total_input_tokens": sum(phase.input_tokens for phase in phases),
        "total_output_tokens": sum(phase.output_tokens for phase in phases),
        "total_summed_provider_latency_ms": sum(
            phase.summed_provider_latency_ms for phase in phases
        ),
        "provider_failure_count": sum(phase.provider_failure_count for phase in phases),
        "provider_cost_status": "not_reported_by_provider",
        "provider_cost_usd": None,
        "sandbox": sandbox,
        "condition_burden": burden,
        "measured_artifact_bytes_excluding_this_report": artifact_bytes,
        "local_analysis_network_calls": 0,
        "local_analysis_sandbox_calls": 0,
        "interpretation_scope": "observed_single_run_workload_not_deployment_cost",
        "completed_at_utc": created_at_utc,
    }
    draft = ResourceReconciliationReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ResourceReconciliationReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _provider_phase(
    phase: Literal["decision_generation", "criterion_generation"],
    responses: tuple[RecordedGenerationResponse, ...],
    *,
    provider_failure_count: int,
    retry_count: int | None,
) -> ProviderPhaseSummary:
    times = tuple(response.captured_at_utc for response in responses)
    first = min(times)
    last = max(times)
    token_records = tuple(
        response
        for response in responses
        if response.provider_metadata.input_tokens is not None
        and response.provider_metadata.output_tokens is not None
    )
    latency_records = tuple(
        response for response in responses if response.provider_metadata.latency_ms is not None
    )
    cost_records = tuple(
        response for response in responses if response.provider_metadata.cost_usd_micros is not None
    )
    provider_values: set[str] = set()
    model_values: set[str] = set()
    for response in responses:
        provider_id = response.provider_metadata.resolved_provider_id
        model_id = response.provider_metadata.resolved_model_id
        if isinstance(provider_id, str):
            provider_values.add(provider_id)
        if isinstance(model_id, str):
            model_values.add(model_id)
    provider_ids = tuple(sorted(provider_values))
    model_ids = tuple(sorted(model_values))
    return ProviderPhaseSummary(
        phase=phase,
        response_count=len(responses),
        responses_with_token_usage=len(token_records),
        responses_with_latency=len(latency_records),
        responses_with_reported_cost=len(cost_records),
        input_tokens=sum(response.provider_metadata.input_tokens or 0 for response in responses),
        output_tokens=sum(response.provider_metadata.output_tokens or 0 for response in responses),
        summed_provider_latency_ms=sum(
            response.provider_metadata.latency_ms or 0 for response in responses
        ),
        first_response_at_utc=first,
        last_response_at_utc=last,
        response_capture_span_seconds=(last - first).total_seconds(),
        provider_ids=provider_ids,
        model_ids=model_ids,
        provider_failure_count=provider_failure_count,
        retry_count=retry_count,
    )


def _sandbox_summary(
    evidence: tuple[ExternalEvidenceExecutionArtifact, ...],
    criterion: ExternalCriterionReport,
    correction: ProbeCorrectionSensitivityReport,
) -> SandboxWorkloadSummary:
    completed = tuple(item for item in evidence if item.outcome_status == "completed")
    passed = sum(item.failed == 0 for item in completed)
    failed = sum(item.failed is not None and item.failed > 0 for item in completed)
    return SandboxWorkloadSummary(
        evidence_execution_count=len(evidence),
        evidence_completed_count=len(completed),
        evidence_test_pass_count=passed,
        evidence_test_fail_count=failed,
        reviewed_test_defect_count=correction.correction_count,
        evidence_test_fail_count_after_reviewed_correction=(failed - correction.correction_count),
        criterion_execution_count=criterion.planned_case_count,
        criterion_completed_count=criterion.completed_execution_count,
        criterion_missing_count=criterion.sandbox_missing_count,
        total_execution_count=len(evidence) + criterion.planned_case_count,
        total_completed_count=len(completed) + criterion.completed_execution_count,
        execution_time_status="not_recorded",
        cpu_and_memory_status="not_recorded",
    )


def _condition_burden(
    primary: PrimaryAnalysisReport,
    correction: ProbeCorrectionSensitivityReport,
) -> tuple[ConditionBurdenSummary, ConditionBurdenSummary, ConditionBurdenSummary]:
    eligible = primary.eligible_case_count
    dialogue_correct = sum(record.dialogue_correct for record in primary.case_results)
    observed_probe_correct = sum(record.valid_evidence_correct for record in primary.case_results)
    corrected_probe_correct = (
        correction.corrected.mcnemar.both_correct_count
        + correction.corrected.mcnemar.right_only_correct_count
    )
    return (
        ConditionBurdenSummary(
            condition="dialogue_only",
            evidence_status="sealed_primary",
            eligible_case_count=eligible,
            correct_case_count=dialogue_correct,
            accuracy_on_fixed_authored_cases=dialogue_correct / eligible,
            external_model_response_channels_per_case=1,
            sandbox_executions_per_case=0,
            required_channels=("public",),
        ),
        ConditionBurdenSummary(
            condition="probe_informed_observed",
            evidence_status="sealed_primary",
            eligible_case_count=eligible,
            correct_case_count=observed_probe_correct,
            accuracy_on_fixed_authored_cases=observed_probe_correct / eligible,
            external_model_response_channels_per_case=2,
            sandbox_executions_per_case=1,
            required_channels=("public", "evidence"),
        ),
        ConditionBurdenSummary(
            condition="probe_informed_post_hoc_corrected",
            evidence_status="post_hoc_correction",
            eligible_case_count=eligible,
            correct_case_count=corrected_probe_correct,
            accuracy_on_fixed_authored_cases=corrected_probe_correct / eligible,
            external_model_response_channels_per_case=2,
            sandbox_executions_per_case=1,
            required_channels=("public", "evidence"),
        ),
    )


def _create_plan(
    *,
    generation: ExternalDecisionGenerationReport,
    criterion: ExternalCriterionReport,
    seal: ExternalDecisionSealReport,
    primary: PrimaryAnalysisReport,
    correction: ProbeCorrectionSensitivityReport,
    decision_responses_hash: Sha256,
    criterion_responses_hash: Sha256,
    pixi_lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ResourceReconciliationPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.resource_reconciliation_plan.v1",
        "benchmark_version": generation.benchmark_version,
        "run_id": generation.run_id,
        "generation_report_hash": generation.report_hash,
        "criterion_report_hash": criterion.report_hash,
        "decision_seal_report_hash": seal.report_hash,
        "primary_report_hash": primary.report_hash,
        "correction_report_hash": correction.report_hash,
        "decision_responses_sha256": decision_responses_hash,
        "criterion_responses_sha256": criterion_responses_hash,
        "method": "observed_external_workload_reconciliation_v1",
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ResourceReconciliationPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ResourceReconciliationPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _load_jsonl(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = tuple(
            RecordedGenerationResponse.model_validate_json(line) for line in lines if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise ResourceReconciliationError(f"Could not verify response file: {path}") from error
    if not records:
        raise ResourceReconciliationError(f"Response file is empty: {path}")
    if len({record.request.request_hash for record in records}) != len(records):
        raise ResourceReconciliationError("Response file contains duplicate requests")
    return records


def _load_evidence_executions(path: Path) -> tuple[ExternalEvidenceExecutionArtifact, ...]:
    try:
        records = tuple(
            ExternalEvidenceExecutionArtifact.model_validate_json(item.read_bytes())
            for item in sorted(path.glob("*.json"))
        )
    except (OSError, ValidationError) as error:
        raise ResourceReconciliationError(
            f"Could not verify evidence executions: {path}"
        ) from error
    if not records:
        raise ResourceReconciliationError("Evidence execution directory is empty")
    if len({record.case_id for record in records}) != len(records):
        raise ResourceReconciliationError("Evidence executions contain duplicate cases")
    return records


def _tree_size_excluding(root: Path, excluded: Path) -> int:
    measured_root = root.resolve()
    excluded_root = excluded.resolve()
    total = 0
    try:
        for path in measured_root.rglob("*"):
            resolved = path.resolve()
            if excluded_root == resolved or excluded_root in resolved.parents:
                continue
            if path.is_file():
                total += path.stat().st_size
    except OSError as error:
        raise ResourceReconciliationError(
            f"Could not measure artifact storage: {measured_root}"
        ) from error
    return total


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ResourceReconciliationError(f"Could not verify resource source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Resource-reconciliation timestamp must be UTC")
