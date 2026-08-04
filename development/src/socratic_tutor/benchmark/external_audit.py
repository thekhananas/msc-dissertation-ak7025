"""Independent pre-criterion accounting audit for an external decision seal."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import (
    authored_manifest_hash,
    project_public_manifest,
)
from socratic_tutor.benchmark.external_decision import ExternalDecisionGenerationReport
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    load_external_model_execution_protocol,
)
from socratic_tutor.benchmark.external_seal import (
    ExternalDecisionSealPlan,
    ExternalDecisionSealReport,
    ExternalEvidenceExecutionArtifact,
)
from socratic_tutor.benchmark.generation import GenerationChannel, ModelRoute
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.datasets import read_condition_predictions
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunPlan,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionStatus,
)
from socratic_tutor.benchmark.public.models import EXPECTED_CONDITIONS, PublicBenchmarkManifest
from socratic_tutor.benchmark.public.prediction_arrow import prediction_arrow_schema
from socratic_tutor.benchmark.public_rating import PublicAnswerRatingReport
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
)
from socratic_tutor.contracts import ContractModel


class ExternalDecisionAuditError(ValueError):
    """The sealed decision artifact failed independent accounting."""


class ExternalDecisionAccountingReport(ContractModel):
    """Content-addressed evidence that the decision seal is internally consistent."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_accounting_report.v1"] = (
        "benchmark.external_decision_accounting_report.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seal_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_run_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_seal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_route: ModelRoute
    system_prompt_version: str = Field(min_length=1)
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_code_revision: str = Field(min_length=1)
    decision_dirty_worktree: bool
    audit_code_revision: str = Field(min_length=1)
    audit_pixi_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    planned_case_count: int = Field(ge=1)
    complete_case_count: int = Field(ge=0)
    missing_case_count: int = Field(ge=0)
    invalid_case_count: int = Field(ge=0)
    recorded_request_count: int = Field(ge=0)
    public_request_count: int = Field(ge=0)
    evidence_request_count: int = Field(ge=0)
    evidence_execution_count: int = Field(ge=0)
    usable_evidence_count: int = Field(ge=0)
    condition_prediction_count: int = Field(ge=0)
    backend_fingerprints: tuple[str, ...]
    criterion_artifact_count: Literal[0] = 0
    checks_passed: tuple[str, ...] = Field(min_length=1)
    audited_at_utc: datetime
    gate_passed: Literal[True] = True
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> ExternalDecisionAccountingReport:
        _require_utc(self.audited_at_utc)
        if self.planned_case_count != (
            self.complete_case_count + self.missing_case_count + self.invalid_case_count
        ):
            raise ValueError("Accounting report case counts do not reconcile")
        if self.recorded_request_count != self.public_request_count + self.evidence_request_count:
            raise ValueError("Accounting report request counts do not reconcile")
        if self.condition_prediction_count != self.complete_case_count * len(EXPECTED_CONDITIONS):
            raise ValueError("Accounting report prediction count does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Accounting report hash does not match its content")
        return self


def audit_external_decision_seal(
    *,
    protocol_path: Path,
    generation_report_path: Path,
    recorded_responses_path: Path,
    public_rating_report_path: Path,
    manifest_path: Path,
    seal_root: Path,
    pixi_lock_path: Path,
    output_path: Path,
    audit_code_revision: str,
    audited_at_utc: datetime,
) -> ExternalDecisionAccountingReport:
    """Reopen and reconcile every public-side artifact before criterion access."""

    _require_utc(audited_at_utc)
    root = seal_root.resolve()
    plan = _load_json(root / "external_decision_seal_plan.json", ExternalDecisionSealPlan)
    seal_report = _load_json(
        root / "external_decision_seal_report.json", ExternalDecisionSealReport
    )
    run_plan = _load_json(root / "run_plan.json", DecisionRunPlan)
    protocol = load_external_model_execution_protocol(protocol_path)
    generation = _load_json(generation_report_path, ExternalDecisionGenerationReport)
    ratings = _load_json(public_rating_report_path, PublicAnswerRatingReport)
    records = _load_jsonl(recorded_responses_path)
    authored = load_and_verify_manifest(manifest_path)
    public = project_public_manifest(authored, projected_at_utc=protocol.created_at_utc)
    try:
        audit_pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ExternalDecisionAuditError(f"Could not read Pixi lock: {pixi_lock_path}") from error

    _require_equal("run identity", plan.run_id, seal_report.run_id, run_plan.run_id)
    _require_equal(
        "source manifest",
        authored_manifest_hash(authored),
        public.source_manifest_hash,
        protocol.source_manifest_hash,
        generation.source_manifest_hash,
        plan.source_manifest_hash,
    )
    _require_equal(
        "public projection",
        public.projection_hash,
        protocol.source_public_projection_hash,
        generation.public_projection_hash,
        plan.public_projection_hash,
        run_plan.provenance.public_manifest_hash,
    )
    _require_equal("protocol", protocol.protocol_hash, generation.protocol_hash, plan.protocol_hash)
    _require_equal(
        "generation report",
        generation.report_hash,
        plan.generation_report_hash,
        seal_report.generation_report_hash,
    )
    _require_equal(
        "public ratings",
        ratings.report_hash,
        plan.public_rating_report_hash,
        seal_report.public_rating_report_hash,
        run_plan.provenance.review_gate_hash,
    )
    _require_equal(
        "decision run plan",
        run_plan.plan_hash,
        seal_report.decision_run_plan_hash,
    )
    _require_equal("seal plan", plan.plan_hash, seal_report.seal_plan_hash)
    if run_plan.provenance.resolved_config_hash != plan.plan_hash:
        raise ExternalDecisionAuditError("Run provenance differs from the seal plan")
    expected_provenance = {
        "code revision": (run_plan.provenance.code_revision, plan.code_revision),
        "dirty-worktree flag": (run_plan.provenance.dirty_worktree, plan.dirty_worktree),
        "Pixi lock": (run_plan.provenance.pixi_lock_hash, plan.pixi_lock_hash),
        "prompt version": (run_plan.provenance.prompt_version, plan.system_prompt_version),
        "prompt hash": (run_plan.provenance.prompt_hash, plan.system_prompt_sha256),
        "tracker version": (run_plan.provenance.tracker_version, plan.tracker_version),
        "policy version": (run_plan.provenance.policy_version, plan.policy_version),
        "analysis plan": (
            run_plan.provenance.analysis_plan_hash,
            plan.inferential_hierarchy_hash,
        ),
        "calibration decision": (
            run_plan.provenance.calibration_decision_hash,
            plan.calibration_decision_hash,
        ),
        "root seed": (run_plan.provenance.root_seed, plan.root_seed),
    }
    for name, values in expected_provenance.items():
        _require_equal(name, *values)

    route = ModelRoute(provider=protocol.provider_id, model=protocol.requested_model_id)
    if plan.model_route != route:
        raise ExternalDecisionAuditError("Seal plan uses another model route")
    _audit_recorded_requests(
        records=records,
        public_case_ids={case.case_id for case in public.cases},
        protocol=protocol,
        generation=generation,
    )
    if ratings.item_count != len(public.cases):
        raise ExternalDecisionAuditError("Public rating count differs from the manifest")

    condition_store = FilesystemConditionCommitStore(root / "decision")
    seal = FilesystemGlobalDecisionSealStore(root / "decision", condition_store).load()
    if seal is None:
        raise ExternalDecisionAuditError("Global decision seal is missing")
    _require_equal("global seal", seal.seal_hash, seal_report.global_seal_hash)
    _require_equal("global run plan", seal.run_plan_hash, run_plan.plan_hash)
    if tuple(status.key for status in seal.sample_statuses) != run_plan.planned_samples:
        raise ExternalDecisionAuditError("Global seal identities differ from the run plan")

    loaded = read_condition_predictions(AtomicParquetDatasetStore(root / "datasets"))
    publication = loaded.manifest
    _require_equal(
        "decision publication",
        publication.publication_hash,
        seal_report.decision_publication_hash,
    )
    _require_equal("publication seal", publication.decision_seal_hash, seal.seal_hash)
    if not loaded.table.schema.equals(prediction_arrow_schema(), check_metadata=True):
        raise ExternalDecisionAuditError("Prediction table schema differs from its contract")

    evidence_records = {
        record.request.case_id: record
        for record in records
        if record.request.channel is GenerationChannel.EVIDENCE
    }
    execution_files = tuple(sorted((root / "decision" / "evidence_executions").glob("*.json")))
    executions = tuple(
        _load_json(path, ExternalEvidenceExecutionArtifact) for path in execution_files
    )
    _audit_evidence_executions(
        executions=executions,
        evidence_records=evidence_records,
        manifest=public,
        benchmark_root=manifest_path.resolve().parent,
    )

    criterion_paths = tuple(
        path for path in root.rglob("*") if path.is_file() and "criterion" in path.name.lower()
    )
    if criterion_paths:
        raise ExternalDecisionAuditError("Criterion artifacts exist inside the decision seal")

    status_counts = {
        status: sum(item.status is status for item in seal.sample_statuses)
        for status in SampleDecisionStatus
    }
    expected_accounting = {
        "planned cases": (seal_report.planned_case_count, len(run_plan.planned_samples)),
        "complete cases": (
            seal_report.complete_case_count,
            status_counts[SampleDecisionStatus.COMPLETE],
        ),
        "missing cases": (
            seal_report.precriterion_missing_case_count,
            status_counts[SampleDecisionStatus.PRECRITERION_MISSING],
        ),
        "invalid cases": (
            seal_report.invalid_case_count,
            status_counts[SampleDecisionStatus.INVALID],
        ),
        "evidence executions": (seal_report.evidence_execution_count, len(executions)),
        "usable evidence": (
            seal_report.usable_evidence_count,
            sum(item.usable for item in executions),
        ),
        "condition predictions": (seal_report.prediction_count, publication.row_count),
    }
    for name, values in expected_accounting.items():
        _require_equal(name, *values)
    if seal_report.all_cases_complete != (
        status_counts[SampleDecisionStatus.COMPLETE] == len(run_plan.planned_samples)
    ):
        raise ExternalDecisionAuditError("All-complete flag differs from global accounting")
    backend_fingerprints = tuple(
        sorted(
            {
                record.provider_metadata.backend_fingerprint
                for record in records
                if record.provider_metadata.backend_fingerprint
            }
        )
    )
    checks = (
        "source_hashes",
        "run_and_case_identities",
        "model_and_request_configuration",
        "recorded_response_integrity",
        "public_rating_coverage",
        "evidence_execution_integrity",
        "condition_set_integrity",
        "global_seal_integrity",
        "parquet_schema_and_hash",
        "publication_lineage",
        "missingness_accounting",
        "criterion_absence",
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_accounting_report.v1",
        "run_id": plan.run_id,
        "benchmark_version": "v1",
        "source_manifest_hash": plan.source_manifest_hash,
        "public_projection_hash": plan.public_projection_hash,
        "protocol_hash": plan.protocol_hash,
        "generation_report_hash": generation.report_hash,
        "seal_plan_hash": plan.plan_hash,
        "decision_run_plan_hash": run_plan.plan_hash,
        "global_seal_hash": seal.seal_hash,
        "decision_publication_hash": publication.publication_hash,
        "model_route": route,
        "system_prompt_version": plan.system_prompt_version,
        "system_prompt_sha256": plan.system_prompt_sha256,
        "decision_code_revision": plan.code_revision,
        "decision_dirty_worktree": plan.dirty_worktree,
        "audit_code_revision": audit_code_revision,
        "audit_pixi_lock_hash": audit_pixi_lock_hash,
        "tracker_version": plan.tracker_version,
        "policy_version": plan.policy_version,
        "planned_case_count": len(run_plan.planned_samples),
        "complete_case_count": status_counts[SampleDecisionStatus.COMPLETE],
        "missing_case_count": status_counts[SampleDecisionStatus.PRECRITERION_MISSING],
        "invalid_case_count": status_counts[SampleDecisionStatus.INVALID],
        "recorded_request_count": len(records),
        "public_request_count": sum(
            record.request.channel is GenerationChannel.PUBLIC for record in records
        ),
        "evidence_request_count": len(evidence_records),
        "evidence_execution_count": len(executions),
        "usable_evidence_count": sum(item.usable for item in executions),
        "condition_prediction_count": publication.row_count,
        "backend_fingerprints": backend_fingerprints,
        "criterion_artifact_count": 0,
        "checks_passed": checks,
        "audited_at_utc": audited_at_utc,
        "gate_passed": True,
    }
    draft = ExternalDecisionAccountingReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalDecisionAccountingReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_path, report)
    return report


def _audit_recorded_requests(
    *,
    records: tuple[RecordedGenerationResponse, ...],
    public_case_ids: set[str],
    protocol: ExternalModelExecutionProtocol,
    generation: ExternalDecisionGenerationReport,
) -> None:
    expected = {
        (case_id, channel)
        for case_id in public_case_ids
        for channel in (GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE)
    }
    observed = {(record.request.case_id, record.request.channel) for record in records}
    if observed != expected or len(records) != len(expected):
        raise ExternalDecisionAuditError("Recorded requests do not cover the decision matrix")
    route = ModelRoute(provider=protocol.provider_id, model=protocol.requested_model_id)
    for record in records:
        request = record.request
        if (
            record.original_source is not OriginalGenerationSource.CEREBRAS
            or request.run_id != generation.run_id
            or request.model_route != route
            or request.system_prompt_version != protocol.system_prompt_version
            or request.system_prompt_sha256 != protocol.system_prompt_sha256
            or request.sampling.temperature != protocol.temperature
            or request.sampling.max_output_tokens != protocol.max_output_tokens
        ):
            raise ExternalDecisionAuditError("Recorded request differs from the frozen protocol")
    attempts = {attempt.request_hash: attempt for attempt in generation.attempts}
    if len(attempts) != len(generation.attempts):
        raise ExternalDecisionAuditError("Generation report contains duplicate request attempts")
    for record in records:
        attempt = attempts.get(record.request.request_hash)
        if (
            attempt is None
            or attempt.case_id != record.request.case_id
            or attempt.channel is not record.request.channel
            or attempt.status != "success"
            or attempt.response_hash != record.response_hash
            or attempt.backend_fingerprint != record.provider_metadata.backend_fingerprint
        ):
            raise ExternalDecisionAuditError(
                "Generation attempt differs from its recorded request or response"
            )
    response_hash = canonical_sha256({"responses": records})
    if response_hash != generation.recorded_responses_hash:
        raise ExternalDecisionAuditError("Recorded response collection hash differs")
    if generation.success_count != len(records) or not generation.gate_passed:
        raise ExternalDecisionAuditError("Generation report does not account for all recordings")


def _audit_evidence_executions(
    *,
    executions: tuple[ExternalEvidenceExecutionArtifact, ...],
    evidence_records: dict[str, RecordedGenerationResponse],
    manifest: PublicBenchmarkManifest,
    benchmark_root: Path,
) -> None:
    cases = manifest.cases
    if len(executions) != len(cases):
        raise ExternalDecisionAuditError("Evidence execution count differs from the manifest")
    by_case = {execution.case_id: execution for execution in executions}
    if set(by_case) != {case.case_id for case in cases}:
        raise ExternalDecisionAuditError("Evidence executions cover another case set")
    for case in cases:
        execution = by_case[case.case_id]
        record = evidence_records[case.case_id]
        try:
            test_hash = file_sha256((benchmark_root / case.evidence_test_ref).read_bytes())
        except OSError as error:
            raise ExternalDecisionAuditError(
                f"Could not read evidence tests for {case.case_id}"
            ) from error
        if (
            execution.response_hash != record.response_hash
            or execution.sample_id != record.request.sample_id
            or execution.evidence_test_ref != case.evidence_test_ref
            or execution.evidence_test_sha256 != test_hash
        ):
            raise ExternalDecisionAuditError(
                f"Evidence execution differs from frozen inputs: {case.case_id}"
            )


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ExternalDecisionAuditError(f"Could not verify audit source: {path}") from error


def _load_jsonl(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExternalDecisionAuditError(f"Could not read recorded responses: {path}") from error
    records: list[RecordedGenerationResponse] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(RecordedGenerationResponse.model_validate_json(line))
        except ValidationError as error:
            raise ExternalDecisionAuditError(
                f"Invalid recorded response at line {line_number}"
            ) from error
    return tuple(records)


def _require_equal(name: str, *values: object) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise ExternalDecisionAuditError(f"External decision audit has mismatched {name}")


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("External decision audit timestamp must be UTC")
