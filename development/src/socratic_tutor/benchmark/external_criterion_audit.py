"""Read-only integrity audit after external criterion reveal."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.criterion import (
    CriterionExecutionStatus,
    CriterionRecord,
)
from socratic_tutor.benchmark.evaluator.datasets import (
    criterion_arrow_schema,
    read_criterion_records,
)
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import EvaluatorBenchmarkManifest
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.external_audit import ExternalDecisionAccountingReport
from socratic_tutor.benchmark.external_criterion import (
    ExternalCriterionPlan,
    ExternalCriterionReport,
    ExternalCriterionRequestArtifact,
)
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    load_external_model_execution_protocol,
)
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationChannel,
    ModelRoute,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.datasets import read_condition_predictions
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunPlan,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionStatus,
    SampleDecisionStatusRecord,
)
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.public.prediction_arrow import prediction_arrow_schema
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
)
from socratic_tutor.contracts import ContractModel


class ExternalCriterionAuditError(ValueError):
    """The revealed criterion artifact failed independent integrity checks."""


class ExternalCriterionIntegrityReport(ContractModel):
    """Content-addressed post-reveal route, label, and timing audit."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_criterion_integrity_report.v1"] = (
        "benchmark.external_criterion_integrity_report.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounting_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_seal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_route: ModelRoute
    resolved_model_ids: tuple[str, ...]
    backend_fingerprints: tuple[str, ...]
    planned_case_count: int = Field(ge=1)
    decision_prediction_count: int = Field(ge=0)
    criterion_attempt_count: int = Field(ge=0)
    provider_success_count: int = Field(ge=0)
    provider_missing_count: int = Field(ge=0)
    criterion_record_count: int = Field(ge=0)
    completed_execution_count: int = Field(ge=0)
    missing_outcome_count: int = Field(ge=0)
    missing_case_ids: tuple[str, ...]
    demonstrated_count: int = Field(ge=0)
    recorded_response_count: int = Field(ge=0)
    recorded_response_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_record_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_code_revision: str = Field(min_length=1)
    audit_pixi_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks_passed: tuple[str, ...] = Field(min_length=1)
    audited_at_utc: datetime
    gate_passed: Literal[True] = True
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> ExternalCriterionIntegrityReport:
        _require_utc(self.audited_at_utc)
        if self.criterion_attempt_count != (
            self.provider_success_count + self.provider_missing_count
        ):
            raise ValueError("Criterion provider counts do not reconcile")
        if self.criterion_record_count != self.planned_case_count:
            raise ValueError("Criterion record count does not cover the plan")
        if self.criterion_record_count != (
            self.completed_execution_count + self.missing_outcome_count
        ):
            raise ValueError("Criterion outcome counts do not reconcile")
        if len(self.missing_case_ids) != self.missing_outcome_count:
            raise ValueError("Missing criterion identities do not reconcile")
        if self.recorded_response_count != self.provider_success_count:
            raise ValueError("Recorded response count differs from provider successes")
        if self.demonstrated_count > self.completed_execution_count:
            raise ValueError("Demonstrated count exceeds completed outcomes")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Criterion integrity report hash does not match its content")
        return self


def audit_external_criterion_run(
    *,
    protocol_path: Path,
    accounting_report_path: Path,
    manifest_path: Path,
    system_prompt_path: Path,
    seal_root: Path,
    pixi_lock_path: Path,
    output_path: Path,
    audit_code_revision: str,
    audited_at_utc: datetime,
) -> ExternalCriterionIntegrityReport:
    """Reconcile the criterion phase without making external calls or scoring conditions."""

    _require_utc(audited_at_utc)
    root = seal_root.resolve()
    criterion_root = root / "criterion"
    protocol = load_external_model_execution_protocol(protocol_path)
    accounting = _load_json(accounting_report_path, ExternalDecisionAccountingReport)
    plan = _load_json(criterion_root / "external_criterion_plan.json", ExternalCriterionPlan)
    criterion_report = _load_json(
        criterion_root / "external_criterion_report.json", ExternalCriterionReport
    )
    run_plan = _load_json(root / "run_plan.json", DecisionRunPlan)
    authored = load_and_verify_manifest(manifest_path)
    public = project_public_manifest(authored, projected_at_utc=protocol.created_at_utc)
    evaluator = project_evaluator_manifest(authored, public, projected_at_utc=plan.created_at_utc)
    _audit_sources(
        protocol=protocol,
        accounting=accounting,
        plan=plan,
        report=criterion_report,
        public=public,
        evaluator=evaluator,
        system_prompt_path=system_prompt_path,
        pixi_lock_path=pixi_lock_path,
    )

    condition_store = FilesystemConditionCommitStore(root / "decision")
    seal = FilesystemGlobalDecisionSealStore(root / "decision", condition_store).load()
    if seal is None:
        raise ExternalCriterionAuditError("Global decision seal is missing")
    _require_equal(
        "global seal",
        seal.seal_hash,
        accounting.global_seal_hash,
        plan.global_seal_hash,
        criterion_report.global_seal_hash,
    )
    if seal.run_plan_hash != run_plan.plan_hash:
        raise ExternalCriterionAuditError("Global seal no longer matches its run plan")
    complete_statuses = tuple(
        item for item in seal.sample_statuses if item.status is SampleDecisionStatus.COMPLETE
    )
    complete_keys = tuple(item.key for item in complete_statuses)
    if len(complete_keys) != plan.expected_case_count:
        raise ExternalCriterionAuditError("Complete sealed cases differ from criterion scope")

    datasets = AtomicParquetDatasetStore(root / "datasets")
    predictions = read_condition_predictions(datasets)
    if (
        predictions.manifest.publication_hash != accounting.decision_publication_hash
        or predictions.manifest.row_count != accounting.condition_prediction_count
        or not predictions.table.schema.equals(prediction_arrow_schema(), check_metadata=True)
    ):
        raise ExternalCriterionAuditError("Sealed decision publication changed after reveal")

    attempt_paths = tuple(sorted((criterion_root / "attempts").glob("*.json")))
    record_paths = tuple(sorted((criterion_root / "records").glob("*.json")))
    attempts = tuple(_load_json(path, ExternalCriterionRequestArtifact) for path in attempt_paths)
    records = tuple(_load_json(path, CriterionRecord) for path in record_paths)
    responses = _load_jsonl(criterion_root / "recorded_responses.jsonl")
    audit_external_criterion_rows(
        plan=plan,
        accounting=accounting,
        evaluator=evaluator,
        benchmark_root=manifest_path.resolve().parent,
        complete_statuses=complete_statuses,
        attempts=attempts,
        records=records,
        responses=responses,
    )

    criterion_dataset = read_criterion_records(datasets)
    publication = criterion_dataset.manifest
    ordered_records = tuple(sorted(records, key=lambda item: complete_keys.index(item.key)))
    if (
        publication.publication_hash != criterion_report.criterion_publication_hash
        or publication.decision_seal_hash != seal.seal_hash
        or publication.row_count != len(ordered_records)
        or publication.source_record_hashes
        != tuple(record.record_hash for record in ordered_records)
        or not criterion_dataset.table.schema.equals(criterion_arrow_schema(), check_metadata=True)
    ):
        raise ExternalCriterionAuditError("Criterion Parquet publication differs from records")

    successful = tuple(item for item in attempts if item.response is not None)
    completed = tuple(
        item for item in records if item.execution.status is CriterionExecutionStatus.COMPLETED
    )
    missing = tuple(
        item for item in records if item.execution.status is not CriterionExecutionStatus.COMPLETED
    )
    resolved_models = tuple(
        sorted(
            {
                item.response.provider_metadata.resolved_model_id
                for item in successful
                if item.response is not None
                and item.response.provider_metadata.resolved_model_id is not None
            }
        )
    )
    fingerprints = tuple(
        sorted(
            {
                item.response.provider_metadata.backend_fingerprint
                for item in successful
                if item.response is not None
                and item.response.provider_metadata.backend_fingerprint is not None
            }
        )
    )
    checks = (
        "precriterion_audit_lineage",
        "source_and_projection_hashes",
        "model_route_and_sampling",
        "provider_response_integrity",
        "criterion_request_coverage",
        "criterion_label_derivation",
        "criterion_test_and_rubric_hashes",
        "reveal_after_decision_seal",
        "typed_missingness",
        "decision_publication_unchanged",
        "criterion_parquet_integrity",
        "request_budget_accounting",
    )
    response_root = canonical_sha256(tuple(item.response_hash for item in responses))
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_integrity_report.v1",
        "run_id": plan.run_id,
        "benchmark_version": "v1",
        "protocol_hash": protocol.protocol_hash,
        "accounting_report_hash": accounting.report_hash,
        "criterion_plan_hash": plan.plan_hash,
        "criterion_report_hash": criterion_report.report_hash,
        "global_seal_hash": seal.seal_hash,
        "decision_publication_hash": predictions.manifest.publication_hash,
        "criterion_publication_hash": publication.publication_hash,
        "model_route": plan.model_route,
        "resolved_model_ids": resolved_models,
        "backend_fingerprints": fingerprints,
        "planned_case_count": len(complete_keys),
        "decision_prediction_count": predictions.manifest.row_count,
        "criterion_attempt_count": len(attempts),
        "provider_success_count": len(successful),
        "provider_missing_count": len(attempts) - len(successful),
        "criterion_record_count": len(records),
        "completed_execution_count": len(completed),
        "missing_outcome_count": len(missing),
        "missing_case_ids": tuple(item.key.case_id for item in missing),
        "demonstrated_count": sum(item.demonstrated_performance is True for item in records),
        "recorded_response_count": len(responses),
        "recorded_response_root_hash": response_root,
        "criterion_record_root_hash": canonical_sha256(
            tuple(item.record_hash for item in ordered_records)
        ),
        "audit_code_revision": audit_code_revision,
        "audit_pixi_lock_hash": file_sha256(pixi_lock_path.read_bytes()),
        "checks_passed": checks,
        "audited_at_utc": audited_at_utc,
        "gate_passed": True,
    }
    draft = ExternalCriterionIntegrityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalCriterionIntegrityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_path, report)
    return report


def _audit_sources(
    *,
    protocol: ExternalModelExecutionProtocol,
    accounting: ExternalDecisionAccountingReport,
    plan: ExternalCriterionPlan,
    report: ExternalCriterionReport,
    public: PublicBenchmarkManifest,
    evaluator: EvaluatorBenchmarkManifest,
    system_prompt_path: Path,
    pixi_lock_path: Path,
) -> None:
    try:
        prompt_hash = file_sha256(system_prompt_path.read_bytes())
        pixi_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ExternalCriterionAuditError("Could not read criterion provenance input") from error
    _require_equal("run identity", plan.run_id, report.run_id, accounting.run_id)
    _require_equal("protocol", plan.protocol_hash, protocol.protocol_hash, accounting.protocol_hash)
    _require_equal("accounting report", plan.accounting_report_hash, accounting.report_hash)
    _require_equal("source manifest", plan.source_manifest_hash, public.source_manifest_hash)
    _require_equal("public projection", plan.public_projection_hash, public.projection_hash)
    _require_equal(
        "evaluator projection", plan.evaluator_projection_hash, evaluator.projection_hash
    )
    _require_equal(
        "system prompt",
        plan.system_prompt_sha256,
        protocol.system_prompt_sha256,
        prompt_hash,
    )
    _require_equal("Pixi lock", plan.pixi_lock_hash, pixi_hash)
    _require_equal("criterion plan", report.criterion_plan_hash, plan.plan_hash)
    if not accounting.gate_passed or plan.created_at_utc <= accounting.audited_at_utc:
        raise ExternalCriterionAuditError(
            "Criterion phase did not follow the clean accounting audit"
        )
    if plan.prior_request_count + plan.expected_case_count > plan.request_budget:
        raise ExternalCriterionAuditError("Criterion phase exceeded the frozen request budget")
    expected_route = ModelRoute(provider=protocol.provider_id, model=protocol.requested_model_id)
    if plan.model_route != expected_route:
        raise ExternalCriterionAuditError("Criterion plan uses another model route")


def audit_external_criterion_rows(
    *,
    plan: ExternalCriterionPlan,
    accounting: ExternalDecisionAccountingReport,
    evaluator: EvaluatorBenchmarkManifest,
    benchmark_root: Path,
    complete_statuses: tuple[SampleDecisionStatusRecord, ...],
    attempts: tuple[ExternalCriterionRequestArtifact, ...],
    records: tuple[CriterionRecord, ...],
    responses: tuple[RecordedGenerationResponse, ...],
) -> None:
    complete_keys = tuple(item.key for item in complete_statuses)
    expected_case_ids = tuple(key.case_id for key in complete_keys)
    attempt_by_case = {item.request.case_id: item for item in attempts}
    record_by_case = {item.key.case_id: item for item in records}
    if len(attempt_by_case) != len(attempts):
        raise ExternalCriterionAuditError("Duplicate criterion attempt identity")
    if len(record_by_case) != len(records):
        raise ExternalCriterionAuditError("Duplicate criterion record identity")
    criteria = {item.case_id: item for item in evaluator.criteria}
    if (
        tuple(sorted(attempt_by_case)) != tuple(sorted(expected_case_ids))
        or tuple(sorted(record_by_case)) != tuple(sorted(expected_case_ids))
        or set(criteria) != set(expected_case_ids)
    ):
        raise ExternalCriterionAuditError("Criterion artifacts cover another case set")
    response_by_hash = {item.response_hash: item for item in responses}
    if len(response_by_hash) != len(responses):
        raise ExternalCriterionAuditError("Recorded criterion responses contain duplicates")

    for ordinal, status in enumerate(complete_statuses):
        key = status.key
        case_id = key.case_id
        sample_id = key.sample_id
        artifact = attempt_by_case[case_id]
        record = record_by_case[case_id]
        request = artifact.request
        criterion = criteria[case_id]
        payload = request.task_payload
        if not isinstance(payload, CriterionTaskPayload):
            raise ExternalCriterionAuditError("Criterion request contains another channel payload")
        try:
            prompt_text = (benchmark_root / criterion.criterion_prompt_ref).read_text(
                encoding="utf-8"
            )
            test_hash = file_sha256((benchmark_root / criterion.test_bundle_ref).read_bytes())
            rubric_hash = file_sha256((benchmark_root / criterion.rubric_ref).read_bytes())
        except OSError as error:
            raise ExternalCriterionAuditError(
                f"Could not verify authored criterion files for {case_id}"
            ) from error
        expected_seed = (plan.root_seed + ordinal) % (2**32)
        if (
            request.channel is not GenerationChannel.CRITERION
            or request.run_id != plan.run_id
            or request.case_id != case_id
            or request.sample_id != sample_id
            or request.model_route != plan.model_route
            or request.system_prompt_version != plan.system_prompt_version
            or request.system_prompt_sha256 != plan.system_prompt_sha256
            or request.sampling.temperature != plan.temperature
            or request.sampling.max_output_tokens != plan.max_output_tokens
            or request.sampling.seed != expected_seed
            or payload.criterion_probe_id != criterion.criterion_probe_id
            or payload.criterion_probe != prompt_text
        ):
            raise ExternalCriterionAuditError(
                f"Criterion request differs from the frozen plan: {case_id}"
            )
        if artifact.attempt.requested_at_utc < plan.created_at_utc:
            raise ExternalCriterionAuditError("Criterion request predates its post-audit plan")
        if record.key != key or record.request_hash != request.request_hash:
            raise ExternalCriterionAuditError(f"Criterion record identity differs: {case_id}")
        if (
            record.global_seal_hash != plan.global_seal_hash
            or record.condition_set_hash != status.condition_set_hash
            or record.criterion_probe_id != criterion.criterion_probe_id
            or record.test_bundle_sha256 != criterion.test_bundle_sha256
            or record.test_bundle_sha256 != test_hash
            or record.rubric_sha256 != criterion.rubric_sha256
            or record.rubric_sha256 != rubric_hash
            or record.revealed_at_utc < artifact.attempt.terminal_at_utc
        ):
            raise ExternalCriterionAuditError(f"Criterion label lineage differs: {case_id}")
        if artifact.response is None:
            if (
                record.response_hash is not None
                or record.execution.status is not CriterionExecutionStatus.PROVIDER_UNAVAILABLE
            ):
                raise ExternalCriterionAuditError("Provider failure was not retained as missing")
        else:
            response = artifact.response
            metadata = response.provider_metadata
            if (
                response.original_source is not OriginalGenerationSource.CEREBRAS
                or response_by_hash.get(response.response_hash) != response
                or record.response_hash != response.response_hash
                or metadata.provider_id != plan.model_route.provider
                or metadata.model_id != plan.model_route.model
                or metadata.resolved_provider_id != plan.model_route.provider
                or metadata.resolved_model_id != plan.model_route.model
            ):
                raise ExternalCriterionAuditError(
                    f"Criterion provider identity differs from the route: {case_id}"
                )
        if record.execution.status is CriterionExecutionStatus.COMPLETED:
            expected = record.execution.failed == 0
            if record.demonstrated_performance is not expected or record.missing_reason is not None:
                raise ExternalCriterionAuditError(
                    f"Criterion outcome was derived incorrectly: {case_id}"
                )
        elif record.demonstrated_performance is not None or record.missing_reason is None:
            raise ExternalCriterionAuditError(
                f"Criterion missingness was derived incorrectly: {case_id}"
            )
    if len(responses) != sum(item.response is not None for item in attempts):
        raise ExternalCriterionAuditError("Recorded responses differ from successful attempts")
    if plan.accounting_report_hash != accounting.report_hash:
        raise ExternalCriterionAuditError("Criterion artifacts lost pre-criterion audit lineage")


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ExternalCriterionAuditError(f"Could not verify audit source: {path}") from error


def _load_jsonl(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExternalCriterionAuditError(f"Could not read criterion responses: {path}") from error
    try:
        return tuple(
            RecordedGenerationResponse.model_validate_json(line) for line in lines if line.strip()
        )
    except ValidationError as error:
        raise ExternalCriterionAuditError("Recorded criterion response is invalid") from error


def _require_equal(name: str, *values: object) -> None:
    if not values or any(value != values[0] for value in values[1:]):
        raise ExternalCriterionAuditError(f"Criterion audit has mismatched {name}")


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Criterion audit timestamp must be UTC")
