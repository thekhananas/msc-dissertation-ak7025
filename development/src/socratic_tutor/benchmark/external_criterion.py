"""Post-seal external criterion generation and isolated execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from socratic_tutor.benchmark.artifacts import write_immutable_json, write_immutable_jsonl
from socratic_tutor.benchmark.cerebras import (
    CerebrasGateway,
    CerebrasGatewayConfig,
    CerebrasGatewayError,
    CerebrasRequestError,
    CerebrasResponseFormatError,
    GatewayGenerationResult,
    HttpxCerebrasTransport,
)
from socratic_tutor.benchmark.evaluator.criterion import (
    CriterionExecutionStatus,
    CriterionRecord,
    CriterionRecordFactory,
    CriterionRecordIndex,
    NormalizedCriterionExecution,
    create_normalized_criterion_execution,
)
from socratic_tutor.benchmark.evaluator.datasets import publish_criterion_records
from socratic_tutor.benchmark.evaluator.gate import CriterionGate, VerifiedCriterionAccess
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import EvaluatorBenchmarkManifest
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.requests import CriterionChannelRequestBuilder
from socratic_tutor.benchmark.external_audit import ExternalDecisionAccountingReport
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    load_external_model_execution_protocol,
)
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunPlan,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionStatus,
)
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel
from socratic_tutor.sandbox import (
    ModalSandboxExecutor,
    SandboxExecutor,
    execute_authored_python_tests,
)


class ExternalCriterionError(ValueError):
    """The post-seal criterion phase violated its frozen boundary."""


class ExternalCriterionPlan(ContractModel):
    """Criterion scope fixed after decision audit and before evaluator access."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_criterion_plan.v1"] = (
        "benchmark.external_criterion_plan.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounting_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_seal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_case_count: int = Field(ge=1)
    model_route: ModelRoute
    system_prompt_version: str = Field(min_length=1)
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)
    root_seed: int = Field(ge=0, le=2**32 - 1)
    request_spacing_seconds: float = Field(ge=0.0, le=120.0)
    prior_request_count: int = Field(ge=0)
    request_budget: int = Field(ge=1)
    code_revision: str = Field(min_length=1)
    pixi_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: datetime
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_plan(self) -> ExternalCriterionPlan:
        _require_utc(self.created_at_utc, "Criterion plan time")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("External criterion-plan hash does not match its content")
        if self.prior_request_count + self.expected_case_count > self.request_budget:
            raise ValueError("External criterion plan exceeds the frozen request budget")
        return self


class ExternalCriterionAttempt(ContractModel):
    """One terminal provider outcome for an exact criterion request."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_criterion_attempt.v1"] = (
        "benchmark.external_criterion_attempt.v1"
    )
    case_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["success", "provider_failure"]
    response_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attempt_count: int | None = Field(default=None, ge=1)
    retried_status_codes: tuple[int, ...] = ()
    backend_fingerprint: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    provider_request_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_http_status: int | None = Field(default=None, ge=100, le=599)
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    provider_finish_reason: str | None = None
    provider_message_fields: tuple[str, ...] | None = None
    requested_at_utc: datetime
    terminal_at_utc: datetime
    attempt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_attempt(self) -> ExternalCriterionAttempt:
        _require_utc(self.requested_at_utc, "Criterion request time")
        _require_utc(self.terminal_at_utc, "Criterion terminal time")
        if self.terminal_at_utc < self.requested_at_utc:
            raise ValueError("Criterion terminal time cannot precede its request")
        if self.status == "success":
            if self.response_hash is None or self.attempt_count is None:
                raise ValueError("Successful criterion attempt requires response metadata")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful criterion attempt cannot contain an error")
        elif self.response_hash is not None or not self.error_type or not self.error_message:
            raise ValueError("Failed criterion attempt requires an error and no response")
        if self.attempt_hash != model_content_hash(self, exclude={"attempt_hash"}):
            raise ValueError("Criterion attempt hash does not match its content")
        return self


class ExternalCriterionRequestArtifact(ContractModel):
    """Durable resume boundary for one criterion provider call."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_criterion_request_artifact.v1"] = (
        "benchmark.external_criterion_request_artifact.v1"
    )
    request: StudentGenerationRequest
    attempt: ExternalCriterionAttempt
    response: RecordedGenerationResponse | None = None
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact(self) -> ExternalCriterionRequestArtifact:
        if self.request.channel.value != "criterion":
            raise ValueError("External criterion artifact requires a criterion request")
        if (
            self.attempt.request_hash != self.request.request_hash
            or self.attempt.case_id != self.request.case_id
        ):
            raise ValueError("Criterion attempt belongs to another request")
        if self.response is None:
            if self.attempt.status != "provider_failure":
                raise ValueError("Only provider failure may omit a criterion response")
        elif (
            self.attempt.status != "success"
            or self.response.request != self.request
            or self.response.response_hash != self.attempt.response_hash
        ):
            raise ValueError("Criterion response differs from its request or attempt")
        if self.artifact_hash != model_content_hash(self, exclude={"artifact_hash"}):
            raise ValueError("Criterion request artifact hash does not match its content")
        return self


class ExternalCriterionReport(ContractModel):
    """Accounting summary published with the immutable criterion dataset."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_criterion_report.v1"] = (
        "benchmark.external_criterion_report.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    criterion_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounting_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_seal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    planned_case_count: int = Field(ge=1)
    provider_success_count: int = Field(ge=0)
    provider_missing_count: int = Field(ge=0)
    completed_execution_count: int = Field(ge=0)
    sandbox_missing_count: int = Field(ge=0)
    demonstrated_count: int = Field(ge=0)
    criterion_record_count: int = Field(ge=0)
    recorded_response_count: int = Field(ge=0)
    criterion_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_record_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at_utc: datetime
    gate_passed: Literal[True] = True
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> ExternalCriterionReport:
        _require_utc(self.completed_at_utc, "Criterion completion time")
        if self.planned_case_count != self.provider_success_count + self.provider_missing_count:
            raise ValueError("Criterion provider counts do not reconcile")
        if self.provider_success_count != (
            self.completed_execution_count + self.sandbox_missing_count
        ):
            raise ValueError("Criterion sandbox counts do not reconcile")
        if self.criterion_record_count != self.planned_case_count:
            raise ValueError("Criterion report must account for every eligible case")
        if self.recorded_response_count != self.provider_success_count:
            raise ValueError("Criterion response count differs from provider successes")
        if self.demonstrated_count > self.completed_execution_count:
            raise ValueError("Demonstrated count exceeds completed executions")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("External criterion report hash does not match its content")
        return self


class CriterionGateway(Protocol):
    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult: ...


class ExternalCriterionSecrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SOCRATIC_", extra="ignore")

    cerebras_api_key: SecretStr | None = None


type Clock = Callable[[], datetime]
type Sleeper = Callable[[float], Awaitable[None]]


def run_external_criterion_from_environment(
    *,
    protocol_path: Path,
    accounting_report_path: Path,
    manifest_path: Path,
    system_prompt_path: Path,
    seal_root: Path,
    pixi_lock_path: Path,
    code_revision: str,
    created_at_utc: datetime,
) -> ExternalCriterionReport:
    """Use the pinned Cerebras route and Modal after the audited decision barrier."""

    protocol = load_external_model_execution_protocol(protocol_path)
    accounting = _load_json(accounting_report_path, ExternalDecisionAccountingReport)
    authored = load_and_verify_manifest(manifest_path)
    public = project_public_manifest(authored, projected_at_utc=protocol.created_at_utc)
    evaluator = project_evaluator_manifest(
        authored,
        public,
        projected_at_utc=created_at_utc,
    )
    try:
        prompt_bytes = system_prompt_path.read_bytes()
        pixi_bytes = pixi_lock_path.read_bytes()
    except OSError as error:
        raise ExternalCriterionError("Could not read criterion provenance input") from error
    if file_sha256(prompt_bytes) != protocol.system_prompt_sha256:
        raise ExternalCriterionError("Criterion system prompt differs from the protocol")
    try:
        prompt = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExternalCriterionError("Criterion system prompt is not UTF-8") from error
    plan = _create_plan(
        protocol=protocol,
        accounting=accounting,
        public=public,
        evaluator=evaluator,
        code_revision=code_revision,
        pixi_lock_hash=file_sha256(pixi_bytes),
        created_at_utc=created_at_utc,
    )
    api_key = ExternalCriterionSecrets().cerebras_api_key
    if api_key is None:
        raise ExternalCriterionError(
            "SOCRATIC_CEREBRAS_API_KEY is required for external criterion generation"
        )
    client = httpx.AsyncClient()
    gateway = CerebrasGateway(
        config=CerebrasGatewayConfig(
            api_key=api_key,
            reasoning_effort=protocol.reasoning_effort or "low",
            timeout_seconds=protocol.timeout_seconds,
            max_attempts=protocol.max_attempts,
        ),
        transport=HttpxCerebrasTransport(client),
    )

    async def run() -> ExternalCriterionReport:
        async with client:
            return await run_external_criterion(
                plan=plan,
                public_manifest=public,
                evaluator_manifest=evaluator,
                benchmark_root=manifest_path.resolve().parent,
                seal_root=seal_root,
                system_prompt=prompt,
                gateway=gateway,
                executor=ModalSandboxExecutor(),
            )

    return asyncio.run(run())


async def run_external_criterion(
    *,
    plan: ExternalCriterionPlan,
    public_manifest: PublicBenchmarkManifest,
    evaluator_manifest: EvaluatorBenchmarkManifest,
    benchmark_root: Path,
    seal_root: Path,
    system_prompt: str,
    gateway: CriterionGateway,
    executor: SandboxExecutor,
    clock: Clock | None = None,
    sleeper: Sleeper = asyncio.sleep,
) -> ExternalCriterionReport:
    """Generate and execute one later criterion response per complete sealed case."""

    now = clock or (lambda: datetime.now(UTC))
    root = seal_root.resolve()
    report_path = root / "criterion" / "external_criterion_report.json"
    if report_path.exists():
        report = _load_json(report_path, ExternalCriterionReport)
        if report.criterion_plan_hash != plan.plan_hash or report.run_id != plan.run_id:
            raise ExternalCriterionError("Existing criterion report belongs to another plan")
        return report
    _validate_plan_sources(plan, public_manifest, evaluator_manifest, system_prompt)
    run_plan = _load_json(root / "run_plan.json", DecisionRunPlan)
    conditions = FilesystemConditionCommitStore(root / "decision")
    global_store = FilesystemGlobalDecisionSealStore(root / "decision", conditions)
    global_seal = global_store.load()
    if global_seal is None or global_seal.seal_hash != plan.global_seal_hash:
        raise ExternalCriterionError("Criterion plan does not match the global decision seal")
    complete_keys = tuple(
        status.key
        for status in global_seal.sample_statuses
        if status.status is SampleDecisionStatus.COMPLETE
    )
    if len(complete_keys) != plan.expected_case_count:
        raise ExternalCriterionError("Criterion plan case count differs from the global seal")
    write_immutable_json(root / "criterion" / "external_criterion_plan.json", plan)
    write_immutable_json(root / "criterion" / "evaluator_manifest.json", evaluator_manifest)

    existing_records = tuple(
        _load_json(path, CriterionRecord)
        for path in sorted((root / "criterion" / "records").glob("*.json"))
    )
    index = CriterionRecordIndex(existing_records)
    gate = CriterionGate(
        global_store,
        conditions,
        run_plan,
        criterion_record_exists=index.exists,
    )
    provider_artifacts: list[ExternalCriterionRequestArtifact] = []
    for ordinal, key in enumerate(complete_keys):
        existing = next((record for record in existing_records if record.key == key), None)
        if existing is not None:
            artifact_path = root / "criterion" / "attempts" / f"{existing.request_hash}.json"
            provider_artifacts.append(_load_json(artifact_path, ExternalCriterionRequestArtifact))
            continue
        access = gate.open(gate.issue(key), manifest_loader=lambda: evaluator_manifest)
        spec = GenerationRequestSpec(
            run_id=plan.run_id,
            sample_id=key.sample_id,
            system_prompt_version=plan.system_prompt_version,
            system_prompt=system_prompt,
            system_prompt_sha256=plan.system_prompt_sha256,
            model_route=plan.model_route,
            sampling=SamplingConfig(
                temperature=plan.temperature,
                max_output_tokens=plan.max_output_tokens,
                seed=(plan.root_seed + ordinal) % (2**32),
            ),
        )
        request = CriterionChannelRequestBuilder(benchmark_root, access).build_criterion(
            case_id=key.case_id,
            spec=spec,
        )
        artifact_path = root / "criterion" / "attempts" / f"{request.request_hash}.json"
        artifact = (
            _load_json(artifact_path, ExternalCriterionRequestArtifact)
            if artifact_path.exists()
            else None
        )
        if artifact is None:
            artifact = await _generate(gateway, request=request, clock=now)
            write_immutable_json(artifact_path, artifact)
            if ordinal < len(complete_keys) - 1 and plan.request_spacing_seconds > 0:
                await sleeper(plan.request_spacing_seconds)
        elif artifact.request != request:
            raise ExternalCriterionError("Stored criterion attempt belongs to another request")
        provider_artifacts.append(artifact)
        execution = _criterion_execution(
            artifact=artifact,
            access=access,
            benchmark_root=benchmark_root,
            executor=executor,
            clock=now,
        )
        record = CriterionRecordFactory().build(
            access,
            request=request,
            response=artifact.response,
            execution=execution,
            revealed_at_utc=_strictly_after(now(), execution.requested_at_utc),
        )
        index.add(record)
        write_immutable_json(root / "criterion" / "records" / f"{record.record_hash}.json", record)

    records = index.records()
    if tuple(record.key for record in records) != complete_keys:
        raise ExternalCriterionError("Criterion records do not cover every complete sealed case")
    responses = tuple(
        artifact.response for artifact in provider_artifacts if artifact.response is not None
    )
    write_immutable_jsonl(root / "criterion" / "recorded_responses.jsonl", responses)
    publication = publish_criterion_records(
        AtomicParquetDatasetStore(root / "datasets"),
        records,
        global_seal=global_seal,
        published_at_utc=_strictly_after(now(), max(record.revealed_at_utc for record in records)),
    )
    completed = sum(
        record.execution.status is CriterionExecutionStatus.COMPLETED for record in records
    )
    provider_missing = sum(
        record.execution.status is CriterionExecutionStatus.PROVIDER_UNAVAILABLE
        for record in records
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_report.v1",
        "run_id": plan.run_id,
        "benchmark_version": "v1",
        "criterion_plan_hash": plan.plan_hash,
        "accounting_report_hash": plan.accounting_report_hash,
        "global_seal_hash": global_seal.seal_hash,
        "planned_case_count": len(complete_keys),
        "provider_success_count": len(responses),
        "provider_missing_count": provider_missing,
        "completed_execution_count": completed,
        "sandbox_missing_count": len(responses) - completed,
        "demonstrated_count": sum(record.demonstrated_performance is True for record in records),
        "criterion_record_count": len(records),
        "recorded_response_count": len(responses),
        "criterion_publication_hash": publication.publication_hash,
        "criterion_record_root_hash": canonical_sha256(
            tuple(record.record_hash for record in records)
        ),
        "completed_at_utc": _strictly_after(now(), publication.published_at_utc),
        "gate_passed": True,
    }
    draft = ExternalCriterionReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalCriterionReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _create_plan(
    *,
    protocol: ExternalModelExecutionProtocol,
    accounting: ExternalDecisionAccountingReport,
    public: PublicBenchmarkManifest,
    evaluator: EvaluatorBenchmarkManifest,
    code_revision: str,
    pixi_lock_hash: str,
    created_at_utc: datetime,
) -> ExternalCriterionPlan:
    if not accounting.gate_passed or accounting.criterion_artifact_count != 0:
        raise ExternalCriterionError("Criterion requires a clean pre-criterion audit")
    if (
        accounting.protocol_hash != protocol.protocol_hash
        or accounting.source_manifest_hash != public.source_manifest_hash
        or accounting.public_projection_hash != public.projection_hash
    ):
        raise ExternalCriterionError("Criterion inputs differ from the accounting report")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_plan.v1",
        "run_id": accounting.run_id,
        "benchmark_version": "v1",
        "protocol_hash": protocol.protocol_hash,
        "accounting_report_hash": accounting.report_hash,
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "evaluator_projection_hash": evaluator.projection_hash,
        "global_seal_hash": accounting.global_seal_hash,
        "expected_case_count": accounting.complete_case_count,
        "model_route": ModelRoute(
            provider=protocol.provider_id,
            model=protocol.requested_model_id,
        ),
        "system_prompt_version": protocol.system_prompt_version,
        "system_prompt_sha256": protocol.system_prompt_sha256,
        "temperature": protocol.temperature,
        "max_output_tokens": protocol.max_output_tokens,
        "root_seed": protocol.run_seed,
        "request_spacing_seconds": protocol.request_spacing_seconds,
        "prior_request_count": accounting.recorded_request_count,
        "request_budget": protocol.request_budget,
        "code_revision": code_revision,
        "pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ExternalCriterionPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ExternalCriterionPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


async def _generate(
    gateway: CriterionGateway,
    *,
    request: StudentGenerationRequest,
    clock: Clock,
) -> ExternalCriterionRequestArtifact:
    requested_at = _require_utc(clock(), "Criterion request time")
    try:
        result = await gateway.generate(request)
    except CerebrasGatewayError as error:
        content: dict[str, object] = {
            "schema_version": 1,
            "schema_id": "benchmark.external_criterion_attempt.v1",
            "case_id": request.case_id,
            "request_hash": request.request_hash,
            "status": "provider_failure",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "requested_at_utc": requested_at,
            "terminal_at_utc": _strictly_after(clock(), requested_at),
            **_failure_details(error),
        }
        return _request_artifact(request, _attempt(content), None)
    metadata = result.response.provider_metadata
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_attempt.v1",
        "case_id": request.case_id,
        "request_hash": request.request_hash,
        "status": "success",
        "response_hash": result.response.response_hash,
        "attempt_count": result.attempt_count,
        "retried_status_codes": result.retried_status_codes,
        "backend_fingerprint": metadata.backend_fingerprint,
        "latency_ms": metadata.latency_ms,
        "provider_request_id": metadata.provider_request_id,
        "requested_at_utc": requested_at,
        "terminal_at_utc": _strictly_after(clock(), requested_at),
    }
    return _request_artifact(request, _attempt(content), result.response)


def _criterion_execution(
    *,
    artifact: ExternalCriterionRequestArtifact,
    access: VerifiedCriterionAccess,
    benchmark_root: Path,
    executor: SandboxExecutor,
    clock: Clock,
) -> NormalizedCriterionExecution:
    criterion = next(
        item for item in access.manifest.criteria if item.case_id == artifact.request.case_id
    )
    requested_at = artifact.attempt.requested_at_utc
    if artifact.response is None:
        return create_normalized_criterion_execution(
            status=CriterionExecutionStatus.PROVIDER_UNAVAILABLE,
            test_bundle_sha256=criterion.test_bundle_sha256,
            requested_at_utc=requested_at,
        )
    result = execute_authored_python_tests(
        executor,
        response=artifact.response.final_response,
        bundle_path=benchmark_root / criterion.test_bundle_ref,
    )
    executed_at = _strictly_after(clock(), requested_at)
    execution = result.execution
    stdout_hash = canonical_sha256({"stream": "stdout", "value": execution.stdout})
    stderr_hash = canonical_sha256({"stream": "stderr", "value": execution.stderr})
    operation_id = execution.sandbox_id or f"criterion-{artifact.request.request_hash}"
    if result.outcome is not None and result.outcome.status == "completed":
        return create_normalized_criterion_execution(
            status=CriterionExecutionStatus.COMPLETED,
            test_bundle_sha256=criterion.test_bundle_sha256,
            requested_at_utc=requested_at,
            operation_id=operation_id,
            passed=result.outcome.passed,
            failed=result.outcome.failed,
            exit_code=execution.exit_code if execution.exit_code is not None else 0,
            stdout_sha256=stdout_hash,
            stderr_sha256=stderr_hash,
            executed_at_utc=executed_at,
        )
    error_name = (execution.error_type or "").lower()
    if result.outcome is None and execution.sandbox_id is None:
        return create_normalized_criterion_execution(
            status=CriterionExecutionStatus.SANDBOX_UNAVAILABLE,
            test_bundle_sha256=criterion.test_bundle_sha256,
            requested_at_utc=requested_at,
        )
    status = CriterionExecutionStatus.SANDBOX_ERROR
    timed_out = False
    resource_limited = False
    if "timeout" in error_name:
        status = CriterionExecutionStatus.SANDBOX_TIMEOUT
        timed_out = True
    elif "memory" in error_name or "resource" in error_name:
        status = CriterionExecutionStatus.SANDBOX_RESOURCE_LIMIT
        resource_limited = True
    return create_normalized_criterion_execution(
        status=status,
        test_bundle_sha256=criterion.test_bundle_sha256,
        requested_at_utc=requested_at,
        operation_id=operation_id,
        exit_code=execution.exit_code,
        timed_out=timed_out,
        resource_limited=resource_limited,
        stdout_sha256=stdout_hash,
        stderr_sha256=stderr_hash,
        executed_at_utc=executed_at,
    )


def _validate_plan_sources(
    plan: ExternalCriterionPlan,
    public: PublicBenchmarkManifest,
    evaluator: EvaluatorBenchmarkManifest,
    prompt: str,
) -> None:
    if (
        plan.source_manifest_hash != public.source_manifest_hash
        or plan.public_projection_hash != public.projection_hash
        or plan.evaluator_projection_hash != evaluator.projection_hash
        or plan.expected_case_count != len(public.cases)
        or file_sha256(prompt.encode("utf-8")) != plan.system_prompt_sha256
    ):
        raise ExternalCriterionError("Criterion plan differs from its frozen sources")


def _attempt(content: dict[str, object]) -> ExternalCriterionAttempt:
    draft = ExternalCriterionAttempt.model_construct(
        _fields_set=set(content), **content, attempt_hash="0" * 64
    )
    return ExternalCriterionAttempt.model_validate(
        {**content, "attempt_hash": model_content_hash(draft, exclude={"attempt_hash"})}
    )


def _request_artifact(
    request: StudentGenerationRequest,
    attempt: ExternalCriterionAttempt,
    response: RecordedGenerationResponse | None,
) -> ExternalCriterionRequestArtifact:
    content = {"request": request, "attempt": attempt, "response": response}
    draft = ExternalCriterionRequestArtifact.model_construct(
        _fields_set=set(content), **content, artifact_hash="0" * 64
    )
    return ExternalCriterionRequestArtifact.model_validate(
        {**content, "artifact_hash": model_content_hash(draft, exclude={"artifact_hash"})}
    )


def _failure_details(error: CerebrasGatewayError) -> dict[str, object]:
    if isinstance(error, CerebrasRequestError):
        return {
            "error_http_status": error.status_code,
            "provider_request_id": error.request_id,
            "provider_error_code": error.provider_error_code,
            "provider_error_message": error.provider_error_message,
        }
    if isinstance(error, CerebrasResponseFormatError):
        return {
            "error_http_status": error.status_code,
            "latency_ms": error.latency_ms,
            "provider_request_id": error.request_id,
            "backend_fingerprint": error.backend_fingerprint,
            "provider_finish_reason": error.finish_reason,
            "provider_message_fields": error.message_fields,
        }
    return {}


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ExternalCriterionError(f"Could not verify criterion artifact: {path}") from error


def _strictly_after(value: datetime, boundary: datetime) -> datetime:
    checked = _require_utc(value, "Criterion event time")
    if checked <= boundary:
        from datetime import timedelta

        return boundary + timedelta(microseconds=1)
    return checked


def _require_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value
