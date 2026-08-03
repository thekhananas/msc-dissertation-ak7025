"""Resumable held-out generation before public ratings and criterion access."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
import yaml
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
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
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.evidence_specificity import ExternalRunPreflight
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    validate_external_model_execution_protocol,
)
from socratic_tutor.benchmark.generation import (
    GenerationChannel,
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.requests import PublicChannelRequestBuilder
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel


class ExternalDecisionGenerationPlan(ContractModel):
    """Frozen scope for the pre-criterion held-out provider calls."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_generation_plan.v1"] = (
        "benchmark.external_decision_generation_plan.v1"
    )
    purpose: Literal["heldout_external_decision_generation"]
    preflight_hash: Sha256
    required_channels: tuple[GenerationChannel, GenerationChannel]
    expected_case_count: int = Field(ge=1)
    expected_request_count: int = Field(ge=2)
    require_backend_fingerprint: bool = True

    @model_validator(mode="after")
    def validate_scope(self) -> ExternalDecisionGenerationPlan:
        if self.required_channels != (GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE):
            raise ValueError("Decision generation must contain public then evidence channels only")
        if self.expected_request_count != self.expected_case_count * 2:
            raise ValueError("Decision generation requires exactly two requests per case")
        return self


class ExternalDecisionAttempt(ContractModel):
    """Terminal outcome for one exact content-addressed provider request."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_attempt.v1"] = (
        "benchmark.external_decision_attempt.v1"
    )
    case_id: str = Field(min_length=1)
    channel: Literal[GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE]
    request_hash: Sha256
    status: Literal["success", "provider_failure"]
    response_hash: Sha256 | None = None
    attempt_count: int | None = Field(default=None, ge=1)
    retried_status_codes: tuple[int, ...] = ()
    resolved_provider_id: str | None = None
    resolved_model_id: str | None = None
    backend_fingerprint: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    error_http_status: int | None = Field(default=None, ge=100, le=599)
    provider_request_id: str | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    provider_finish_reason: str | None = None
    provider_message_fields: tuple[str, ...] | None = None
    terminal_at_utc: datetime
    attempt_hash: Sha256

    @model_validator(mode="after")
    def validate_terminal_outcome(self) -> ExternalDecisionAttempt:
        if self.terminal_at_utc.tzinfo is None or self.terminal_at_utc.utcoffset() != UTC.utcoffset(
            self.terminal_at_utc
        ):
            raise ValueError("Decision-attempt timestamp must be UTC")
        if self.status == "success":
            if self.response_hash is None or self.attempt_count is None:
                raise ValueError("Successful decision attempts require response and retry metadata")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful decision attempts cannot contain provider errors")
        elif self.response_hash is not None or not self.error_type or not self.error_message:
            raise ValueError("Failed decision attempts require an error and no response")
        if self.attempt_hash != model_content_hash(self, exclude={"attempt_hash"}):
            raise ValueError("Decision-attempt hash does not match its content")
        return self


class ExternalDecisionRequestArtifact(ContractModel):
    """One durable resume boundary written immediately after a terminal outcome."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_request_artifact.v1"] = (
        "benchmark.external_decision_request_artifact.v1"
    )
    request: StudentGenerationRequest
    attempt: ExternalDecisionAttempt
    response: RecordedGenerationResponse | None = None
    artifact_hash: Sha256

    @model_validator(mode="after")
    def validate_artifact(self) -> ExternalDecisionRequestArtifact:
        if self.request.channel is GenerationChannel.CRITERION:
            raise ValueError("Criterion requests are forbidden before the decision seal")
        if self.attempt.request_hash != self.request.request_hash:
            raise ValueError("Attempt belongs to another request")
        if self.attempt.channel is not self.request.channel:
            raise ValueError("Attempt channel does not match request channel")
        if self.response is None:
            if self.attempt.status != "provider_failure":
                raise ValueError("Only provider failures may omit a recorded response")
        elif (
            self.attempt.status != "success"
            or self.response.request != self.request
            or self.response.response_hash != self.attempt.response_hash
        ):
            raise ValueError("Recorded response does not match the successful attempt")
        if self.artifact_hash != model_content_hash(self, exclude={"artifact_hash"}):
            raise ValueError("Decision request artifact hash does not match its content")
        return self


class ExternalDecisionGenerationReport(ContractModel):
    """Audit summary for all planned public and evidence requests."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_generation_report.v1"] = (
        "benchmark.external_decision_generation_report.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    protocol_hash: Sha256
    preflight_hash: Sha256
    generation_plan_hash: Sha256
    source_manifest_hash: Sha256
    public_projection_hash: Sha256
    generated_at_utc: datetime
    case_count: int = Field(ge=1)
    request_count: int = Field(ge=1)
    terminal_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    provider_failure_count: int = Field(ge=0)
    eligible_case_count: int = Field(ge=0)
    precriterion_missing_case_count: int = Field(ge=0)
    generation_complete: bool
    gate_passed: bool
    attempts: tuple[ExternalDecisionAttempt, ...]
    recorded_responses_hash: Sha256
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> ExternalDecisionGenerationReport:
        if (
            self.generated_at_utc.tzinfo is None
            or self.generated_at_utc.utcoffset() != UTC.utcoffset(self.generated_at_utc)
        ):
            raise ValueError("Decision-generation report timestamp must be UTC")
        if self.terminal_count != len(self.attempts):
            raise ValueError("Terminal count does not match attempts")
        if self.success_count + self.provider_failure_count != self.terminal_count:
            raise ValueError("Decision-generation outcome counts do not reconcile")
        if self.case_count != self.eligible_case_count + self.precriterion_missing_case_count:
            raise ValueError("Decision-generation case counts do not reconcile")
        if self.generation_complete != (self.terminal_count == self.request_count):
            raise ValueError("Generation-complete flag does not match terminal requests")
        if self.gate_passed and (
            not self.generation_complete or self.success_count != self.request_count
        ):
            raise ValueError("Decision-generation gate cannot pass with missing responses")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Decision-generation report hash does not match its content")
        return self


class ExternalDecisionGateway(Protocol):
    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult: ...


class CerebrasDecisionSecrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SOCRATIC_", extra="ignore")

    cerebras_api_key: SecretStr | None = None


type Clock = Callable[[], datetime]
type Sleeper = Callable[[float], Awaitable[None]]


def load_external_decision_generation_plan(path: Path) -> ExternalDecisionGenerationPlan:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read external decision-generation plan: {path}") from error
    return ExternalDecisionGenerationPlan.model_validate(raw)


async def run_external_decision_generation(
    *,
    protocol: ExternalModelExecutionProtocol,
    preflight: ExternalRunPreflight,
    plan: ExternalDecisionGenerationPlan,
    manifest_path: Path,
    analysis_specification_path: Path,
    system_prompt_path: Path,
    output_root: Path,
    run_id: str,
    gateway: ExternalDecisionGateway,
    clock: Clock | None = None,
    sleeper: Sleeper = asyncio.sleep,
) -> ExternalDecisionGenerationReport:
    """Generate only decision-phase channels and resume from terminal request artifacts."""

    now = clock or (lambda: datetime.now(UTC))
    root = output_root.resolve()
    manifest = load_and_verify_manifest(manifest_path)
    public = project_public_manifest(manifest, projected_at_utc=protocol.created_at_utc)
    analysis = load_analysis_specification(analysis_specification_path)
    validate_external_model_execution_protocol(
        protocol,
        manifest=manifest,
        public_manifest=public,
        analysis_specification=analysis,
    )
    _validate_preflight(
        preflight,
        protocol=protocol,
        analysis_hash=analysis_specification_hash(analysis),
    )
    _validate_plan(plan, preflight=preflight, case_count=len(public.cases), protocol=protocol)
    prompt = _load_system_prompt(system_prompt_path, protocol=protocol)

    existing_report = _load_completed_report(root / "generation_report.json", run_id=run_id)
    if existing_report is not None:
        _validate_report_identity(
            existing_report, protocol=protocol, preflight=preflight, plan=plan
        )
        if (
            existing_report.source_manifest_hash != public.source_manifest_hash
            or existing_report.public_projection_hash != public.projection_hash
        ):
            raise ValueError("Existing decision-generation report belongs to another manifest")
        return existing_report

    write_immutable_json(root / "generation_plan.json", plan)
    write_immutable_json(root / "external_model_protocol.json", protocol)
    write_immutable_json(root / "external_run_preflight.json", preflight)
    write_immutable_json(root / "public_manifest.json", public)

    builder = PublicChannelRequestBuilder(manifest_path.resolve().parent, public)
    route = ModelRoute(provider=protocol.provider_id, model=protocol.requested_model_id)
    requests: list[StudentGenerationRequest] = []
    for ordinal, case in enumerate(public.cases):
        spec = GenerationRequestSpec(
            run_id=run_id,
            sample_id=f"{run_id}:{case.case_id}:001",
            system_prompt_version=protocol.system_prompt_version,
            system_prompt=prompt,
            system_prompt_sha256=protocol.system_prompt_sha256,
            model_route=route,
            sampling=SamplingConfig(
                temperature=protocol.temperature,
                max_output_tokens=protocol.max_output_tokens,
                seed=(protocol.run_seed + ordinal) % (2**32),
            ),
        )
        requests.extend(
            (
                builder.build_public(case_id=case.case_id, spec=spec),
                builder.build_evidence(case_id=case.case_id, spec=spec),
            )
        )

    artifacts: list[ExternalDecisionRequestArtifact] = []
    for index, request in enumerate(requests):
        artifact_path = root / "generation" / "attempts" / f"{request.request_hash}.json"
        artifact = _load_request_artifact(artifact_path, request=request)
        if artifact is None:
            artifact = await _generate_request(gateway, request=request, clock=now)
            write_immutable_json(artifact_path, artifact)
            if index < len(requests) - 1 and protocol.request_spacing_seconds > 0:
                await sleeper(protocol.request_spacing_seconds)
        artifacts.append(artifact)

    attempts = tuple(artifact.attempt for artifact in artifacts)
    records = tuple(artifact.response for artifact in artifacts if artifact.response is not None)
    success_by_case = {
        case.case_id: {
            artifact.request.channel
            for artifact in artifacts
            if artifact.request.case_id == case.case_id and artifact.attempt.status == "success"
        }
        for case in public.cases
    }
    eligible_count = sum(
        channels == {GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE}
        for channels in success_by_case.values()
    )
    gate_passed = len(records) == len(requests)
    if plan.require_backend_fingerprint:
        gate_passed = gate_passed and all(
            artifact.attempt.backend_fingerprint for artifact in artifacts
        )
    responses_hash = model_content_hash(_ResponseCollection(responses=records), exclude=set())
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_generation_report.v1",
        "run_id": run_id,
        "benchmark_version": "v1",
        "protocol_hash": protocol.protocol_hash,
        "preflight_hash": preflight.preflight_hash,
        "generation_plan_hash": model_content_hash(plan),
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "generated_at_utc": _require_utc(now()),
        "case_count": len(public.cases),
        "request_count": len(requests),
        "terminal_count": len(attempts),
        "success_count": len(records),
        "provider_failure_count": len(attempts) - len(records),
        "eligible_case_count": eligible_count,
        "precriterion_missing_case_count": len(public.cases) - eligible_count,
        "generation_complete": len(attempts) == len(requests),
        "gate_passed": gate_passed,
        "attempts": attempts,
        "recorded_responses_hash": responses_hash,
    }
    draft = ExternalDecisionGenerationReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalDecisionGenerationReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_jsonl(root / "generation_attempts.jsonl", attempts)
    write_immutable_jsonl(root / "recorded_responses.jsonl", records)
    write_immutable_json(root / "generation_report.json", report)
    return report


def run_external_decision_generation_from_environment(
    *,
    protocol: ExternalModelExecutionProtocol,
    preflight: ExternalRunPreflight,
    plan: ExternalDecisionGenerationPlan,
    manifest_path: Path,
    analysis_specification_path: Path,
    system_prompt_path: Path,
    output_root: Path,
    run_id: str,
) -> ExternalDecisionGenerationReport:
    """Run the explicit held-out generation using only the ignored local API key."""

    api_key = CerebrasDecisionSecrets().cerebras_api_key
    if api_key is None:
        raise ValueError(
            "SOCRATIC_CEREBRAS_API_KEY is required for held-out decision generation; "
            "store it only in ignored development/.env"
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

    async def run() -> ExternalDecisionGenerationReport:
        async with client:
            return await run_external_decision_generation(
                protocol=protocol,
                preflight=preflight,
                plan=plan,
                manifest_path=manifest_path,
                analysis_specification_path=analysis_specification_path,
                system_prompt_path=system_prompt_path,
                output_root=output_root,
                run_id=run_id,
                gateway=gateway,
            )

    return asyncio.run(run())


class _ResponseCollection(ContractModel):
    responses: tuple[RecordedGenerationResponse, ...]


async def _generate_request(
    gateway: ExternalDecisionGateway,
    *,
    request: StudentGenerationRequest,
    clock: Clock,
) -> ExternalDecisionRequestArtifact:
    terminal_at = _require_utc(clock())
    try:
        result = await gateway.generate(request)
    except CerebrasGatewayError as error:
        content = {
            "schema_version": 1,
            "schema_id": "benchmark.external_decision_attempt.v1",
            "case_id": request.case_id,
            "channel": request.channel,
            "request_hash": request.request_hash,
            "status": "provider_failure",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "terminal_at_utc": terminal_at,
            **_failure_details(error),
        }
        attempt = _attempt(content)
        return _request_artifact(request=request, attempt=attempt, response=None)
    metadata = result.response.provider_metadata
    attempt = _attempt(
        {
            "schema_version": 1,
            "schema_id": "benchmark.external_decision_attempt.v1",
            "case_id": request.case_id,
            "channel": request.channel,
            "request_hash": request.request_hash,
            "status": "success",
            "response_hash": result.response.response_hash,
            "attempt_count": result.attempt_count,
            "retried_status_codes": result.retried_status_codes,
            "resolved_provider_id": metadata.resolved_provider_id,
            "resolved_model_id": metadata.resolved_model_id,
            "backend_fingerprint": metadata.backend_fingerprint,
            "latency_ms": metadata.latency_ms,
            "terminal_at_utc": terminal_at,
        }
    )
    return _request_artifact(request=request, attempt=attempt, response=result.response)


def _attempt(content: dict[str, object]) -> ExternalDecisionAttempt:
    draft = ExternalDecisionAttempt.model_construct(
        _fields_set=set(content), **content, attempt_hash="0" * 64
    )
    return ExternalDecisionAttempt.model_validate(
        {**content, "attempt_hash": model_content_hash(draft, exclude={"attempt_hash"})}
    )


def _request_artifact(
    *,
    request: StudentGenerationRequest,
    attempt: ExternalDecisionAttempt,
    response: RecordedGenerationResponse | None,
) -> ExternalDecisionRequestArtifact:
    content = {"request": request, "attempt": attempt, "response": response}
    draft = ExternalDecisionRequestArtifact.model_construct(
        _fields_set=set(content), **content, artifact_hash="0" * 64
    )
    return ExternalDecisionRequestArtifact.model_validate(
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


def _load_request_artifact(
    path: Path, *, request: StudentGenerationRequest
) -> ExternalDecisionRequestArtifact | None:
    if not path.exists():
        return None
    artifact = _load_json(path, ExternalDecisionRequestArtifact)
    if artifact.request != request:
        raise ValueError(f"Stored decision artifact belongs to another request: {path}")
    return artifact


def _load_completed_report(path: Path, *, run_id: str) -> ExternalDecisionGenerationReport | None:
    if not path.exists():
        return None
    report = _load_json(path, ExternalDecisionGenerationReport)
    if report.run_id != run_id:
        raise ValueError("Existing decision-generation report belongs to another run")
    return report


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read decision-generation artifact: {path}") from error
    return model.model_validate(raw)


def _validate_report_identity(
    report: ExternalDecisionGenerationReport,
    *,
    protocol: ExternalModelExecutionProtocol,
    preflight: ExternalRunPreflight,
    plan: ExternalDecisionGenerationPlan,
) -> None:
    if (
        report.protocol_hash != protocol.protocol_hash
        or report.preflight_hash != preflight.preflight_hash
        or report.generation_plan_hash != model_content_hash(plan)
    ):
        raise ValueError("Existing decision-generation report has another frozen identity")


def _validate_preflight(
    preflight: ExternalRunPreflight,
    *,
    protocol: ExternalModelExecutionProtocol,
    analysis_hash: str,
) -> None:
    if preflight.external_protocol_hash != protocol.protocol_hash:
        raise ValueError("External-run preflight belongs to another protocol")
    if preflight.analysis_specification_hash != analysis_hash:
        raise ValueError("External-run preflight belongs to another analysis specification")


def _validate_plan(
    plan: ExternalDecisionGenerationPlan,
    *,
    preflight: ExternalRunPreflight,
    case_count: int,
    protocol: ExternalModelExecutionProtocol,
) -> None:
    if plan.preflight_hash != preflight.preflight_hash:
        raise ValueError("Decision-generation plan belongs to another preflight")
    if plan.expected_case_count != case_count:
        raise ValueError("Decision-generation case count does not match the public manifest")
    if protocol.repeats_per_case != 1:
        raise ValueError("Decision-generation v1 requires exactly one repeat per case")
    if plan.expected_request_count > protocol.request_budget:
        raise ValueError("Decision-generation requests exceed the frozen provider budget")


def _load_system_prompt(path: Path, *, protocol: ExternalModelExecutionProtocol) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Could not read external-model system prompt: {path}") from error
    if file_sha256(content) != protocol.system_prompt_sha256:
        raise ValueError("External-model system prompt hash does not match the protocol")
    text = content.decode("utf-8")
    if not text.strip():
        raise ValueError("External-model system prompt is empty")
    return text


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Decision-generation clock must return a UTC-aware timestamp")
    return value
