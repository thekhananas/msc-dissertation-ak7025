"""Development-only full-channel rehearsal for a frozen external-model route."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import Field, SecretStr, model_validator
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
from socratic_tutor.benchmark.evaluator.gate import CriterionGate
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.requests import CriterionChannelRequestBuilder
from socratic_tutor.benchmark.external_protocol import ExternalModelExecutionProtocol
from socratic_tutor.benchmark.generation import (
    GenerationChannel,
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.global_seal import FilesystemGlobalDecisionSealStore
from socratic_tutor.benchmark.public.offline import (
    OfflineDecisionPlan,
    run_offline_decision_phase,
)
from socratic_tutor.benchmark.public.requests import PublicChannelRequestBuilder
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel


class ExternalRouteRehearsalPlan(ContractModel):
    """Inputs allowed for a non-held-out three-channel route rehearsal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_route_rehearsal_plan.v1"] = (
        "benchmark.external_route_rehearsal_plan.v1"
    )
    purpose: Literal["development_only_external_route_rehearsal"]
    development_manifest: str = Field(min_length=1)
    fixture_decision_plan: str = Field(min_length=1)
    permitted_case_ids: tuple[str, ...] = Field(min_length=1)
    require_backend_fingerprint: bool = True

    @model_validator(mode="after")
    def require_two_distinct_cases(self) -> ExternalRouteRehearsalPlan:
        if len(self.permitted_case_ids) != 2 or len(set(self.permitted_case_ids)) != 2:
            raise ValueError(
                "External route rehearsal requires exactly two distinct development cases"
            )
        return self


class ExternalRouteRehearsalAttempt(ContractModel):
    """One visible-channel request outcome, retaining enough facts to diagnose failure."""

    case_id: str = Field(min_length=1)
    channel: GenerationChannel
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["success", "provider_failure"]
    response_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
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

    @model_validator(mode="after")
    def validate_outcome(self) -> ExternalRouteRehearsalAttempt:
        if self.status == "success":
            if self.response_hash is None or self.resolved_model_id is None:
                raise ValueError(
                    "Successful rehearsal attempts require response and model identity"
                )
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful rehearsal attempts cannot contain an error")
        elif self.response_hash is not None:
            raise ValueError("Failed rehearsal attempts cannot contain a response hash")
        return self


class ExternalRouteRehearsalReport(ContractModel):
    """Immutable operational report for M6.3, never an empirical result."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_route_rehearsal_report.v1"] = (
        "benchmark.external_route_rehearsal_report.v1"
    )
    run_id: str = Field(min_length=1)
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rehearsal_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at_utc: datetime
    attempt_count: int = Field(ge=1)
    usable_count: int = Field(ge=0)
    attempts: tuple[ExternalRouteRehearsalAttempt, ...] = Field(min_length=1)
    gate_passed: bool
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> ExternalRouteRehearsalReport:
        timestamp_is_utc = (
            self.generated_at_utc.tzinfo is not None
            and self.generated_at_utc.utcoffset() == UTC.utcoffset(self.generated_at_utc)
        )
        if not timestamp_is_utc:
            raise ValueError("Route rehearsal timestamp must be UTC")
        if self.attempt_count != len(self.attempts):
            raise ValueError("Route rehearsal attempt count does not match attempts")
        if self.usable_count != sum(attempt.status == "success" for attempt in self.attempts):
            raise ValueError("Route rehearsal usable count does not match attempts")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Route rehearsal report hash does not match its content")
        return self


class RehearsalGateway(Protocol):
    """Small network seam for a full route rehearsal."""

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult: ...


type Sleeper = Callable[[float], Awaitable[None]]
type Clock = Callable[[], datetime]


class CerebrasExternalEvaluationSecrets(BaseSettings):
    """The ignored local secret used for development rehearsal and primary execution."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SOCRATIC_", extra="ignore")

    cerebras_api_key: SecretStr | None = None


def load_external_route_rehearsal_plan(path: Path) -> ExternalRouteRehearsalPlan:
    """Load one explicit development-only rehearsal plan."""

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read external route rehearsal plan: {path}") from error
    return ExternalRouteRehearsalPlan.model_validate(raw)


async def run_external_route_rehearsal(
    *,
    protocol: ExternalModelExecutionProtocol,
    plan: ExternalRouteRehearsalPlan,
    manifest_path: Path,
    fixture_decision_plan_path: Path,
    system_prompt_path: Path,
    output_root: Path,
    run_id: str,
    gateway: RehearsalGateway,
    clock: Clock | None = None,
    sleeper: Sleeper = asyncio.sleep,
) -> ExternalRouteRehearsalReport:
    """Call public, evidence, and gated criterion channels on two development cases only."""

    now = clock or (lambda: datetime.now(UTC))
    generated_at_utc = _require_utc(now())
    authored = load_and_verify_manifest(manifest_path)
    _validate_development_inputs(
        plan,
        manifest_path,
        authored.benchmark_version,
        authored.status.value,
    )
    public = project_public_manifest(authored, projected_at_utc=generated_at_utc)
    evaluator = project_evaluator_manifest(authored, public, projected_at_utc=generated_at_utc)
    root = output_root.resolve()
    existing = _load_existing_report(root, protocol, plan, public.source_manifest_hash, run_id)
    if existing is not None:
        return existing

    prompt = system_prompt_path.read_text(encoding="utf-8")
    if file_sha256(prompt.encode("utf-8")) != protocol.system_prompt_sha256:
        raise ValueError("Route rehearsal system prompt does not match the frozen protocol")
    fixture_plan = _fixture_plan(
        fixture_decision_plan_path,
        protocol=protocol,
        run_id=run_id,
        created_at_utc=generated_at_utc,
    )
    fixture_root = root / "fixture_decision"
    run_offline_decision_phase(
        manifest=public,
        benchmark_root=manifest_path.parent,
        plan=fixture_plan,
        output_root=fixture_root,
    )

    route = ModelRoute(provider=protocol.provider_id, model=protocol.requested_model_id)
    request_builder = PublicChannelRequestBuilder(manifest_path.parent, public)
    conditions = FilesystemConditionCommitStore(fixture_root / "decision")
    seal_store = FilesystemGlobalDecisionSealStore(fixture_root / "decision", conditions)
    run_plan = _load_run_plan(fixture_root / "run_plan.json")
    gate = CriterionGate(
        seal_store,
        conditions,
        run_plan,
        criterion_record_exists=lambda _: False,
    )
    sample_keys = {key.case_id: key for key in run_plan.planned_samples}
    records: list[RecordedGenerationResponse] = []
    attempts: list[ExternalRouteRehearsalAttempt] = []
    completed_requests = 0
    expected_request_count = len(plan.permitted_case_ids) * 3

    async def call(request: StudentGenerationRequest) -> None:
        nonlocal completed_requests
        attempt, record = await _generate_attempt(gateway, request)
        attempts.append(attempt)
        if record is not None:
            records.append(record)
        completed_requests += 1
        if completed_requests < expected_request_count and protocol.request_spacing_seconds > 0:
            await sleeper(protocol.request_spacing_seconds)

    for ordinal, case_id in enumerate(plan.permitted_case_ids):
        key = sample_keys.get(case_id)
        if key is None:
            raise ValueError(f"Fixture decision plan did not create a sample for {case_id}")
        spec = GenerationRequestSpec(
            run_id=run_id,
            sample_id=key.sample_id,
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
        await call(request_builder.build_public(case_id=case_id, spec=spec))
        await call(request_builder.build_evidence(case_id=case_id, spec=spec))
        access = gate.open(gate.issue(key), manifest_loader=lambda: evaluator)
        criterion_builder = CriterionChannelRequestBuilder(manifest_path.parent, access)
        await call(criterion_builder.build_criterion(case_id=case_id, spec=spec))

    gate_passed = len(attempts) == 6 and all(item.status == "success" for item in attempts)
    if plan.require_backend_fingerprint:
        gate_passed = gate_passed and all(item.backend_fingerprint for item in attempts)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_route_rehearsal_report.v1",
        "run_id": run_id,
        "protocol_hash": protocol.protocol_hash,
        "development_manifest_hash": public.source_manifest_hash,
        "rehearsal_plan_hash": model_content_hash(plan),
        "generated_at_utc": generated_at_utc,
        "attempt_count": len(attempts),
        "usable_count": sum(item.status == "success" for item in attempts),
        "attempts": tuple(attempts),
        "gate_passed": gate_passed,
    }
    draft = ExternalRouteRehearsalReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalRouteRehearsalReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(root / "external_protocol.json", protocol)
    write_immutable_json(root / "rehearsal_plan.json", plan)
    write_immutable_json(root / "public_manifest.json", public)
    write_immutable_json(root / "evaluator_manifest.json", evaluator)
    write_immutable_jsonl(root / "recorded_responses.jsonl", tuple(records))
    write_immutable_json(root / "route_rehearsal_report.json", report)
    return report


def run_external_route_rehearsal_from_environment(
    *,
    protocol: ExternalModelExecutionProtocol,
    plan: ExternalRouteRehearsalPlan,
    manifest_path: Path,
    fixture_decision_plan_path: Path,
    system_prompt_path: Path,
    output_root: Path,
    run_id: str,
) -> ExternalRouteRehearsalReport:
    """Run one explicit development rehearsal using the ignored local Cerebras key."""

    api_key = CerebrasExternalEvaluationSecrets().cerebras_api_key
    if api_key is None:
        raise ValueError(
            "SOCRATIC_CEREBRAS_API_KEY is required for external route rehearsal; "
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

    async def run() -> ExternalRouteRehearsalReport:
        async with client:
            return await run_external_route_rehearsal(
                protocol=protocol,
                plan=plan,
                manifest_path=manifest_path,
                fixture_decision_plan_path=fixture_decision_plan_path,
                system_prompt_path=system_prompt_path,
                output_root=output_root,
                run_id=run_id,
                gateway=gateway,
            )

    return asyncio.run(run())


async def _generate_attempt(
    gateway: RehearsalGateway, request: StudentGenerationRequest
) -> tuple[ExternalRouteRehearsalAttempt, RecordedGenerationResponse | None]:
    try:
        result = await gateway.generate(request)
    except CerebrasGatewayError as error:
        failure: dict[str, object] = {
            "case_id": request.case_id,
            "channel": request.channel,
            "request_hash": request.request_hash,
            "status": "provider_failure",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        failure.update(_provider_failure_details(error))
        return (
            ExternalRouteRehearsalAttempt.model_validate(failure),
            None,
        )
    metadata = result.response.provider_metadata
    return (
        ExternalRouteRehearsalAttempt(
            case_id=request.case_id,
            channel=request.channel,
            request_hash=request.request_hash,
            status="success",
            response_hash=result.response.response_hash,
            resolved_provider_id=metadata.resolved_provider_id,
            resolved_model_id=metadata.resolved_model_id,
            backend_fingerprint=metadata.backend_fingerprint,
            latency_ms=metadata.latency_ms,
        ),
        result.response,
    )


def _fixture_plan(
    path: Path,
    *,
    protocol: ExternalModelExecutionProtocol,
    run_id: str,
    created_at_utc: datetime,
) -> OfflineDecisionPlan:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Fixture decision plan must be a mapping")
    values = cast(dict[str, Any], raw)
    values.update(
        {
            "run_id": run_id,
            "model_route": {"provider": protocol.provider_id, "model": protocol.requested_model_id},
            "created_at_utc": created_at_utc,
            "root_seed": protocol.run_seed,
        }
    )
    return OfflineDecisionPlan.model_validate(values)


def _load_run_plan(path: Path):
    from socratic_tutor.benchmark.public.global_seal import DecisionRunPlan

    return DecisionRunPlan.model_validate_json(path.read_bytes())


def _validate_development_inputs(
    plan: ExternalRouteRehearsalPlan,
    manifest_path: Path,
    benchmark_version: str,
    status: str,
) -> None:
    if benchmark_version != "dev-v0" or status != "draft":
        raise ValueError("External route rehearsal may run only against draft dev-v0 cases")
    if manifest_path.resolve().as_posix().endswith(plan.development_manifest) is False:
        raise ValueError("Route rehearsal manifest does not match the development plan")


def _load_existing_report(
    root: Path,
    protocol: ExternalModelExecutionProtocol,
    plan: ExternalRouteRehearsalPlan,
    development_manifest_hash: str,
    run_id: str,
) -> ExternalRouteRehearsalReport | None:
    path = root / "route_rehearsal_report.json"
    if not path.exists():
        return None
    report = ExternalRouteRehearsalReport.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if (
        report.run_id != run_id
        or report.protocol_hash != protocol.protocol_hash
        or report.rehearsal_plan_hash != model_content_hash(plan)
        or report.development_manifest_hash != development_manifest_hash
    ):
        raise ValueError("Immutable external route rehearsal artifact already differs")
    return report


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Route rehearsal clock must return a UTC datetime")
    return value


def _provider_failure_details(error: CerebrasGatewayError) -> dict[str, object]:
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
            "provider_request_id": error.request_id,
            "backend_fingerprint": error.backend_fingerprint,
            "latency_ms": error.latency_ms,
            "provider_finish_reason": error.finish_reason,
            "provider_message_fields": error.message_fields,
        }
    return {}
