"""Development-only qualification for one direct external generation route."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from socratic_tutor.benchmark.artifacts import write_immutable_json, write_immutable_jsonl
from socratic_tutor.benchmark.cerebras import (
    CerebrasGateway,
    CerebrasGatewayConfig,
    CerebrasGatewayError,
    GatewayGenerationResult,
    HttpxCerebrasTransport,
)
from socratic_tutor.benchmark.common import RelativePath, Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import (
    BenchmarkSplit,
    PublicArtifactClass,
    PublicBenchmarkManifest,
)
from socratic_tutor.benchmark.public.requests import PublicChannelRequestBuilder
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel


class QualificationTransportConfig(ContractModel):
    """Bounded network behaviour for one qualification run."""

    timeout_seconds: float = Field(gt=0.0, le=120.0)
    max_attempts: int = Field(ge=1, le=3)
    retry_delay_seconds: float = Field(ge=0.0, le=10.0)


class QualificationCaseConfig(ContractModel):
    """Only development cases that may be used to qualify a provider."""

    manifest: RelativePath
    permitted_case_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> QualificationCaseConfig:
        if len(set(self.permitted_case_ids)) != len(self.permitted_case_ids):
            raise ValueError("Qualification case IDs must be unique")
        return self


class QualificationRecordConfig(ContractModel):
    """Operational fields that every successful qualification must retain."""

    provider_request_id: Literal[True] = True
    resolved_model_id: Literal[True] = True
    backend_fingerprint: Literal[True] = True
    input_tokens: Literal[True] = True
    output_tokens: Literal[True] = True
    latency_ms: Literal[True] = True


class ProviderQualificationPlan(ContractModel):
    """Explicit non-held-out route qualification plan."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.provider_qualification.v1"] = (
        "benchmark.provider_qualification.v1"
    )
    purpose: Literal["development_only_provider_qualification"]
    candidate_id: str = Field(min_length=1)
    model_route: ModelRoute
    sampling: SamplingConfig
    system_prompt_version: str = Field(min_length=1)
    system_prompt_ref: RelativePath
    system_prompt_sha256: Sha256
    transport: QualificationTransportConfig
    qualification_cases: QualificationCaseConfig
    record: QualificationRecordConfig
    require_backend_fingerprint: bool = True
    selection_rules: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_direct_cerebras_route(self) -> ProviderQualificationPlan:
        if self.model_route.provider != "cerebras":
            raise ValueError("This qualification plan requires the direct cerebras provider")
        if any(not rule.strip() for rule in self.selection_rules):
            raise ValueError("Qualification selection rules cannot be blank")
        return self


class QualificationCaseResult(ContractModel):
    """One public development-case outcome, without response content."""

    case_id: str = Field(min_length=1)
    request_hash: Sha256
    status: Literal["success", "provider_failure"]
    response_hash: Sha256 | None = None
    backend_fingerprint: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> QualificationCaseResult:
        if self.status == "success":
            if self.response_hash is None:
                raise ValueError("Successful qualification results require a response hash")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Successful qualification results cannot contain provider errors")
        elif self.response_hash is not None:
            raise ValueError("Provider failures cannot contain a response hash")
        return self


class ProviderQualificationReport(ContractModel):
    """Immutable summary used to decide whether a route may be frozen later."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.provider_qualification_report.v1"] = (
        "benchmark.provider_qualification_report.v1"
    )
    candidate_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["dev-v0"] = "dev-v0"
    source_manifest_hash: Sha256
    public_projection_hash: Sha256
    qualification_plan_hash: Sha256
    generated_at_utc: datetime
    result_count: int = Field(ge=1)
    success_count: int = Field(ge=0)
    results: tuple[QualificationCaseResult, ...] = Field(min_length=1)
    gate_passed: bool
    qualification_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> ProviderQualificationReport:
        if self.generated_at_utc.tzinfo is None:
            raise ValueError("Qualification timestamp must include a timezone")
        if self.result_count != len(self.results):
            raise ValueError("Qualification result count does not match results")
        if self.success_count != sum(result.status == "success" for result in self.results):
            raise ValueError("Qualification success count does not match results")
        if self.qualification_hash != model_content_hash(self, exclude={"qualification_hash"}):
            raise ValueError("Qualification report hash does not match its content")
        return self


class QualificationGateway(Protocol):
    """The small generation surface needed by development qualification."""

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult: ...


type Clock = Callable[[], datetime]


class CerebrasQualificationSecrets(BaseSettings):
    """The one ignored local secret required for a live qualification call."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SOCRATIC_", extra="ignore")

    cerebras_api_key: SecretStr | None = None


def load_provider_qualification_plan(path: Path) -> ProviderQualificationPlan:
    """Load a strict YAML plan with no provider side effects."""

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read provider qualification plan: {path}") from error
    return ProviderQualificationPlan.model_validate(raw)


def cerebras_gateway_from_environment(
    *,
    plan: ProviderQualificationPlan,
) -> tuple[CerebrasGateway, httpx.AsyncClient]:
    """Build a direct gateway only when an explicit local secret exists."""

    api_key = CerebrasQualificationSecrets().cerebras_api_key
    if api_key is None:
        raise ValueError(
            "SOCRATIC_CEREBRAS_API_KEY is required for provider qualification; "
            "store it only in ignored development/.env"
        )
    client = httpx.AsyncClient()
    gateway = CerebrasGateway(
        config=CerebrasGatewayConfig(
            api_key=api_key,
            timeout_seconds=plan.transport.timeout_seconds,
            max_attempts=plan.transport.max_attempts,
            retry_delay_seconds=plan.transport.retry_delay_seconds,
        ),
        transport=HttpxCerebrasTransport(client),
    )
    return gateway, client


async def run_provider_qualification(
    *,
    plan: ProviderQualificationPlan,
    manifest_path: Path,
    output_root: Path,
    run_id: str,
    gateway: QualificationGateway,
    clock: Clock | None = None,
) -> ProviderQualificationReport:
    """Generate public development responses and publish immutable local artifacts."""

    now = clock or (lambda: datetime.now(UTC))
    generated_at_utc = _require_utc(now())
    authored = load_and_verify_manifest(manifest_path)
    _validate_development_manifest(
        plan=plan,
        manifest_path=manifest_path,
        benchmark_version=authored.benchmark_version,
        status=authored.status,
    )
    public = project_public_manifest(authored, projected_at_utc=generated_at_utc)
    _validate_development_cases(plan=plan, public=public)
    benchmark_root = manifest_path.resolve().parent
    system_prompt = _load_public_system_prompt(
        plan=plan, benchmark_root=benchmark_root, public=public
    )
    spec = GenerationRequestSpec(
        run_id=run_id,
        sample_id="qualification-sample-001",
        system_prompt_version=plan.system_prompt_version,
        system_prompt=system_prompt,
        system_prompt_sha256=plan.system_prompt_sha256,
        model_route=plan.model_route,
        sampling=plan.sampling,
    )
    builder = PublicChannelRequestBuilder(benchmark_root, public)
    records: list[RecordedGenerationResponse] = []
    results: list[QualificationCaseResult] = []
    for case_id in plan.qualification_cases.permitted_case_ids:
        request = builder.build_public(case_id=case_id, spec=spec)
        try:
            result = await gateway.generate(request)
        except CerebrasGatewayError as error:
            results.append(
                QualificationCaseResult(
                    case_id=case_id,
                    request_hash=request.request_hash,
                    status="provider_failure",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            continue
        metadata = result.response.provider_metadata
        records.append(result.response)
        results.append(
            QualificationCaseResult(
                case_id=case_id,
                request_hash=request.request_hash,
                status="success",
                response_hash=result.response.response_hash,
                backend_fingerprint=metadata.backend_fingerprint,
                latency_ms=metadata.latency_ms,
            )
        )

    gate_passed = all(result.status == "success" for result in results)
    if plan.require_backend_fingerprint:
        gate_passed = gate_passed and all(result.backend_fingerprint for result in results)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.provider_qualification_report.v1",
        "candidate_id": plan.candidate_id,
        "run_id": run_id,
        "benchmark_version": "dev-v0",
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "qualification_plan_hash": model_content_hash(plan),
        "generated_at_utc": generated_at_utc,
        "result_count": len(results),
        "success_count": sum(result.status == "success" for result in results),
        "results": tuple(results),
        "gate_passed": gate_passed,
    }
    draft = ProviderQualificationReport.model_construct(
        _fields_set=set(content), **content, qualification_hash="0" * 64
    )
    report = ProviderQualificationReport.model_validate(
        {
            **content,
            "qualification_hash": model_content_hash(draft, exclude={"qualification_hash"}),
        }
    )
    write_immutable_json(output_root / "qualification_plan.json", plan)
    write_immutable_json(output_root / "public_manifest.json", public)
    write_immutable_jsonl(output_root / "recorded_responses.jsonl", tuple(records))
    write_immutable_json(output_root / "qualification_report.json", report)
    return report


def run_provider_qualification_from_environment(
    *,
    plan: ProviderQualificationPlan,
    manifest_path: Path,
    output_root: Path,
    run_id: str,
) -> ProviderQualificationReport:
    """Run the explicit direct-provider qualification with one local secret."""

    gateway, client = cerebras_gateway_from_environment(
        plan=plan,
    )

    async def run() -> ProviderQualificationReport:
        async with client:
            return await run_provider_qualification(
                plan=plan,
                manifest_path=manifest_path,
                output_root=output_root,
                run_id=run_id,
                gateway=gateway,
            )

    return asyncio.run(run())


def _validate_development_manifest(
    *,
    plan: ProviderQualificationPlan,
    manifest_path: Path,
    benchmark_version: str,
    status: ManifestStatus,
) -> None:
    if benchmark_version != "dev-v0":
        raise ValueError("Provider qualification may run only against the dev-v0 benchmark")
    if status is not ManifestStatus.DRAFT:
        raise ValueError("Provider qualification requires the development manifest to remain draft")
    if manifest_path.resolve().as_posix().endswith(plan.qualification_cases.manifest) is False:
        raise ValueError("Qualification manifest does not match the plan's development manifest")


def _validate_development_cases(
    *, plan: ProviderQualificationPlan, public: PublicBenchmarkManifest
) -> None:
    cases = {case.case_id: case for case in public.cases}
    for case_id in plan.qualification_cases.permitted_case_ids:
        try:
            case = cases[case_id]
        except KeyError as error:
            raise ValueError(
                f"Qualification case is absent from the manifest: {case_id}"
            ) from error
        if case.split is not BenchmarkSplit.DEVELOPMENT:
            raise ValueError(f"Qualification case is not in the development split: {case_id}")


def _load_public_system_prompt(
    *, plan: ProviderQualificationPlan, benchmark_root: Path, public: PublicBenchmarkManifest
) -> str:
    entries = {entry.path: entry for entry in public.files}
    try:
        entry = entries[plan.system_prompt_ref]
    except KeyError as error:
        raise ValueError(
            "Qualification system prompt is absent from the public inventory"
        ) from error
    if entry.artifact_class is not PublicArtifactClass.PROMPT:
        raise ValueError("Qualification system prompt must be a public prompt artifact")
    path = (benchmark_root / plan.system_prompt_ref).resolve()
    if not path.is_relative_to(benchmark_root.resolve()):
        raise ValueError("Qualification system prompt resolves outside the benchmark root")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError("Could not read qualification system prompt") from error
    if (
        file_sha256(content) != plan.system_prompt_sha256
        or entry.sha256 != plan.system_prompt_sha256
    ):
        raise ValueError("Qualification system prompt hash does not match the public inventory")
    return content.decode("utf-8")


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Qualification clock must return a UTC-aware timestamp")
    return value
