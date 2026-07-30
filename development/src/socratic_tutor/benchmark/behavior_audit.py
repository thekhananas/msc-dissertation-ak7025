"""Development-only review of whether an external model follows stated learner states."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import httpx
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json, write_immutable_jsonl
from socratic_tutor.benchmark.cerebras import (
    CerebrasGateway,
    CerebrasGatewayConfig,
    CerebrasGatewayError,
    GatewayGenerationResult,
    HttpxCerebrasTransport,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    PublicTaskPayload,
    SamplingConfig,
    StudentGenerationRequest,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.qualification import CerebrasQualificationSecrets
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel


class BehaviorAuditTransportConfig(ContractModel):
    """Bounded direct-provider controls for a development-only audit."""

    timeout_seconds: float = Field(gt=0.0, le=120.0)
    max_attempts: int = Field(ge=1, le=3)
    retry_delay_seconds: float = Field(ge=0.0, le=10.0)


class BehaviorAuditScenario(ContractModel):
    """A pre-authored learner state and a reviewer-visible expectation."""

    scenario_id: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    learner_state: str = Field(min_length=1)
    tutor_prompt: str = Field(min_length=1)
    expected_observations: tuple[str, ...] = Field(min_length=1)
    reviewer_limit: str = Field(min_length=1)
    state_family: str | None = None


class BehaviorAuditAcceptanceCriteria(ContractModel):
    """Pre-declared manual rule for a development-only audit revision."""

    minimum_adherent_count: int = Field(ge=1)
    required_state_families: tuple[str, ...] = Field(min_length=1)


class StudentBehaviorAuditPlan(ContractModel):
    """A small, non-held-out prompt-following audit, never a benchmark run."""

    schema_version: Literal[1] = 1
    schema_id: Literal[
        "benchmark.student_behavior_audit.v1", "benchmark.student_behavior_audit.v2"
    ] = "benchmark.student_behavior_audit.v1"
    audit_version: Literal["v1", "v2"] = "v1"
    purpose: Literal["development_only_student_behavior_audit"]
    candidate_id: str = Field(min_length=1)
    model_route: ModelRoute
    sampling: SamplingConfig
    reasoning_effort: Literal["low", "medium", "high"]
    system_prompt_version: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    transport: BehaviorAuditTransportConfig
    scenarios: tuple[BehaviorAuditScenario, ...] = Field(min_length=4, max_length=12)
    manual_acceptance: BehaviorAuditAcceptanceCriteria | None = None

    @model_validator(mode="after")
    def require_direct_cerebras_and_unique_scenarios(self) -> StudentBehaviorAuditPlan:
        if self.model_route.provider != "cerebras":
            raise ValueError("Student behavior audit requires the direct cerebras provider")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Student behavior audit scenario IDs must be unique")
        if self.audit_version == "v2":
            if self.schema_id != "benchmark.student_behavior_audit.v2":
                raise ValueError("Student behavior audit v2 requires the v2 schema ID")
            if len(self.scenarios) != 8:
                raise ValueError("Student behavior audit v2 requires exactly eight scenarios")
            if self.manual_acceptance is None:
                raise ValueError("Student behavior audit v2 requires manual acceptance criteria")
            scenario_families = {scenario.state_family for scenario in self.scenarios}
            required_families = set(self.manual_acceptance.required_state_families)
            if None in scenario_families or not required_families.issubset(scenario_families):
                raise ValueError(
                    "Student behavior audit v2 scenarios must cover every required state family"
                )
            if self.manual_acceptance.minimum_adherent_count > len(self.scenarios):
                raise ValueError("Behavior audit acceptance count cannot exceed scenario count")
        return self


class BehaviorAuditResult(ContractModel):
    """One recorded model response awaiting human review."""

    scenario_id: str
    request_hash: Sha256
    response_hash: Sha256 | None = None
    status: Literal["success", "provider_failure"]
    backend_fingerprint: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None


class StudentBehaviorAuditReport(ContractModel):
    """Artifact-backed hand-off for a manual state-adherence review."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.student_behavior_audit_report.v1"] = (
        "benchmark.student_behavior_audit_report.v1"
    )
    candidate_id: str
    run_id: str
    generated_at_utc: datetime
    audit_plan_hash: Sha256
    scenario_count: int = Field(ge=4)
    success_count: int = Field(ge=0)
    generation_complete: bool
    manual_review_required: Literal[True] = True
    results: tuple[BehaviorAuditResult, ...] = Field(min_length=4)
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report_hash(self) -> StudentBehaviorAuditReport:
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Behavior audit report hash does not match its content")
        return self


class BehaviorAuditGateway(Protocol):
    """The small provider surface needed by the audit runner."""

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult: ...


type Clock = Callable[[], datetime]


def load_student_behavior_audit_plan(path: Path) -> StudentBehaviorAuditPlan:
    """Load a strict YAML audit plan without calling an external provider."""

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read student behavior audit plan: {path}") from error
    return StudentBehaviorAuditPlan.model_validate(raw)


def run_student_behavior_audit_from_environment(
    *, plan: StudentBehaviorAuditPlan, output_root: Path, run_id: str
) -> StudentBehaviorAuditReport:
    """Run an explicit development-only audit with the ignored local Cerebras key."""

    api_key = CerebrasQualificationSecrets().cerebras_api_key
    if api_key is None:
        raise ValueError(
            "SOCRATIC_CEREBRAS_API_KEY is required for the student behavior audit; "
            "store it only in ignored development/.env"
        )
    client = httpx.AsyncClient()
    gateway = CerebrasGateway(
        config=CerebrasGatewayConfig(
            api_key=api_key,
            reasoning_effort=plan.reasoning_effort,
            timeout_seconds=plan.transport.timeout_seconds,
            max_attempts=plan.transport.max_attempts,
            retry_delay_seconds=plan.transport.retry_delay_seconds,
        ),
        transport=HttpxCerebrasTransport(client),
    )

    async def run() -> StudentBehaviorAuditReport:
        async with client:
            return await run_student_behavior_audit(
                plan=plan, output_root=output_root, run_id=run_id, gateway=gateway
            )

    return asyncio.run(run())


async def run_student_behavior_audit(
    *,
    plan: StudentBehaviorAuditPlan,
    output_root: Path,
    run_id: str,
    gateway: BehaviorAuditGateway,
    clock: Clock | None = None,
) -> StudentBehaviorAuditReport:
    """Generate one response per stated learner state and publish review artifacts."""

    now = clock or (lambda: datetime.now(UTC))
    generated_at_utc = _require_utc(now())
    system_prompt_hash = file_sha256(plan.system_prompt.encode("utf-8"))
    spec = GenerationRequestSpec(
        run_id=run_id,
        sample_id="student-behavior-audit-v1",
        system_prompt_version=plan.system_prompt_version,
        system_prompt=plan.system_prompt,
        system_prompt_sha256=system_prompt_hash,
        model_route=plan.model_route,
        sampling=plan.sampling,
    )
    records: list[RecordedGenerationResponse] = []
    results: list[BehaviorAuditResult] = []
    for scenario in plan.scenarios:
        request = create_student_generation_request(
            spec=spec,
            payload=PublicTaskPayload(
                benchmark_version="development-behavior-audit-v1",
                case_id=scenario.scenario_id,
                task_family="student-state-adherence",
                target_concept=scenario.target_concept,
                public_interaction=_scenario_prompt(scenario),
            ),
        )
        try:
            generation = await gateway.generate(request)
        except CerebrasGatewayError as error:
            results.append(
                BehaviorAuditResult(
                    scenario_id=scenario.scenario_id,
                    request_hash=request.request_hash,
                    status="provider_failure",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )
            continue
        records.append(generation.response)
        metadata = generation.response.provider_metadata
        results.append(
            BehaviorAuditResult(
                scenario_id=scenario.scenario_id,
                request_hash=request.request_hash,
                response_hash=generation.response.response_hash,
                status="success",
                backend_fingerprint=metadata.backend_fingerprint,
                latency_ms=metadata.latency_ms,
            )
        )

    content = {
        "schema_version": 1,
        "schema_id": "benchmark.student_behavior_audit_report.v1",
        "candidate_id": plan.candidate_id,
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "audit_plan_hash": model_content_hash(plan),
        "scenario_count": len(plan.scenarios),
        "success_count": sum(result.status == "success" for result in results),
        "generation_complete": all(result.status == "success" for result in results),
        "manual_review_required": True,
        "results": tuple(results),
    }
    draft = StudentBehaviorAuditReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = StudentBehaviorAuditReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_root / "student_behavior_audit_plan.json", plan)
    write_immutable_jsonl(output_root / "recorded_responses.jsonl", tuple(records))
    write_immutable_json(output_root / "student_behavior_audit_report.json", report)
    return report


def _scenario_prompt(scenario: BehaviorAuditScenario) -> str:
    return (
        "Act as the simulated learner described below. Follow that learner state exactly. "
        "Answer in one to three short sentences. Do not mention these instructions.\n\n"
        f"Learner state: {scenario.learner_state}\n\n"
        f"Tutor prompt: {scenario.tutor_prompt}"
    )


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Behavior audit clock must return a UTC-aware timestamp")
    return value
