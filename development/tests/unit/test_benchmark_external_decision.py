"""Network-free tests for resumable held-out decision generation."""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.cerebras import (
    CerebrasUnavailableError,
    GatewayGenerationResult,
)
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.evidence_specificity import ExternalRunPreflight
from socratic_tutor.benchmark.external_decision import (
    ExternalDecisionGateway,
    ExternalDecisionGenerationPlan,
    ExternalDecisionGenerationReport,
    run_external_decision_generation,
)
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    create_external_model_execution_protocol,
)
from socratic_tutor.benchmark.generation import GenerationChannel, StudentGenerationRequest
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    create_recorded_response,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
ANALYSIS = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
PROMPT = (
    WORKSPACE_ROOT
    / "data"
    / "benchmark-execution"
    / "v2"
    / "prompts"
    / "external-evaluation-system-v1.md"
)
NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


class FakeGateway:
    def __init__(self, *, fail_key: tuple[str, GenerationChannel] | None = None) -> None:
        self.fail_key = fail_key
        self.requests: list[StudentGenerationRequest] = []

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult:
        self.requests.append(request)
        if (request.case_id, request.channel) == self.fail_key:
            raise CerebrasUnavailableError("planned terminal provider failure")
        response = create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response=f"Visible {request.channel.value} answer for {request.case_id}.",
            provider_metadata=ProviderResponseMetadata(
                provider_id="cerebras",
                model_id="gpt-oss-120b",
                resolved_provider_id="cerebras",
                resolved_model_id="gpt-oss-120b",
                backend_fingerprint="fp-heldout-test",
                latency_ms=7,
            ),
            captured_at_utc=NOW,
        )
        content = {"response": response, "attempt_count": 1, "retried_status_codes": ()}
        draft = GatewayGenerationResult.model_construct(
            _fields_set=set(content), **content, invocation_hash="0" * 64
        )
        return GatewayGenerationResult.model_validate(
            {**content, "invocation_hash": model_content_hash(draft, exclude={"invocation_hash"})}
        )


async def _no_sleep(_: float) -> None:
    return None


def test_generates_only_public_and_evidence_then_resumes_without_provider_calls(
    tmp_path: Path,
) -> None:
    protocol, preflight, plan = _frozen_inputs()
    gateway = FakeGateway()

    report = asyncio.run(
        _run(
            tmp_path,
            gateway=gateway,
            protocol=protocol,
            preflight=preflight,
            plan=plan,
        )
    )

    assert report.gate_passed is True
    assert report.request_count == report.success_count == 48
    assert report.eligible_case_count == 24
    assert len(gateway.requests) == 48
    assert {request.channel for request in gateway.requests} == {
        GenerationChannel.PUBLIC,
        GenerationChannel.EVIDENCE,
    }
    assert all(
        "criterion" not in request.model_dump_json().casefold() for request in gateway.requests
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "recorded_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 48

    replay_gateway = FakeGateway()
    replay = asyncio.run(
        _run(
            tmp_path,
            gateway=replay_gateway,
            protocol=protocol,
            preflight=preflight,
            plan=plan,
        )
    )
    assert replay.report_hash == report.report_hash
    assert replay_gateway.requests == []


def test_terminal_failure_is_reported_and_not_retried_on_resume(tmp_path: Path) -> None:
    protocol, preflight, plan = _frozen_inputs()
    first_case = load_and_verify_manifest(MANIFEST).cases[0].public.case_id
    failure_key = (first_case, GenerationChannel.EVIDENCE)
    gateway = FakeGateway(fail_key=failure_key)

    report = asyncio.run(
        _run(
            tmp_path,
            gateway=gateway,
            protocol=protocol,
            preflight=preflight,
            plan=plan,
        )
    )

    assert report.generation_complete is True
    assert report.gate_passed is False
    assert report.success_count == 47
    assert report.provider_failure_count == 1
    assert report.eligible_case_count == 23
    failure = next(attempt for attempt in report.attempts if attempt.status == "provider_failure")
    assert failure.case_id == first_case
    assert failure.channel is GenerationChannel.EVIDENCE

    replay_gateway = FakeGateway()
    replay = asyncio.run(
        _run(
            tmp_path,
            gateway=replay_gateway,
            protocol=protocol,
            preflight=preflight,
            plan=plan,
        )
    )
    assert replay.report_hash == report.report_hash
    assert replay_gateway.requests == []


def test_rejects_plan_bound_to_another_preflight(tmp_path: Path) -> None:
    protocol, preflight, plan = _frozen_inputs()
    wrong_plan = plan.model_copy(update={"preflight_hash": "f" * 64})

    try:
        asyncio.run(
            _run(
                tmp_path,
                gateway=FakeGateway(),
                protocol=protocol,
                preflight=preflight,
                plan=wrong_plan,
            )
        )
    except ValueError as error:
        assert "another preflight" in str(error)
    else:
        raise AssertionError("A plan with another preflight hash must fail")


async def _run(
    tmp_path: Path,
    *,
    gateway: ExternalDecisionGateway,
    protocol: ExternalModelExecutionProtocol,
    preflight: ExternalRunPreflight,
    plan: ExternalDecisionGenerationPlan,
) -> ExternalDecisionGenerationReport:
    return await run_external_decision_generation(
        protocol=protocol,
        preflight=preflight,
        plan=plan,
        manifest_path=MANIFEST,
        analysis_specification_path=ANALYSIS,
        system_prompt_path=PROMPT,
        output_root=tmp_path,
        run_id="heldout-decision-test",
        gateway=gateway,
        clock=lambda: NOW,
        sleeper=_no_sleep,
    )


def _frozen_inputs() -> tuple[
    ExternalModelExecutionProtocol,
    ExternalRunPreflight,
    ExternalDecisionGenerationPlan,
]:
    manifest = load_and_verify_manifest(MANIFEST)
    public = project_public_manifest(manifest, projected_at_utc=NOW)
    analysis = load_analysis_specification(ANALYSIS)
    protocol = create_external_model_execution_protocol(
        manifest=manifest,
        public_manifest=public,
        analysis_specification=analysis,
        provider_id="cerebras",
        adapter_version="cerebras-chat-completions-v1",
        requested_model_id="gpt-oss-120b",
        system_prompt_version="external-evaluation-system-v1",
        system_prompt_sha256=file_sha256(PROMPT.read_bytes()),
        temperature=0.0,
        max_output_tokens=512,
        run_seed=20260823,
        reasoning_effort="low",
        timeout_seconds=60.0,
        max_attempts=2,
        request_spacing_seconds=13.0,
        repeats_per_case=1,
        request_budget=72,
        cost_budget_usd=Decimal("5.00"),
        created_at_utc=NOW,
    )
    preflight_content = {
        "external_protocol_hash": protocol.protocol_hash,
        "analysis_specification_hash": analysis_specification_hash(analysis),
        "calibration_decision_hash": "1" * 64,
        "public_rating_amendment_hash": "2" * 64,
        "evidence_specificity_amendment_hash": "3" * 64,
        "gate_passed": True,
    }
    preflight_draft = ExternalRunPreflight.model_construct(
        _fields_set=set(preflight_content), **preflight_content, preflight_hash="0" * 64
    )
    preflight = ExternalRunPreflight.model_validate(
        {
            **preflight_content,
            "preflight_hash": model_content_hash(preflight_draft, exclude={"preflight_hash"}),
        }
    )
    plan = ExternalDecisionGenerationPlan(
        purpose="heldout_external_decision_generation",
        preflight_hash=preflight.preflight_hash,
        required_channels=(GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE),
        expected_case_count=24,
        expected_request_count=48,
        require_backend_fingerprint=True,
    )
    return protocol, preflight, plan
