"""Network-free tests for the development-only full-channel external rehearsal."""

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from socratic_tutor.benchmark.analysis_spec import load_analysis_specification
from socratic_tutor.benchmark.cerebras import GatewayGenerationResult
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.external_protocol import (
    create_external_model_execution_protocol,
)
from socratic_tutor.benchmark.external_rehearsal import (
    load_external_route_rehearsal_plan,
    run_external_route_rehearsal,
)
from socratic_tutor.benchmark.generation import StudentGenerationRequest
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    create_recorded_response,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEV_MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "manifest.yaml"
V1_MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-external-route-rehearsal-cerebras.yaml"
FIXTURE_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "dev-v0-decision.yaml"
ANALYSIS = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
PROMPT = (
    WORKSPACE_ROOT
    / "data"
    / "benchmark-execution"
    / "v2"
    / "prompts"
    / "external-evaluation-system-v1.md"
)


class FakeGateway:
    def __init__(self) -> None:
        self.requests: list[StudentGenerationRequest] = []

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult:
        self.requests.append(request)
        response = create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response=f"Visible answer for {request.channel.value}.",
            provider_metadata=ProviderResponseMetadata(
                provider_id="cerebras",
                model_id="gpt-oss-120b",
                resolved_provider_id="cerebras",
                resolved_model_id="gpt-oss-120b",
                backend_fingerprint="fp-route-rehearsal",
                input_tokens=10,
                output_tokens=5,
                latency_ms=7,
            ),
            captured_at_utc=datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
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


def _protocol():
    v1 = load_and_verify_manifest(V1_MANIFEST)
    public = project_public_manifest(v1, projected_at_utc=datetime(2026, 8, 23, 1, 15, tzinfo=UTC))
    return create_external_model_execution_protocol(
        manifest=v1,
        public_manifest=public,
        analysis_specification=load_analysis_specification(ANALYSIS),
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
        created_at_utc=datetime(2026, 8, 23, 1, 15, tzinfo=UTC),
    )


def test_rehearsal_calls_each_isolated_channel_on_two_development_cases(tmp_path: Path) -> None:
    gateway = FakeGateway()
    report = asyncio.run(
        run_external_route_rehearsal(
            protocol=_protocol(),
            plan=load_external_route_rehearsal_plan(PLAN),
            manifest_path=DEV_MANIFEST,
            fixture_decision_plan_path=FIXTURE_PLAN,
            system_prompt_path=PROMPT,
            output_root=tmp_path,
            run_id="development-route-rehearsal-test",
            gateway=gateway,
            clock=lambda: datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
            sleeper=_no_sleep,
        )
    )

    assert report.gate_passed is True
    assert report.attempt_count == 6
    assert report.usable_count == 6
    assert {(request.case_id, request.channel.value) for request in gateway.requests} == {
        (case_id, channel)
        for case_id in ("dev-aliasing-001", "dev-none-falsy-001")
        for channel in ("public", "evidence", "criterion")
    }
    records = [
        json.loads(line)
        for line in (tmp_path / "recorded_responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 6
    assert {record["request"]["channel"] for record in records} == {
        "public",
        "evidence",
        "criterion",
    }
    assert (tmp_path / "fixture_decision" / "decision" / "commitment_manifest.json").exists()

    replay = asyncio.run(
        run_external_route_rehearsal(
            protocol=_protocol(),
            plan=load_external_route_rehearsal_plan(PLAN),
            manifest_path=DEV_MANIFEST,
            fixture_decision_plan_path=FIXTURE_PLAN,
            system_prompt_path=PROMPT,
            output_root=tmp_path,
            run_id="development-route-rehearsal-test",
            gateway=gateway,
            clock=lambda: datetime(2026, 8, 23, 13, 0, tzinfo=UTC),
            sleeper=_no_sleep,
        )
    )
    assert replay.report_hash == report.report_hash
    assert len(gateway.requests) == 6
