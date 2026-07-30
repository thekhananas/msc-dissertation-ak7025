"""Network-free tests for the development-only student behavior audit."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.behavior_audit import (
    StudentBehaviorAuditPlan,
    load_student_behavior_audit_plan,
    run_student_behavior_audit,
)
from socratic_tutor.benchmark.cerebras import GatewayGenerationResult
from socratic_tutor.benchmark.generation import StudentGenerationRequest
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    create_recorded_response,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "student-behavior-audit-cerebras.yaml"
PLAN_V2 = WORKSPACE_ROOT / "configs" / "benchmark" / "student-behavior-audit-cerebras-v2.yaml"


class FakeGateway:
    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult:
        response = create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response="I think the learner state should guide this response.",
            provider_metadata=ProviderResponseMetadata(
                provider_id="cerebras",
                model_id="gpt-oss-120b",
                resolved_provider_id="cerebras",
                resolved_model_id="gpt-oss-120b",
                backend_fingerprint="fp-behavior-audit",
                latency_ms=7,
            ),
            captured_at_utc=datetime(2026, 8, 23, 11, 0, tzinfo=UTC),
        )
        content = {"response": response, "attempt_count": 1, "retried_status_codes": ()}
        draft = GatewayGenerationResult.model_construct(
            _fields_set=set(content), **content, invocation_hash="0" * 64
        )
        return GatewayGenerationResult.model_validate(
            {**content, "invocation_hash": model_content_hash(draft, exclude={"invocation_hash"})}
        )


def test_behavior_audit_writes_records_for_manual_review(tmp_path: Path) -> None:
    plan = load_student_behavior_audit_plan(PLAN)
    delays: list[float] = []

    async def collect_delay(delay: float) -> None:
        delays.append(delay)

    report = asyncio.run(
        run_student_behavior_audit(
            plan=plan,
            output_root=tmp_path,
            run_id="behavior-audit-test",
            gateway=FakeGateway(),
            clock=lambda: datetime(2026, 8, 23, 11, 0, tzinfo=UTC),
            sleeper=collect_delay,
        )
    )

    assert report.generation_complete is True
    assert report.manual_review_required is True
    assert report.scenario_count == 4
    assert report.success_count == 4
    assert (tmp_path / "student_behavior_audit_plan.json").exists()
    records = (tmp_path / "recorded_responses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 4
    assert delays == []


def test_behavior_audit_plan_requires_four_scenarios() -> None:
    plan = load_student_behavior_audit_plan(PLAN)
    raw = plan.model_dump(mode="python")
    raw["scenarios"] = raw["scenarios"][:3]

    try:
        StudentBehaviorAuditPlan.model_validate(raw)
    except ValueError as error:
        assert "at least 4 items" in str(error)
    else:
        raise AssertionError("Expected an audit plan with fewer than four scenarios to fail")


def test_behavior_audit_v2_predeclares_manual_acceptance_rule() -> None:
    plan = load_student_behavior_audit_plan(PLAN_V2)

    assert plan.audit_version == "v2"
    assert len(plan.scenarios) == 8
    assert plan.manual_acceptance is not None
    assert plan.manual_acceptance.minimum_adherent_count == 7
    assert set(plan.manual_acceptance.required_state_families) == {
        "aliasing-misconception",
        "falsy-misconception",
        "supported-mastery",
        "cautious-partial",
    }
    assert plan.transport.inter_request_delay_seconds == 13.0


def test_behavior_audit_v2_paces_each_request(tmp_path: Path) -> None:
    delays: list[float] = []

    async def collect_delay(delay: float) -> None:
        delays.append(delay)

    report = asyncio.run(
        run_student_behavior_audit(
            plan=load_student_behavior_audit_plan(PLAN_V2),
            output_root=tmp_path,
            run_id="behavior-audit-v2-pacing-test",
            gateway=FakeGateway(),
            clock=lambda: datetime(2026, 8, 23, 11, 0, tzinfo=UTC),
            sleeper=collect_delay,
        )
    )

    assert report.generation_complete is True
    assert delays == [13.0] * 7
