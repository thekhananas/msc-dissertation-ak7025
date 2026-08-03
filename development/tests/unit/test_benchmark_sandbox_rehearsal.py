"""Network-free coverage for the recorded development sandbox rehearsal."""

import json
from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    EvidenceTaskPayload,
    GenerationRequestSpec,
    ModelRoute,
    PublicTaskPayload,
    SamplingConfig,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    RecordedGenerationResponse,
    create_recorded_response,
)
from socratic_tutor.benchmark.sandbox_rehearsal import (
    load_sandbox_rehearsal_plan,
    run_recorded_sandbox_rehearsal,
)
from socratic_tutor.sandbox import SandboxExecutionRequest, SandboxExecutionResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "manifest.yaml"
PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-sandbox-rehearsal.yaml"

CASES = (
    (
        "dev-aliasing-001",
        "dev-list-identity",
        "list-aliasing-and-copying",
        "dev-aliasing-snapshot-criterion-001",
    ),
    (
        "dev-none-falsy-001",
        "dev-missing-values",
        "none-versus-falsy-values",
        "dev-none-falsy-label-criterion-001",
    ),
)


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[SandboxExecutionRequest] = []

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        self.requests.append(request)
        return SandboxExecutionResult(
            status="completed",
            sandbox_id=f"sb-{len(self.requests)}",
            exit_code=0,
            stdout='{"status":"completed","passed":2,"failed":0}',
        )


def _write_recorded_responses(path: Path) -> None:
    system_prompt = "Return only the requested visible answer."
    spec = GenerationRequestSpec(
        run_id="development-route-rehearsal-test",
        sample_id="sample-001",
        system_prompt_version="test-system-v1",
        system_prompt=system_prompt,
        system_prompt_sha256=file_sha256(system_prompt.encode("utf-8")),
        model_route=ModelRoute(provider="cerebras", model="gpt-oss-120b"),
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=128, seed=20260823),
    )
    records: list[RecordedGenerationResponse] = []
    for case_id, task_family, target_concept, criterion_probe_id in CASES:
        payloads = (
            PublicTaskPayload(
                benchmark_version="dev-v0",
                case_id=case_id,
                task_family=task_family,
                target_concept=target_concept,
                public_interaction="Explain what the program prints.",
            ),
            EvidenceTaskPayload(
                benchmark_version="dev-v0",
                case_id=case_id,
                task_family=task_family,
                target_concept=target_concept,
                evidence_probe="Complete the evidence function.",
            ),
            CriterionTaskPayload(
                benchmark_version="dev-v0",
                case_id=case_id,
                criterion_probe_id=criterion_probe_id,
                criterion_probe="Complete the held-out function.",
            ),
        )
        for payload in payloads:
            request = create_student_generation_request(spec=spec, payload=payload)
            final_response = (
                f"Recorded public response for {case_id}."
                if payload.channel.value == "public"
                else "```python\ndef solution():\n    return None\n```"
            )
            records.append(
                create_recorded_response(
                    request=request,
                    original_source=OriginalGenerationSource.CEREBRAS,
                    final_response=final_response,
                    provider_metadata=ProviderResponseMetadata(
                        provider_id="cerebras",
                        model_id="gpt-oss-120b",
                    ),
                    captured_at_utc=datetime(2026, 8, 23, 22, 0, tzinfo=UTC),
                )
            )
    path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )


def test_replays_all_channels_and_executes_only_recorded_code_channels(tmp_path: Path) -> None:
    records_path = tmp_path / "recorded_responses.jsonl"
    _write_recorded_responses(records_path)
    executor = FakeExecutor()
    report = run_recorded_sandbox_rehearsal(
        plan=load_sandbox_rehearsal_plan(PLAN),
        manifest_path=MANIFEST,
        recorded_responses_path=records_path,
        output_root=tmp_path,
        run_id="development-sandbox-rehearsal-test",
        executor=executor,
        generated_at_utc=datetime(2026, 8, 23, 22, 30, tzinfo=UTC),
    )

    assert report.gate_passed is True
    assert report.recorded_response_count == 6
    assert report.replayed_response_count == 6
    assert report.execution_count == 4
    assert report.completed_execution_count == 4
    assert len(executor.requests) == 4
    assert {attempt.channel.value for attempt in report.attempts} == {"evidence", "criterion"}
    stored = json.loads((tmp_path / "sandbox_rehearsal_report.json").read_text(encoding="utf-8"))
    assert stored["report_hash"] == report.report_hash
