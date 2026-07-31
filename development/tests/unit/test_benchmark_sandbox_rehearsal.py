"""Network-free coverage for the recorded development sandbox rehearsal."""

import json
from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.sandbox_rehearsal import (
    load_sandbox_rehearsal_plan,
    run_recorded_sandbox_rehearsal,
)
from socratic_tutor.sandbox import SandboxExecutionRequest, SandboxExecutionResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "manifest.yaml"
PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-sandbox-rehearsal.yaml"
RECORDS = (
    WORKSPACE_ROOT
    / "artifacts"
    / "external-route-rehearsal"
    / "cerebras-gpt-oss-120b-20260823-001"
    / "recorded_responses.jsonl"
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


def test_replays_all_channels_and_executes_only_recorded_code_channels(tmp_path: Path) -> None:
    executor = FakeExecutor()
    report = run_recorded_sandbox_rehearsal(
        plan=load_sandbox_rehearsal_plan(PLAN),
        manifest_path=MANIFEST,
        recorded_responses_path=RECORDS,
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
