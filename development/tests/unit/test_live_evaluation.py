"""Boundary checks for the optional live evaluation illustration."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.generation import GenerationChannel, StudentGenerationRequest
from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.contracts import LiveEvaluationPhase, StartLiveEvaluationRequest
from socratic_tutor.live_evaluation import (
    LiveEvaluationService,
    LiveModelResponse,
    load_live_evaluation_assets,
)
from socratic_tutor.sandbox import SandboxExecutionRequest, SandboxExecutionResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class FakeGateway:
    def __init__(self) -> None:
        self.channels: list[GenerationChannel] = []

    async def generate(self, request: StudentGenerationRequest) -> LiveModelResponse:
        self.channels.append(request.channel)
        responses = {
            GenerationChannel.PUBLIC: "Both names refer to the same list object.",
            GenerationChannel.EVIDENCE: (
                "```python\ndef add_marker(items):\n"
                "    alias = items\n    alias.append('checked')\n    return alias\n```"
            ),
            GenerationChannel.CRITERION: (
                "```python\ndef snapshot_then_append(items, value):\n"
                "    snapshot = list(items)\n    items.append(value)\n"
                "    return snapshot, items\n```"
            ),
        }
        response = responses[request.channel]
        return LiveModelResponse(
            final_response=response,
            response_hash=canonical_sha256(
                {"request_hash": request.request_hash, "response": response}
            ),
            latency_ms=25,
        )


class FakeExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        self.calls += 1
        passed = 2 if self.calls == 1 else 1
        return SandboxExecutionResult(
            status="completed",
            sandbox_id=f"sb-live-{self.calls}",
            exit_code=0,
            stdout=(f'{{"status":"completed","passed":{passed},"failed":0,"detail":null}}'),
        )


async def _run_live_flow() -> None:
    gateway = FakeGateway()
    executor = FakeExecutor()
    times = iter(
        (
            datetime(2026, 9, 11, 10, 0, tzinfo=UTC),
            datetime(2026, 9, 11, 10, 1, tzinfo=UTC),
        )
    )
    service = LiveEvaluationService(
        enabled=True,
        case_id="dev-aliasing-001",
        assets=load_live_evaluation_assets(
            WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "manifest.yaml",
            WORKSPACE_ROOT / "data" / "demo" / "live-evaluation-system-v1.md",
            "dev-aliasing-001",
        ),
        gateway=gateway,
        executor=executor,
        clock=lambda: next(times),
    )
    request = StartLiveEvaluationRequest(
        idempotency_key="live-boundary-test",
        case_id="dev-aliasing-001",
    )

    committed = await service.start(request)
    repeated = await service.start(request)
    revealed = await service.reveal(committed.run_id)

    assert repeated == committed
    assert committed.phase is LiveEvaluationPhase.PREDICTIONS_COMMITTED
    assert committed.outcome is None
    assert committed.model_calls_made == 2
    assert committed.sandbox_calls_made == 1
    assert revealed.phase is LiveEvaluationPhase.OUTCOME_REVEALED
    assert revealed.outcome is not None
    assert revealed.predictions == committed.predictions
    assert revealed.commitment_hash == committed.commitment_hash
    assert revealed.model_calls_made == 3
    assert revealed.sandbox_calls_made == 2
    assert gateway.channels == [
        GenerationChannel.PUBLIC,
        GenerationChannel.EVIDENCE,
        GenerationChannel.CRITERION,
    ]
    assert executor.calls == 2


def test_live_result_is_requested_only_after_predictions_are_committed() -> None:
    asyncio.run(_run_live_flow())
