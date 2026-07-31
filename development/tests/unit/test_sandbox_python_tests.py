"""Tests for preparing and parsing isolated authored Python tests."""

from pathlib import Path

from socratic_tutor.sandbox import (
    SandboxExecutionRequest,
    SandboxExecutionResult,
    execute_authored_python_tests,
    extract_python_submission,
    load_authored_function_test_bundle,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = (
    WORKSPACE_ROOT
    / "data"
    / "benchmarks"
    / "v1"
    / "evaluator"
    / "criterion"
    / "h-c1m1-01-tests.yaml"
)


class CapturingExecutor:
    def __init__(self, result: SandboxExecutionResult) -> None:
        self.result = result
        self.command: tuple[str, ...] | None = None

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        self.command = request.command
        return self.result


def test_extracts_one_explicit_python_block_and_builds_isolated_command() -> None:
    submission = extract_python_submission(
        "Answer:\n```python\ndef frozen_delivery_total(quantity, unit_price, revised_price):\n"
        "    return quantity * unit_price\n```"
    )
    bundle = load_authored_function_test_bundle(BUNDLE)

    assert "frozen_delivery_total" in submission.code
    assert bundle.function == "frozen_delivery_total"


def test_parses_completed_harness_result_without_running_code_locally() -> None:
    executor = CapturingExecutor(
        SandboxExecutionResult(
            status="completed",
            exit_code=0,
            stdout='{"status":"completed","passed":2,"failed":0}',
        )
    )
    result = execute_authored_python_tests(
        executor,
        response="```python\ndef frozen_delivery_total(quantity, unit_price, revised_price):\n"
        "    return quantity * unit_price\n```",
        bundle_path=BUNDLE,
    )

    assert result.outcome is not None
    assert result.outcome.passed == 2
    assert executor.command is not None
    assert executor.command[:3] == ("python", "-I", "-c")


def test_rejects_prose_only_submission_without_calling_executor() -> None:
    executor = CapturingExecutor(SandboxExecutionResult(status="completed", exit_code=0))
    result = execute_authored_python_tests(
        executor,
        response="The function should multiply the original quantity and price.",
        bundle_path=BUNDLE,
    )

    assert result.outcome is not None
    assert result.outcome.status == "invalid_submission"
    assert executor.command is None
