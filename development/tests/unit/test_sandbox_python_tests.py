"""Tests for preparing and parsing isolated authored Python tests."""

from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
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
DEV_BUNDLES = (
    WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "evidence" / "aliasing-tests.yaml",
    WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0" / "evidence" / "none-falsy-tests.yaml",
    WORKSPACE_ROOT
    / "data"
    / "benchmarks"
    / "dev-v0"
    / "evaluator"
    / "criterion"
    / "aliasing-tests.yaml",
    WORKSPACE_ROOT
    / "data"
    / "benchmarks"
    / "dev-v0"
    / "evaluator"
    / "criterion"
    / "none-falsy-tests.yaml",
)
ZERO_ARGUMENT_BUNDLE = (
    WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "evidence" / "h-c1m1-01-tests.yaml"
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


def test_loads_every_reviewed_development_test_shape() -> None:
    bundles = tuple(load_authored_function_test_bundle(path) for path in DEV_BUNDLES)

    assert tuple(bundle.function for bundle in bundles) == (
        "add_marker",
        "display_score",
        "snapshot_then_append",
        "choose_label",
    )
    assert bundles[1].checks[0].harness_check() == {
        "kind": "input",
        "input": None,
        "expected": "missing",
    }
    assert bundles[3].checks[0].harness_check() == {
        "kind": "label",
        "label": None,
        "default": "fallback",
        "expected": "fallback",
    }


def test_loads_frozen_zero_argument_evidence_shape() -> None:
    bundle = load_authored_function_test_bundle(ZERO_ARGUMENT_BUNDLE)

    assert bundle.function == "predicted_delivery_quote"
    assert bundle.checks[0].harness_check() == {"kind": "no_args", "expected": 24}


def test_loads_every_frozen_evidence_bundle_before_heldout_execution() -> None:
    manifest = load_and_verify_manifest(
        WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
    )
    bundles = tuple(
        load_authored_function_test_bundle(
            WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / case.public.evidence_test_ref
        )
        for case in manifest.cases
    )

    assert len(bundles) == 24
    assert {check.harness_check()["kind"] for bundle in bundles for check in bundle.checks} == {
        "no_args"
    }
