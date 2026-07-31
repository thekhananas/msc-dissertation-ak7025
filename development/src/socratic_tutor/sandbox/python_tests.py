"""Prepare isolated execution of extracted Python against one authored test bundle."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import Field, model_validator

from socratic_tutor.contracts import ContractModel
from socratic_tutor.sandbox.modal import SandboxExecutionRequest, SandboxExecutionResult

_PYTHON_BLOCK = re.compile(r"```(?:python|py)\s*\n(?P<code>.*?)```", re.DOTALL | re.IGNORECASE)


class ExtractedPythonSubmission(ContractModel):
    """One visible Python block extracted from a model response."""

    code: str = Field(min_length=1)


class AuthoredFunctionTestBundle(ContractModel):
    """The limited reviewed test shape used by the frozen v1 criterion bundles."""

    schema_id: Literal["benchmark.authored_tests.v1"]
    function: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    checks: tuple[AuthoredFunctionCheck, ...] = Field(min_length=1)
    scoring: dict[str, str] = Field(min_length=1)


class AuthoredFunctionCheck(ContractModel):
    """One zero-argument or keyword-argument expected-output check."""

    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    expected: Any


class IsolatedTestOutcome(ContractModel):
    """Parsed observable result emitted by the sandbox test harness."""

    status: Literal["completed", "invalid_submission", "execution_error"]
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    detail: str | None = None

    @model_validator(mode="after")
    def require_completed_counts(self) -> IsolatedTestOutcome:
        if self.status == "completed" and self.passed + self.failed == 0:
            raise ValueError("Completed test outcome requires at least one test")
        return self


class SandboxExecutor(Protocol):
    """The only executor surface required by the benchmark scorer."""

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult: ...


class SandboxTestResult(ContractModel):
    """Execution result plus a typed outcome when the harness completed."""

    execution: SandboxExecutionResult
    outcome: IsolatedTestOutcome | None = None


def extract_python_submission(response: str) -> ExtractedPythonSubmission:
    """Accept one explicit Python fence and reject prose-only or ambiguous responses."""

    matches = tuple(match.group("code").strip() for match in _PYTHON_BLOCK.finditer(response))
    non_empty = tuple(code for code in matches if code)
    if len(non_empty) != 1:
        raise ValueError("Expected exactly one non-empty fenced Python code block")
    return ExtractedPythonSubmission(code=non_empty[0])


def load_authored_function_test_bundle(path: Path) -> AuthoredFunctionTestBundle:
    """Load one reviewed evaluator-side test bundle."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read authored test bundle: {path}") from error
    return AuthoredFunctionTestBundle.model_validate(raw)


def build_isolated_python_test_request(
    submission: ExtractedPythonSubmission,
    bundle: AuthoredFunctionTestBundle,
    *,
    timeout_seconds: int = 15,
    max_output_characters: int = 4096,
) -> SandboxExecutionRequest:
    """Build a `python -I` command for a network-blocked external sandbox."""

    encoded_submission = _encode(submission.code)
    encoded_bundle = _encode(json.dumps(bundle.model_dump(mode="json"), sort_keys=True))
    return SandboxExecutionRequest(
        command=(
            "python",
            "-I",
            "-c",
            _HARNESS,
            encoded_submission,
            encoded_bundle,
        ),
        timeout_seconds=timeout_seconds,
        max_output_characters=max_output_characters,
    )


def execute_authored_python_tests(
    executor: SandboxExecutor,
    *,
    response: str,
    bundle_path: Path,
) -> SandboxTestResult:
    """Submit one visible response to the external sandbox and parse its bounded result."""

    try:
        submission = extract_python_submission(response)
    except ValueError as error:
        return SandboxTestResult(
            execution=SandboxExecutionResult(
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            ),
            outcome=IsolatedTestOutcome(
                status="invalid_submission", passed=0, failed=0, detail=str(error)
            ),
        )
    bundle = load_authored_function_test_bundle(bundle_path)
    request = build_isolated_python_test_request(submission, bundle)
    execution = executor.execute(request)
    if execution.status != "completed" or execution.exit_code != 0:
        return SandboxTestResult(execution=execution)
    try:
        outcome = IsolatedTestOutcome.model_validate(json.loads(execution.stdout))
    except (json.JSONDecodeError, ValueError) as error:
        return SandboxTestResult(
            execution=execution.model_copy(
                update={
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": "Sandbox did not emit a valid harness result",
                }
            )
        )
    return SandboxTestResult(execution=execution, outcome=outcome)


def _encode(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


_HARNESS = r"""
import base64
import contextlib
import io
import json
import sys


class BoundedOutput(io.TextIOBase):
    def __init__(self, limit):
        self.limit = limit
        self.size = 0

    def write(self, value):
        self.size += len(value)
        if self.size > self.limit:
            raise RuntimeError("submission output exceeded the harness limit")
        return len(value)


def decode(value):
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


try:
    source = decode(sys.argv[1])
    bundle = json.loads(decode(sys.argv[2]))
    namespace = {"__name__": "submission"}
    with contextlib.redirect_stdout(BoundedOutput(4096)):
        with contextlib.redirect_stderr(BoundedOutput(4096)):
            exec(compile(source, "submission.py", "exec"), namespace, namespace)
            candidate = namespace[bundle["function"]]
            if not callable(candidate):
                raise TypeError("reviewed function name is not callable")
            passed = 0
            failed = 0
            for check in bundle["checks"]:
                actual = candidate(**check.get("args", {}))
                if actual == check["expected"]:
                    passed += 1
                else:
                    failed += 1
    print(json.dumps({"status": "completed", "passed": passed, "failed": failed}))
except (KeyError, SyntaxError, TypeError, ValueError, RuntimeError) as error:
    print(json.dumps({"status": "execution_error", "passed": 0, "failed": 0, "detail": str(error)}))
"""
