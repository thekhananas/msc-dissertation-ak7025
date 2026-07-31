"""Development-only replay and isolated execution of recorded external responses."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.generation import GenerationChannel
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
    replay_recorded_response,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.sandbox import (
    ModalSandboxExecutor,
    SandboxExecutor,
    execute_authored_python_tests,
)


class SandboxRehearsalPlan(ContractModel):
    """Narrow configuration for a recorded, development-only sandbox rehearsal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.sandbox_rehearsal_plan.v1"] = (
        "benchmark.sandbox_rehearsal_plan.v1"
    )
    purpose: Literal["development_only_recorded_sandbox_rehearsal"]
    development_manifest: str = Field(min_length=1)
    permitted_case_ids: tuple[str, ...] = Field(min_length=1)
    required_channels: tuple[GenerationChannel, ...] = (
        GenerationChannel.PUBLIC,
        GenerationChannel.EVIDENCE,
        GenerationChannel.CRITERION,
    )
    execution_channels: tuple[GenerationChannel, ...] = (
        GenerationChannel.EVIDENCE,
        GenerationChannel.CRITERION,
    )

    @model_validator(mode="after")
    def require_fixed_development_scope(self) -> SandboxRehearsalPlan:
        if len(self.permitted_case_ids) != 2 or len(set(self.permitted_case_ids)) != 2:
            raise ValueError("Sandbox rehearsal requires exactly two development cases")
        if set(self.execution_channels) != {
            GenerationChannel.EVIDENCE,
            GenerationChannel.CRITERION,
        }:
            raise ValueError("Sandbox rehearsal may execute only evidence and criterion channels")
        if set(self.required_channels) != {
            GenerationChannel.PUBLIC,
            GenerationChannel.EVIDENCE,
            GenerationChannel.CRITERION,
        }:
            raise ValueError("Sandbox rehearsal requires each isolated channel for every case")
        return self


class SandboxRehearsalAttempt(ContractModel):
    """One recorded executable response and its observable isolated result."""

    case_id: str = Field(min_length=1)
    channel: Literal[GenerationChannel.EVIDENCE, GenerationChannel.CRITERION]
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_bundle_ref: str = Field(min_length=1)
    test_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_status: Literal["completed", "failed"]
    sandbox_id: str | None = None
    exit_code: int | None = None
    outcome_status: Literal["completed", "invalid_submission", "execution_error"] | None = None
    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None


class SandboxRehearsalReport(ContractModel):
    """Immutable M6.4 operational report, not an empirical benchmark result."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.sandbox_rehearsal_report.v1"] = (
        "benchmark.sandbox_rehearsal_report.v1"
    )
    run_id: str = Field(min_length=1)
    development_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recorded_responses_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rehearsal_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at_utc: datetime
    recorded_response_count: int = Field(ge=1)
    replayed_response_count: int = Field(ge=0)
    execution_count: int = Field(ge=0)
    completed_execution_count: int = Field(ge=0)
    passed_check_count: int = Field(ge=0)
    failed_check_count: int = Field(ge=0)
    attempts: tuple[SandboxRehearsalAttempt, ...]
    gate_passed: bool
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> SandboxRehearsalReport:
        timestamp_is_utc = (
            self.generated_at_utc.tzinfo is not None
            and self.generated_at_utc.utcoffset() == UTC.utcoffset(self.generated_at_utc)
        )
        if not timestamp_is_utc:
            raise ValueError("Sandbox rehearsal timestamp must be UTC")
        if self.execution_count != len(self.attempts):
            raise ValueError("Sandbox rehearsal execution count does not match attempts")
        if self.completed_execution_count != sum(
            attempt.execution_status == "completed" for attempt in self.attempts
        ):
            raise ValueError("Sandbox rehearsal completed count does not match attempts")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Sandbox rehearsal report hash does not match its content")
        return self


def load_sandbox_rehearsal_plan(path: Path) -> SandboxRehearsalPlan:
    """Load explicit development-only configuration."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read sandbox rehearsal plan: {path}") from error
    return SandboxRehearsalPlan.model_validate(raw)


def run_recorded_sandbox_rehearsal(
    *,
    plan: SandboxRehearsalPlan,
    manifest_path: Path,
    recorded_responses_path: Path,
    output_root: Path,
    run_id: str,
    executor: SandboxExecutor,
    generated_at_utc: datetime | None = None,
) -> SandboxRehearsalReport:
    """Replay records exactly, then execute only their code-bearing channels remotely."""

    now = generated_at_utc or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
        raise ValueError("Sandbox rehearsal time must be UTC")
    authored = load_and_verify_manifest(manifest_path)
    if authored.benchmark_version != "dev-v0" or authored.status.value != "draft":
        raise ValueError("Sandbox rehearsal accepts only the draft development benchmark")
    root = output_root.resolve()
    records = _load_records(recorded_responses_path)
    expected_keys = {
        (case_id, channel)
        for case_id in plan.permitted_case_ids
        for channel in plan.required_channels
    }
    observed_keys = {(record.request.case_id, record.request.channel) for record in records}
    if observed_keys != expected_keys:
        raise ValueError(
            "Recorded responses do not match the required development channel coverage"
        )
    if len(records) != len(expected_keys):
        raise ValueError(
            "Recorded responses must contain exactly one response per development channel"
        )
    for record in records:
        if record.original_source is not OriginalGenerationSource.CEREBRAS:
            raise ValueError("Sandbox rehearsal accepts only pinned external-model recordings")
        replay_recorded_response(record.request, record)

    cases = {case.public.case_id: case for case in authored.cases}
    attempts: list[SandboxRehearsalAttempt] = []
    for record in records:
        if record.request.channel not in plan.execution_channels:
            continue
        case = cases[record.request.case_id]
        test_ref = (
            case.public.evidence_test_ref
            if record.request.channel is GenerationChannel.EVIDENCE
            else case.criterion.test_bundle_ref
        )
        result = execute_authored_python_tests(
            executor,
            response=record.final_response,
            bundle_path=manifest_path.parent / test_ref,
        )
        outcome = result.outcome
        attempts.append(
            SandboxRehearsalAttempt(
                case_id=record.request.case_id,
                channel=cast(
                    Literal[GenerationChannel.EVIDENCE, GenerationChannel.CRITERION],
                    record.request.channel,
                ),
                response_hash=record.response_hash,
                test_bundle_ref=test_ref,
                test_bundle_sha256=file_sha256((manifest_path.parent / test_ref).read_bytes()),
                execution_status=result.execution.status,
                sandbox_id=result.execution.sandbox_id,
                exit_code=result.execution.exit_code,
                outcome_status=outcome.status if outcome is not None else None,
                passed=outcome.passed if outcome is not None else None,
                failed=outcome.failed if outcome is not None else None,
                error_type=result.execution.error_type,
                error_message=result.execution.error_message,
            )
        )

    content = {
        "run_id": run_id,
        "development_manifest_hash": authored.manifest_hash
        or model_content_hash(authored, exclude={"manifest_hash"}),
        "recorded_responses_sha256": file_sha256(recorded_responses_path.read_bytes()),
        "rehearsal_plan_hash": model_content_hash(plan),
        "generated_at_utc": now,
        "recorded_response_count": len(records),
        "replayed_response_count": len(records),
        "execution_count": len(attempts),
        "completed_execution_count": sum(item.execution_status == "completed" for item in attempts),
        "passed_check_count": sum(item.passed or 0 for item in attempts),
        "failed_check_count": sum(item.failed or 0 for item in attempts),
        "attempts": tuple(attempts),
        "gate_passed": len(attempts) == 4
        and all(
            item.execution_status == "completed" and item.outcome_status == "completed"
            for item in attempts
        ),
    }
    draft = SandboxRehearsalReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = SandboxRehearsalReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(root / "sandbox_rehearsal_report.json", report)
    return report


def run_recorded_sandbox_rehearsal_with_modal(
    *,
    plan: SandboxRehearsalPlan,
    manifest_path: Path,
    recorded_responses_path: Path,
    output_root: Path,
    run_id: str,
) -> SandboxRehearsalReport:
    """Use the only permitted executor for externally generated Python."""

    return run_recorded_sandbox_rehearsal(
        plan=plan,
        manifest_path=manifest_path,
        recorded_responses_path=recorded_responses_path,
        output_root=output_root,
        run_id=run_id,
        executor=ModalSandboxExecutor(),
    )


def _load_records(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read recorded response file: {path}") from error
    records: list[RecordedGenerationResponse] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(RecordedGenerationResponse.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid recorded response at line {line_number}: {error}") from error
    if not records:
        raise ValueError("Recorded response file is empty")
    return tuple(records)
