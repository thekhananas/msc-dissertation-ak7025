"""Post-hoc replay of frozen responses after a test-harness correction."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.criterion import CriterionRecord
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import AuthoredBenchmarkManifest, ManifestStatus
from socratic_tutor.benchmark.external_seal import ExternalEvidenceExecutionArtifact
from socratic_tutor.benchmark.generation import GenerationChannel
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
    replay_recorded_response,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.sandbox import (
    ModalSandboxExecutor,
    SandboxExecutor,
    SandboxTestResult,
    execute_authored_python_tests,
)


class HarnessCorrectionReplayError(ValueError):
    """A correction replay input or lineage check failed."""


class HarnessExecutionSnapshot(ContractModel):
    """Comparable outcome before or after the harness correction."""

    status: Literal[
        "completed",
        "invalid_submission",
        "execution_error",
        "provider_unavailable",
        "sandbox_unavailable",
        "sandbox_timeout",
        "sandbox_resource_limit",
        "sandbox_error",
    ]
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    demonstrated_performance: bool | None

    @model_validator(mode="after")
    def validate_snapshot(self) -> HarnessExecutionSnapshot:
        if self.status == "completed":
            if self.passed + self.failed == 0:
                raise ValueError("Completed harness outcome requires at least one check")
            if self.demonstrated_performance is not (self.failed == 0):
                raise ValueError("Demonstrated performance must follow completed checks")
        elif self.passed != 0 or self.failed != 0 or self.demonstrated_performance is not None:
            raise ValueError("Unavailable harness outcome cannot claim test performance")
        return self

    @property
    def terminal(self) -> bool:
        """Return whether code evaluation reached a meaningful terminal outcome."""

        return self.status in {"completed", "invalid_submission", "execution_error"}


class HarnessCorrectionReplayPlan(ContractModel):
    """Immutable source and code identity for one post-hoc replay."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_replay_plan.v1"] = (
        "benchmark.harness_correction_replay_plan.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_responses_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_responses_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_evidence_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_criterion_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pixi_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    harness_rule: Literal["recursive_list_tuple_sequence_equivalence"] = (
        "recursive_list_tuple_sequence_equivalence"
    )
    interpretation_scope: Literal["post_hoc_harness_correction_not_primary"] = (
        "post_hoc_harness_correction_not_primary"
    )
    code_revision: str = Field(min_length=1)
    created_at_utc: datetime
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_plan(self) -> HarnessCorrectionReplayPlan:
        _require_utc(self.created_at_utc, "Harness correction replay plan time")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Harness correction replay plan hash does not match its content")
        return self


class HarnessCorrectionAttempt(ContractModel):
    """One frozen response compared under the original and corrected harness."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_attempt.v1"] = (
        "benchmark.harness_correction_attempt.v1"
    )
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str = Field(min_length=1)
    channel: Literal[GenerationChannel.EVIDENCE, GenerationChannel.CRITERION]
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_bundle_ref: str = Field(min_length=1)
    test_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original: HarnessExecutionSnapshot
    corrected: HarnessExecutionSnapshot
    corrected_sandbox_id: str | None = None
    corrected_exit_code: int | None = None
    corrected_error_type: str | None = None
    corrected_error_message: str | None = None
    outcome_changed: bool
    replayed_at_utc: datetime
    attempt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_attempt(self) -> HarnessCorrectionAttempt:
        _require_utc(self.replayed_at_utc, "Harness correction attempt time")
        if self.outcome_changed is not (self.original != self.corrected):
            raise ValueError("Harness correction change flag does not match the snapshots")
        if self.attempt_hash != model_content_hash(self, exclude={"attempt_hash"}):
            raise ValueError("Harness correction attempt hash does not match its content")
        return self


class HarnessCorrectionReplayReport(ContractModel):
    """Accounting report for the isolated post-hoc correction replay."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_replay_report.v1"] = (
        "benchmark.harness_correction_replay_report.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    replay_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_execution_count: int = Field(ge=1)
    execution_count: int = Field(ge=0)
    terminal_execution_count: int = Field(ge=0)
    changed_outcome_count: int = Field(ge=0)
    evidence_changed_count: int = Field(ge=0)
    criterion_changed_count: int = Field(ge=0)
    original_demonstrated_count: int = Field(ge=0)
    corrected_demonstrated_count: int = Field(ge=0)
    newly_demonstrated_count: int = Field(ge=0)
    no_longer_demonstrated_count: int = Field(ge=0)
    provider_call_count: Literal[0] = 0
    interpretation_scope: Literal["post_hoc_harness_correction_not_primary"] = (
        "post_hoc_harness_correction_not_primary"
    )
    completed_at_utc: datetime
    attempts: tuple[HarnessCorrectionAttempt, ...]
    gate_passed: bool
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> HarnessCorrectionReplayReport:
        _require_utc(self.completed_at_utc, "Harness correction replay completion time")
        if self.execution_count != len(self.attempts):
            raise ValueError("Harness correction execution count does not match attempts")
        if self.terminal_execution_count != sum(
            attempt.corrected.terminal for attempt in self.attempts
        ):
            raise ValueError("Harness correction terminal count does not match attempts")
        if self.changed_outcome_count != sum(attempt.outcome_changed for attempt in self.attempts):
            raise ValueError("Harness correction changed count does not match attempts")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Harness correction replay report hash does not match its content")
        return self


type Clock = Callable[[], datetime]


def run_harness_correction_replay_from_files(
    *,
    manifest_path: Path,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    original_evidence_root: Path,
    original_criterion_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    code_revision: str,
    created_at_utc: datetime,
    executor: SandboxExecutor,
    clock: Clock | None = None,
) -> HarnessCorrectionReplayReport:
    """Verify frozen lineage and replay all stored code-bearing responses."""

    authored = load_and_verify_manifest(manifest_path)
    if authored.status is not ManifestStatus.FROZEN or authored.benchmark_version != "v1":
        raise HarnessCorrectionReplayError("Harness correction replay requires frozen benchmark v1")
    case_ids = {case.public.case_id for case in authored.cases}
    evidence_records = _load_channel_records(
        decision_responses_path,
        channel=GenerationChannel.EVIDENCE,
        expected_case_ids=case_ids,
        permitted_channels={GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE},
    )
    criterion_records = _load_channel_records(
        criterion_responses_path,
        channel=GenerationChannel.CRITERION,
        expected_case_ids=case_ids,
        permitted_channels={GenerationChannel.CRITERION},
    )
    original_evidence = _load_evidence_artifacts(original_evidence_root, case_ids)
    original_criterion = _load_criterion_artifacts(original_criterion_root, case_ids)
    plan = _create_plan(
        run_id=run_id,
        source_manifest_hash=authored.manifest_hash
        or model_content_hash(authored, exclude={"manifest_hash"}),
        decision_responses_path=decision_responses_path,
        criterion_responses_path=criterion_responses_path,
        evidence_artifacts=original_evidence,
        criterion_artifacts=original_criterion,
        pixi_lock_path=pixi_lock_path,
        code_revision=code_revision,
        created_at_utc=created_at_utc,
    )
    return run_harness_correction_replay(
        plan=plan,
        authored_manifest=authored,
        benchmark_root=manifest_path.resolve().parent,
        evidence_records=evidence_records,
        criterion_records=criterion_records,
        original_evidence=original_evidence,
        original_criterion=original_criterion,
        output_root=output_root,
        executor=executor,
        clock=clock,
    )


def run_harness_correction_replay_with_modal(
    *,
    manifest_path: Path,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    original_evidence_root: Path,
    original_criterion_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    code_revision: str,
    created_at_utc: datetime,
) -> HarnessCorrectionReplayReport:
    """Replay frozen code using the required isolated Modal executor."""

    return run_harness_correction_replay_from_files(
        manifest_path=manifest_path,
        decision_responses_path=decision_responses_path,
        criterion_responses_path=criterion_responses_path,
        original_evidence_root=original_evidence_root,
        original_criterion_root=original_criterion_root,
        pixi_lock_path=pixi_lock_path,
        output_root=output_root,
        run_id=run_id,
        code_revision=code_revision,
        created_at_utc=created_at_utc,
        executor=ModalSandboxExecutor(),
    )


def run_harness_correction_replay(
    *,
    plan: HarnessCorrectionReplayPlan,
    authored_manifest: AuthoredBenchmarkManifest,
    benchmark_root: Path,
    evidence_records: dict[str, RecordedGenerationResponse],
    criterion_records: dict[str, RecordedGenerationResponse],
    original_evidence: dict[str, ExternalEvidenceExecutionArtifact],
    original_criterion: dict[str, CriterionRecord],
    output_root: Path,
    executor: SandboxExecutor,
    clock: Clock | None = None,
) -> HarnessCorrectionReplayReport:
    """Execute a resumable correction replay without provider access."""

    cases = authored_manifest.cases
    now = clock or (lambda: datetime.now(UTC))
    root = output_root.resolve()
    write_immutable_json(root / "harness_correction_replay_plan.json", plan)
    report_path = root / "harness_correction_replay_report.json"
    if report_path.exists():
        report = _load_json_model(report_path, HarnessCorrectionReplayReport)
        if report.run_id != plan.run_id or report.replay_plan_hash != plan.plan_hash:
            raise HarnessCorrectionReplayError("Existing correction report belongs to another plan")
        return report

    attempts: list[HarnessCorrectionAttempt] = []
    for case in cases:
        public = case.public
        criterion = case.criterion
        case_id = public.case_id
        channel_inputs = (
            (
                GenerationChannel.EVIDENCE,
                evidence_records[case_id],
                public.evidence_test_ref,
                original_evidence[case_id],
            ),
            (
                GenerationChannel.CRITERION,
                criterion_records[case_id],
                criterion.test_bundle_ref,
                original_criterion[case_id],
            ),
        )
        for channel, response, test_ref, original_artifact in channel_inputs:
            attempts.append(
                _load_or_execute_attempt(
                    root=root,
                    plan=plan,
                    case_id=case_id,
                    channel=channel,
                    response=response,
                    test_ref=test_ref,
                    benchmark_root=benchmark_root,
                    original_artifact=original_artifact,
                    executor=executor,
                    clock=now,
                )
            )

    attempts.sort(key=lambda item: (item.case_id, item.channel.value))
    expected_count = len(cases) * 2
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_replay_report.v1",
        "run_id": plan.run_id,
        "benchmark_version": "v1",
        "replay_plan_hash": plan.plan_hash,
        "expected_execution_count": expected_count,
        "execution_count": len(attempts),
        "terminal_execution_count": sum(item.corrected.terminal for item in attempts),
        "changed_outcome_count": sum(item.outcome_changed for item in attempts),
        "evidence_changed_count": sum(
            item.outcome_changed and item.channel is GenerationChannel.EVIDENCE for item in attempts
        ),
        "criterion_changed_count": sum(
            item.outcome_changed and item.channel is GenerationChannel.CRITERION
            for item in attempts
        ),
        "original_demonstrated_count": sum(
            item.original.demonstrated_performance is True for item in attempts
        ),
        "corrected_demonstrated_count": sum(
            item.corrected.demonstrated_performance is True for item in attempts
        ),
        "newly_demonstrated_count": sum(
            item.original.demonstrated_performance is not True
            and item.corrected.demonstrated_performance is True
            for item in attempts
        ),
        "no_longer_demonstrated_count": sum(
            item.original.demonstrated_performance is True
            and item.corrected.demonstrated_performance is not True
            for item in attempts
        ),
        "provider_call_count": 0,
        "interpretation_scope": "post_hoc_harness_correction_not_primary",
        "completed_at_utc": _require_utc(now(), "Harness correction completion time"),
        "attempts": tuple(attempts),
        "gate_passed": len(attempts) == expected_count
        and all(item.corrected.terminal for item in attempts),
    }
    draft = HarnessCorrectionReplayReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = HarnessCorrectionReplayReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _create_plan(
    *,
    run_id: str,
    source_manifest_hash: str,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    evidence_artifacts: dict[str, ExternalEvidenceExecutionArtifact],
    criterion_artifacts: dict[str, CriterionRecord],
    pixi_lock_path: Path,
    code_revision: str,
    created_at_utc: datetime,
) -> HarnessCorrectionReplayPlan:
    try:
        decision_bytes = decision_responses_path.read_bytes()
        criterion_bytes = criterion_responses_path.read_bytes()
        pixi_bytes = pixi_lock_path.read_bytes()
    except OSError as error:
        raise HarnessCorrectionReplayError("Could not read correction replay provenance") from error
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_replay_plan.v1",
        "run_id": run_id,
        "benchmark_version": "v1",
        "source_manifest_hash": source_manifest_hash,
        "decision_responses_sha256": file_sha256(decision_bytes),
        "criterion_responses_sha256": file_sha256(criterion_bytes),
        "original_evidence_root_hash": canonical_sha256(
            tuple(
                (case_id, artifact.artifact_hash)
                for case_id, artifact in sorted(evidence_artifacts.items())
            )
        ),
        "original_criterion_root_hash": canonical_sha256(
            tuple(
                (case_id, artifact.record_hash)
                for case_id, artifact in sorted(criterion_artifacts.items())
            )
        ),
        "pixi_lock_sha256": file_sha256(pixi_bytes),
        "harness_rule": "recursive_list_tuple_sequence_equivalence",
        "interpretation_scope": "post_hoc_harness_correction_not_primary",
        "code_revision": code_revision,
        "created_at_utc": _require_utc(created_at_utc, "Harness correction replay plan time"),
    }
    draft = HarnessCorrectionReplayPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return HarnessCorrectionReplayPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _load_or_execute_attempt(
    *,
    root: Path,
    plan: HarnessCorrectionReplayPlan,
    case_id: str,
    channel: GenerationChannel,
    response: RecordedGenerationResponse,
    test_ref: str,
    benchmark_root: Path,
    original_artifact: ExternalEvidenceExecutionArtifact | CriterionRecord,
    executor: SandboxExecutor,
    clock: Clock,
) -> HarnessCorrectionAttempt:
    test_path = benchmark_root / test_ref
    try:
        test_hash = file_sha256(test_path.read_bytes())
    except OSError as error:
        raise HarnessCorrectionReplayError(f"Could not read test bundle: {test_ref}") from error
    original_hash, original = _original_snapshot(
        channel=channel,
        artifact=original_artifact,
        response=response,
        test_ref=test_ref,
        test_hash=test_hash,
    )
    path = root / "attempts" / channel.value / f"{response.response_hash}.json"
    if path.exists():
        attempt = _load_json_model(path, HarnessCorrectionAttempt)
        if (
            attempt.plan_hash != plan.plan_hash
            or attempt.case_id != case_id
            or attempt.channel is not channel
            or attempt.response_hash != response.response_hash
            or attempt.test_bundle_sha256 != test_hash
            or attempt.original_artifact_hash != original_hash
        ):
            raise HarnessCorrectionReplayError(
                f"Existing correction attempt belongs to another input: {case_id}/{channel.value}"
            )
        return attempt
    result = execute_authored_python_tests(
        executor,
        response=response.final_response,
        bundle_path=test_path,
    )
    corrected = _corrected_snapshot(result)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_attempt.v1",
        "plan_hash": plan.plan_hash,
        "case_id": case_id,
        "channel": channel,
        "response_hash": response.response_hash,
        "test_bundle_ref": test_ref,
        "test_bundle_sha256": test_hash,
        "original_artifact_hash": original_hash,
        "original": original,
        "corrected": corrected,
        "corrected_sandbox_id": result.execution.sandbox_id,
        "corrected_exit_code": result.execution.exit_code,
        "corrected_error_type": result.execution.error_type,
        "corrected_error_message": result.execution.error_message,
        "outcome_changed": original != corrected,
        "replayed_at_utc": _require_utc(clock(), "Harness correction attempt time"),
    }
    draft = HarnessCorrectionAttempt.model_construct(
        _fields_set=set(content), **content, attempt_hash="0" * 64
    )
    attempt = HarnessCorrectionAttempt.model_validate(
        {**content, "attempt_hash": model_content_hash(draft, exclude={"attempt_hash"})}
    )
    write_immutable_json(path, attempt)
    return attempt


def _original_snapshot(
    *,
    channel: GenerationChannel,
    artifact: ExternalEvidenceExecutionArtifact | CriterionRecord,
    response: RecordedGenerationResponse,
    test_ref: str,
    test_hash: str,
) -> tuple[str, HarnessExecutionSnapshot]:
    if channel is GenerationChannel.EVIDENCE:
        if not isinstance(artifact, ExternalEvidenceExecutionArtifact):
            raise HarnessCorrectionReplayError("Evidence channel has a criterion artifact")
        if (
            artifact.case_id != response.request.case_id
            or artifact.response_hash != response.response_hash
            or artifact.evidence_test_ref != test_ref
            or artifact.evidence_test_sha256 != test_hash
        ):
            raise HarnessCorrectionReplayError(
                f"Original evidence artifact has mismatched lineage: {response.request.case_id}"
            )
        status = artifact.outcome_status or "sandbox_error"
        passed = artifact.passed or 0
        failed = artifact.failed or 0
        demonstrated = failed == 0 if status == "completed" else None
        return artifact.artifact_hash, HarnessExecutionSnapshot(
            status=status,
            passed=passed,
            failed=failed,
            demonstrated_performance=demonstrated,
        )
    if not isinstance(artifact, CriterionRecord):
        raise HarnessCorrectionReplayError("Criterion channel has an evidence artifact")
    if (
        artifact.key.case_id != response.request.case_id
        or artifact.response_hash != response.response_hash
        or artifact.test_bundle_sha256 != test_hash
    ):
        raise HarnessCorrectionReplayError(
            f"Original criterion artifact has mismatched lineage: {response.request.case_id}"
        )
    execution = artifact.execution
    status = execution.status.value
    return artifact.record_hash, HarnessExecutionSnapshot(
        status=status,
        passed=execution.passed,
        failed=execution.failed,
        demonstrated_performance=artifact.demonstrated_performance,
    )


def _corrected_snapshot(result: SandboxTestResult) -> HarnessExecutionSnapshot:
    outcome = result.outcome
    if outcome is None:
        return HarnessExecutionSnapshot(
            status="sandbox_error",
            passed=0,
            failed=0,
            demonstrated_performance=None,
        )
    return HarnessExecutionSnapshot(
        status=outcome.status,
        passed=outcome.passed,
        failed=outcome.failed,
        demonstrated_performance=(outcome.failed == 0) if outcome.status == "completed" else None,
    )


def _load_channel_records(
    path: Path,
    *,
    channel: GenerationChannel,
    expected_case_ids: set[str],
    permitted_channels: set[GenerationChannel],
) -> dict[str, RecordedGenerationResponse]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise HarnessCorrectionReplayError(f"Could not read recorded responses: {path}") from error
    selected: dict[str, RecordedGenerationResponse] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = RecordedGenerationResponse.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise HarnessCorrectionReplayError(
                f"Invalid recorded response at line {line_number}: {path}"
            ) from error
        if record.request.channel not in permitted_channels:
            raise HarnessCorrectionReplayError(
                f"Unexpected {record.request.channel.value} response in {path}"
            )
        if record.original_source is not OriginalGenerationSource.CEREBRAS:
            raise HarnessCorrectionReplayError(
                "Correction replay accepts only frozen provider records"
            )
        replay_recorded_response(record.request, record)
        if record.request.channel is channel:
            if record.request.case_id in selected:
                raise HarnessCorrectionReplayError(
                    f"Duplicate {channel.value} response: {record.request.case_id}"
                )
            selected[record.request.case_id] = record
    if set(selected) != expected_case_ids:
        raise HarnessCorrectionReplayError(
            f"{channel.value.title()} responses do not cover every frozen case exactly once"
        )
    return selected


def _load_evidence_artifacts(
    root: Path, expected_case_ids: set[str]
) -> dict[str, ExternalEvidenceExecutionArtifact]:
    artifacts = tuple(
        _load_json_model(path, ExternalEvidenceExecutionArtifact)
        for path in sorted(root.glob("*.json"))
    )
    indexed = _index_by_case(artifacts, lambda item: item.case_id, "evidence artifact")
    if set(indexed) != expected_case_ids:
        raise HarnessCorrectionReplayError(
            "Original evidence artifacts do not cover every frozen case exactly once"
        )
    return indexed


def _load_criterion_artifacts(
    root: Path, expected_case_ids: set[str]
) -> dict[str, CriterionRecord]:
    artifacts = tuple(
        _load_json_model(path, CriterionRecord) for path in sorted(root.glob("*.json"))
    )
    indexed = _index_by_case(artifacts, lambda item: item.key.case_id, "criterion artifact")
    if set(indexed) != expected_case_ids:
        raise HarnessCorrectionReplayError(
            "Original criterion artifacts do not cover every frozen case exactly once"
        )
    return indexed


def _index_by_case[ModelT](
    values: tuple[ModelT, ...], key: Callable[[ModelT], str], name: str
) -> dict[str, ModelT]:
    indexed: dict[str, ModelT] = {}
    for value in values:
        case_id = key(value)
        if case_id in indexed:
            raise HarnessCorrectionReplayError(f"Duplicate {name}: {case_id}")
        indexed[case_id] = value
    return indexed


def _load_json_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise HarnessCorrectionReplayError(
            f"Could not verify immutable artifact: {path}"
        ) from error


def _require_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must use UTC")
    return value
