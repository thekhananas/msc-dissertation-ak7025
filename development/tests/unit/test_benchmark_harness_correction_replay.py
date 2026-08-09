"""Network-free tests for the post-hoc harness correction replay."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pytest

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.criterion import (
    CriterionExecutionStatus,
    CriterionRecord,
    create_normalized_criterion_execution,
    criterion_record_hash,
)
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.external_seal import ExternalEvidenceExecutionArtifact
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
    create_student_generation_request,
)
from socratic_tutor.benchmark.harness_correction_replay import (
    HarnessCorrectionReplayError,
    run_harness_correction_replay_from_files,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public import BenchmarkSampleKey
from socratic_tutor.benchmark.public.requests import PublicChannelRequestBuilder
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    RecordedGenerationResponse,
    create_recorded_response,
)
from socratic_tutor.sandbox import SandboxExecutionRequest, SandboxExecutionResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST = BENCHMARK_ROOT / "manifest.yaml"
PIXI_LOCK = WORKSPACE_ROOT / "pixi.lock"
RUN_ID = "harness-correction-unit-001"
ROUTE = ModelRoute(provider="cerebras", model="gpt-oss-120b")
CREATED_AT = datetime(2026, 9, 3, 14, 0, tzinfo=UTC)


class ReplayInputs(TypedDict):
    manifest_path: Path
    decision_responses_path: Path
    criterion_responses_path: Path
    original_evidence_root: Path
    original_criterion_root: Path
    pixi_lock_path: Path


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[SandboxExecutionRequest] = []

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        self.requests.append(request)
        return SandboxExecutionResult(
            status="completed",
            sandbox_id=f"sb-correction-{len(self.requests)}",
            exit_code=0,
            stdout='{"status":"completed","passed":1,"failed":0}',
        )


class IncrementingClock:
    def __init__(self) -> None:
        self.current = CREATED_AT

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


def test_replays_all_frozen_code_without_provider_calls_and_resumes(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    executor = FakeExecutor()
    clock = IncrementingClock()

    report = run_harness_correction_replay_from_files(
        **inputs,
        output_root=tmp_path / "correction",
        run_id=RUN_ID,
        code_revision="sequence-comparison-fix",
        created_at_utc=CREATED_AT,
        executor=executor,
        clock=clock,
    )

    assert report.gate_passed is True
    assert report.expected_execution_count == 48
    assert report.execution_count == 48
    assert report.terminal_execution_count == 48
    assert report.changed_outcome_count == 2
    assert report.evidence_changed_count == 1
    assert report.criterion_changed_count == 1
    assert report.original_demonstrated_count == 46
    assert report.corrected_demonstrated_count == 48
    assert report.newly_demonstrated_count == 2
    assert report.no_longer_demonstrated_count == 0
    assert report.provider_call_count == 0
    assert len(executor.requests) == 48

    retried = run_harness_correction_replay_from_files(
        **inputs,
        output_root=tmp_path / "correction",
        run_id=RUN_ID,
        code_revision="sequence-comparison-fix",
        created_at_utc=CREATED_AT,
        executor=executor,
        clock=clock,
    )

    assert retried == report
    assert len(executor.requests) == 48


def test_rejects_incomplete_original_artifact_coverage(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    first_artifact = next(inputs["original_evidence_root"].glob("*.json"))
    first_artifact.unlink()

    with pytest.raises(
        HarnessCorrectionReplayError,
        match="evidence artifacts do not cover every frozen case",
    ):
        run_harness_correction_replay_from_files(
            **inputs,
            output_root=tmp_path / "correction",
            run_id=RUN_ID,
            code_revision="sequence-comparison-fix",
            created_at_utc=CREATED_AT,
            executor=FakeExecutor(),
        )


def _write_inputs(tmp_path: Path) -> ReplayInputs:
    authored = load_and_verify_manifest(MANIFEST)
    public = project_public_manifest(authored, projected_at_utc=CREATED_AT)
    request_builder = PublicChannelRequestBuilder(BENCHMARK_ROOT, public)
    evidence_records: list[RecordedGenerationResponse] = []
    criterion_records: list[RecordedGenerationResponse] = []
    evidence_root = tmp_path / "original-evidence"
    criterion_root = tmp_path / "original-criterion"
    provider = ProviderResponseMetadata(provider_id=ROUTE.provider, model_id=ROUTE.model)
    source = "```python\ndef submitted_solution(*args, **kwargs):\n    return True\n```"

    for ordinal, case in enumerate(authored.cases):
        case_id = case.public.case_id
        sample_id = f"source-run:{case_id}:001"
        spec = GenerationRequestSpec(
            run_id="source-run",
            sample_id=sample_id,
            system_prompt_version="test-v1",
            system_prompt="Return one Python function.",
            system_prompt_sha256=file_sha256(b"Return one Python function."),
            model_route=ROUTE,
            sampling=SamplingConfig(temperature=0.0, max_output_tokens=256, seed=ordinal),
        )
        evidence = create_recorded_response(
            request=request_builder.build_evidence(case_id=case_id, spec=spec),
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response=source,
            provider_metadata=provider,
            captured_at_utc=CREATED_AT,
        )
        criterion_request = create_student_generation_request(
            spec=spec,
            payload=CriterionTaskPayload(
                benchmark_version="v1",
                case_id=case_id,
                criterion_probe_id=case.criterion.criterion_probe_id,
                criterion_probe="Complete the frozen transfer task.",
            ),
        )
        criterion = create_recorded_response(
            request=criterion_request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response=source,
            provider_metadata=provider,
            captured_at_utc=CREATED_AT,
        )
        evidence_records.append(evidence)
        criterion_records.append(criterion)
        old_failed = ordinal == 0
        _write_evidence_artifact(
            evidence_root,
            case_id=case_id,
            sample_id=sample_id,
            response=evidence,
            test_ref=case.public.evidence_test_ref,
            failed=old_failed,
        )
        _write_criterion_artifact(
            criterion_root,
            case_id=case_id,
            sample_id=sample_id,
            response=criterion,
            criterion_probe_id=case.criterion.criterion_probe_id,
            test_sha256=case.criterion.test_bundle_sha256,
            rubric_sha256=case.criterion.rubric_sha256,
            failed=old_failed,
        )

    decision_path = tmp_path / "decision_responses.jsonl"
    criterion_path = tmp_path / "criterion_responses.jsonl"
    decision_path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in evidence_records),
        encoding="utf-8",
    )
    criterion_path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in criterion_records),
        encoding="utf-8",
    )
    return {
        "manifest_path": MANIFEST,
        "decision_responses_path": decision_path,
        "criterion_responses_path": criterion_path,
        "original_evidence_root": evidence_root,
        "original_criterion_root": criterion_root,
        "pixi_lock_path": PIXI_LOCK,
    }


def _write_evidence_artifact(
    root: Path,
    *,
    case_id: str,
    sample_id: str,
    response: RecordedGenerationResponse,
    test_ref: str,
    failed: bool,
) -> None:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_evidence_execution.v1",
        "case_id": case_id,
        "sample_id": sample_id,
        "response_hash": response.response_hash,
        "evidence_test_ref": test_ref,
        "evidence_test_sha256": file_sha256((BENCHMARK_ROOT / test_ref).read_bytes()),
        "execution_status": "completed",
        "outcome_status": "completed",
        "passed": 0 if failed else 1,
        "failed": 1 if failed else 0,
        "sandbox_id": f"sb-original-evidence-{case_id}",
        "exit_code": 0,
        "executed_at_utc": CREATED_AT,
    }
    draft = ExternalEvidenceExecutionArtifact.model_construct(
        _fields_set=set(content), **content, artifact_hash="0" * 64
    )
    artifact = ExternalEvidenceExecutionArtifact.model_validate(
        {**content, "artifact_hash": model_content_hash(draft, exclude={"artifact_hash"})}
    )
    write_immutable_json(root / f"{response.response_hash}.json", artifact)


def _write_criterion_artifact(
    root: Path,
    *,
    case_id: str,
    sample_id: str,
    response: RecordedGenerationResponse,
    criterion_probe_id: str,
    test_sha256: str,
    rubric_sha256: str,
    failed: bool,
) -> None:
    execution = create_normalized_criterion_execution(
        status=CriterionExecutionStatus.COMPLETED,
        test_bundle_sha256=test_sha256,
        requested_at_utc=CREATED_AT,
        operation_id=f"sb-original-criterion-{case_id}",
        passed=0 if failed else 1,
        failed=1 if failed else 0,
        exit_code=0,
        stdout_sha256="a" * 64,
        stderr_sha256="b" * 64,
        executed_at_utc=CREATED_AT,
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.criterion_record.v1",
        "key": BenchmarkSampleKey(
            benchmark_version="v1",
            run_id="source-run",
            case_id=case_id,
            sample_id=sample_id,
            model_route_id=f"{ROUTE.provider}/{ROUTE.model}",
        ),
        "global_seal_hash": "c" * 64,
        "condition_set_hash": "d" * 64,
        "criterion_probe_id": criterion_probe_id,
        "request_hash": response.request.request_hash,
        "response_hash": response.response_hash,
        "test_bundle_sha256": test_sha256,
        "rubric_sha256": rubric_sha256,
        "execution": execution,
        "demonstrated_performance": not failed,
        "missing_reason": None,
        "revealed_at_utc": CREATED_AT,
    }
    draft = CriterionRecord.model_construct(
        _fields_set=set(content), **content, record_hash="0" * 64
    )
    record = CriterionRecord.model_validate(
        {**content, "record_hash": criterion_record_hash(draft)}
    )
    write_immutable_json(root / f"{record.record_hash}.json", record)
