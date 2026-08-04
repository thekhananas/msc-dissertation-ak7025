"""Network-free lifecycle tests for post-seal external criterion execution."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from socratic_tutor.benchmark.cerebras import (
    CerebrasUnavailableError,
    GatewayGenerationResult,
)
from socratic_tutor.benchmark.evaluator.criterion import CriterionRecord
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.external_audit import ExternalDecisionAccountingReport
from socratic_tutor.benchmark.external_criterion import (
    ExternalCriterionPlan,
    ExternalCriterionRequestArtifact,
    run_external_criterion,
)
from socratic_tutor.benchmark.external_criterion_audit import (
    ExternalCriterionAuditError,
    audit_external_criterion_rows,
)
from socratic_tutor.benchmark.external_replay import (
    replay_published_datasets,
    replay_recorded_collection,
)
from socratic_tutor.benchmark.generation import (
    GenerationChannel,
    ModelRoute,
    SamplingConfig,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.global_seal import FilesystemGlobalDecisionSealStore
from socratic_tutor.benchmark.public.offline import (
    OfflineDecisionCase,
    OfflineDecisionPlan,
    run_offline_decision_phase,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    RecordedGenerationResponse,
    create_recorded_response,
)
from socratic_tutor.contracts import Evidence, EvidenceCategory
from socratic_tutor.sandbox import SandboxExecutionRequest, SandboxExecutionResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST = BENCHMARK_ROOT / "manifest.yaml"
PROMPT = BENCHMARK_ROOT / "prompts" / "student-system-v1.md"
RUN_ID = "external-transfer-unit-001"
DECISION_AT = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)
CRITERION_AT = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
ROUTE = ModelRoute(provider="recorded_fixture", model="authored-deterministic")


class FakeGateway:
    def __init__(self, *, fail_case_id: str | None = None) -> None:
        self.fail_case_id = fail_case_id
        self.requests: list[StudentGenerationRequest] = []

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult:
        self.requests.append(request)
        if request.case_id == self.fail_case_id:
            raise CerebrasUnavailableError("planned criterion provider failure")
        response = create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response="```python\ndef submitted_solution():\n    return True\n```",
            provider_metadata=ProviderResponseMetadata(
                provider_id=ROUTE.provider,
                model_id=ROUTE.model,
                resolved_provider_id=ROUTE.provider,
                resolved_model_id=ROUTE.model,
                backend_fingerprint="fp-criterion-unit",
                latency_ms=5,
            ),
            captured_at_utc=CRITERION_AT,
        )
        content = {"response": response, "attempt_count": 1, "retried_status_codes": ()}
        draft = GatewayGenerationResult.model_construct(
            _fields_set=set(content), **content, invocation_hash="0" * 64
        )
        return GatewayGenerationResult.model_validate(
            {
                **content,
                "invocation_hash": model_content_hash(draft, exclude={"invocation_hash"}),
            }
        )


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[SandboxExecutionRequest] = []

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        self.requests.append(request)
        return SandboxExecutionResult(
            status="completed",
            sandbox_id=f"sb-criterion-{len(self.requests)}",
            exit_code=0,
            stdout='{"status":"completed","passed":1,"failed":0}',
        )


class IncrementingClock:
    def __init__(self) -> None:
        self.current = CRITERION_AT

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


async def _no_sleep(_: float) -> None:
    return None


def test_runs_criterion_once_after_seal_and_resumes_without_external_calls(
    tmp_path: Path,
) -> None:
    public, evaluator, plan = _sealed_fixture(tmp_path)
    gateway = FakeGateway()
    executor = FakeExecutor()

    report = asyncio.run(
        run_external_criterion(
            plan=plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=gateway,
            executor=executor,
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )

    assert report.gate_passed
    assert report.planned_case_count == 24
    assert report.provider_success_count == 24
    assert report.provider_missing_count == 0
    assert report.completed_execution_count == 24
    assert report.demonstrated_count == 24
    assert report.criterion_record_count == 24
    assert len(gateway.requests) == 24
    assert len(executor.requests) == 24
    assert (tmp_path / "datasets" / "criterion_records.parquet").exists()

    replay_gateway = FakeGateway()
    replay_executor = FakeExecutor()
    replay = asyncio.run(
        run_external_criterion(
            plan=plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=replay_gateway,
            executor=replay_executor,
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )
    assert replay == report
    assert replay_gateway.requests == []
    assert replay_executor.requests == []


def test_preserves_provider_failure_as_missing_instead_of_incorrect(tmp_path: Path) -> None:
    public, evaluator, plan = _sealed_fixture(tmp_path)
    failed_case = public.cases[0].case_id
    gateway = FakeGateway(fail_case_id=failed_case)
    executor = FakeExecutor()

    report = asyncio.run(
        run_external_criterion(
            plan=plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=gateway,
            executor=executor,
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )

    assert report.provider_success_count == 23
    assert report.provider_missing_count == 1
    assert report.completed_execution_count == 23
    assert report.criterion_record_count == 24
    assert len(executor.requests) == 23


def test_postcriterion_audit_accepts_exact_rows_and_rejects_route_drift(
    tmp_path: Path,
) -> None:
    public, evaluator, plan = _sealed_fixture(tmp_path)
    asyncio.run(
        run_external_criterion(
            plan=plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=FakeGateway(),
            executor=FakeExecutor(),
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )
    attempts = tuple(
        ExternalCriterionRequestArtifact.model_validate_json(path.read_bytes())
        for path in sorted((tmp_path / "criterion" / "attempts").glob("*.json"))
    )
    records = tuple(
        CriterionRecord.model_validate_json(path.read_bytes())
        for path in sorted((tmp_path / "criterion" / "records").glob("*.json"))
    )
    responses = tuple(
        RecordedGenerationResponse.model_validate_json(line)
        for line in (tmp_path / "criterion" / "recorded_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    condition_store = FilesystemConditionCommitStore(tmp_path / "decision")
    seal = FilesystemGlobalDecisionSealStore(tmp_path / "decision", condition_store).load()
    assert seal is not None
    accounting = ExternalDecisionAccountingReport.model_construct(
        report_hash=plan.accounting_report_hash
    )

    audit_external_criterion_rows(
        plan=plan,
        accounting=accounting,
        evaluator=evaluator,
        benchmark_root=BENCHMARK_ROOT,
        complete_statuses=seal.sample_statuses,
        attempts=attempts,
        records=records,
        responses=responses,
    )

    first = attempts[0]
    assert first.response is not None
    changed_metadata = first.response.provider_metadata.model_copy(
        update={"resolved_model_id": "another-model"}
    )
    changed_response = first.response.model_copy(update={"provider_metadata": changed_metadata})
    changed_attempt = first.model_copy(update={"response": changed_response})
    with pytest.raises(ExternalCriterionAuditError, match="provider identity"):
        audit_external_criterion_rows(
            plan=plan,
            accounting=accounting,
            evaluator=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            complete_statuses=seal.sample_statuses,
            attempts=(changed_attempt, *attempts[1:]),
            records=records,
            responses=responses,
        )


def test_network_free_replay_rebuilds_identical_preanalysis_datasets(
    tmp_path: Path,
) -> None:
    public, evaluator, plan = _sealed_fixture(tmp_path)
    asyncio.run(
        run_external_criterion(
            plan=plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=FakeGateway(),
            executor=FakeExecutor(),
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )
    responses = tuple(
        RecordedGenerationResponse.model_validate_json(line)
        for line in (tmp_path / "criterion" / "recorded_responses.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    replayed = asyncio.run(
        replay_recorded_collection(
            responses,
            allowed_channels=frozenset({GenerationChannel.CRITERION}),
        )
    )
    comparison = replay_published_datasets(
        seal_root=tmp_path,
        output_dataset_root=tmp_path / "replay" / "datasets",
    )

    assert replayed == 24
    assert (
        comparison["condition_source_publication_hash"]
        == comparison["condition_replay_publication_hash"]
    )
    assert (
        comparison["condition_source_parquet_hash"] == comparison["condition_replay_parquet_hash"]
    )
    assert (
        comparison["criterion_source_publication_hash"]
        == comparison["criterion_replay_publication_hash"]
    )
    assert (
        comparison["criterion_source_parquet_hash"] == comparison["criterion_replay_parquet_hash"]
    )


def _sealed_fixture(tmp_path: Path, *, calibration_decision_hash: str | None = None):
    authored = load_and_verify_manifest(MANIFEST)
    public = project_public_manifest(authored, projected_at_utc=DECISION_AT)
    evaluator = project_evaluator_manifest(authored, public, projected_at_utc=CRITERION_AT)
    prompt_hash = file_sha256(PROMPT.read_bytes())
    decision_plan = OfflineDecisionPlan(
        mode="reference_heldout",
        run_id=RUN_ID,
        sample_ids=("001",),
        model_route=ROUTE,
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=20260823),
        system_prompt_version="external-evaluation-system-v1",
        system_prompt_ref="prompts/student-system-v1.md",
        system_prompt_sha256=prompt_hash,
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
        code_revision="criterion-unit-decision",
        dirty_worktree=False,
        environment_lock_hash="1" * 64,
        calibration_decision_hash=calibration_decision_hash,
        root_seed=20260823,
        created_at_utc=DECISION_AT,
        cases=tuple(
            OfflineDecisionCase(
                case_id=case.case_id,
                public_response="The visible answer is uncertain.",
                evidence_response="```python\ndef submitted_solution():\n    return True\n```",
                public_evidence=Evidence(
                    category=EvidenceCategory.UNCERTAIN,
                    confidence=0.5,
                    rationale="Network-free later-outcome lifecycle fixture.",
                ),
                probe_passed=1,
                probe_failed=0,
            )
            for case in public.cases
        ),
    )
    summary = run_offline_decision_phase(
        manifest=public,
        benchmark_root=BENCHMARK_ROOT,
        plan=decision_plan,
        output_root=tmp_path,
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_plan.v1",
        "run_id": RUN_ID,
        "benchmark_version": "v1",
        "protocol_hash": "2" * 64,
        "accounting_report_hash": "3" * 64,
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "evaluator_projection_hash": evaluator.projection_hash,
        "global_seal_hash": summary.global_seal_hash,
        "expected_case_count": 24,
        "model_route": ROUTE,
        "system_prompt_version": "external-evaluation-system-v1",
        "system_prompt_sha256": prompt_hash,
        "temperature": 0.0,
        "max_output_tokens": 512,
        "root_seed": 20260823,
        "request_spacing_seconds": 0.0,
        "prior_request_count": 48,
        "request_budget": 72,
        "code_revision": "criterion-unit-runner",
        "pixi_lock_hash": "4" * 64,
        "created_at_utc": CRITERION_AT,
    }
    draft = ExternalCriterionPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    plan = ExternalCriterionPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )
    return public, evaluator, plan
