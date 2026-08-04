"""Held-out external decision sealing without criterion access."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.external_seal import (
    ExternalDecisionSealPlan,
    run_external_decision_seal,
)
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.public.requests import PublicChannelRequestBuilder
from socratic_tutor.benchmark.public_rating import FinalPublicAnswerRating
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    RecordedGenerationResponse,
    create_recorded_response,
)
from socratic_tutor.contracts import EvidenceCategory
from socratic_tutor.sandbox import SandboxExecutionRequest, SandboxExecutionResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST = BENCHMARK_ROOT / "manifest.yaml"
PROJECTED_AT = datetime(2026, 8, 23, 1, 15, tzinfo=UTC)
PLAN_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
RUN_ID = "external-seal-unit-001"
ROUTE = ModelRoute(provider="cerebras", model="gpt-oss-120b")


class FakeExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        self.calls += 1
        return SandboxExecutionResult(
            status="completed",
            sandbox_id=f"sb-unit-{self.calls}",
            exit_code=0,
            stdout='{"status":"completed","passed":1,"failed":0}',
        )


class IncrementingClock:
    def __init__(self) -> None:
        self.current = PLAN_AT

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


def test_seals_all_external_decisions_and_resumes_without_criterion_access(
    tmp_path: Path,
) -> None:
    authored = load_and_verify_manifest(MANIFEST)
    public = project_public_manifest(authored, projected_at_utc=PROJECTED_AT)
    records, ratings = _records_and_ratings(public)
    executor = FakeExecutor()
    clock = IncrementingClock()

    report = run_external_decision_seal(
        plan=_plan(public),
        manifest=public,
        benchmark_root=BENCHMARK_ROOT,
        records=records,
        final_ratings=ratings,
        output_root=tmp_path,
        executor=executor,
        clock=clock,
    )

    assert report.gate_passed
    assert report.all_cases_complete
    assert report.planned_case_count == 24
    assert report.complete_case_count == 24
    assert report.precriterion_missing_case_count == 0
    assert report.invalid_case_count == 0
    assert report.evidence_execution_count == 24
    assert report.usable_evidence_count == 24
    assert report.prediction_count == 96
    assert executor.calls == 24
    assert (tmp_path / "decision" / "commitment_manifest.json").exists()
    assert (tmp_path / "datasets" / "condition_predictions.parquet").exists()
    assert not (tmp_path / "evaluator_manifest.json").exists()
    assert not (tmp_path / "datasets" / "criterion_outcomes.parquet").exists()

    retried = run_external_decision_seal(
        plan=_plan(public),
        manifest=public,
        benchmark_root=BENCHMARK_ROOT,
        records=records,
        final_ratings=ratings,
        output_root=tmp_path,
        executor=executor,
        clock=clock,
    )

    assert retried == report
    assert executor.calls == 24


def _plan(public: PublicBenchmarkManifest) -> ExternalDecisionSealPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_seal_plan.v1",
        "run_id": RUN_ID,
        "benchmark_version": "v1",
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "protocol_hash": "1" * 64,
        "preflight_hash": "2" * 64,
        "generation_report_hash": "3" * 64,
        "recorded_responses_hash": "4" * 64,
        "public_rating_report_hash": "5" * 64,
        "methodology_clarification_hash": "6" * 64,
        "inferential_hierarchy_hash": "7" * 64,
        "analysis_specification_hash": "8" * 64,
        "calibration_decision_hash": "9" * 64,
        "model_route": ROUTE,
        "tracker_version": "simple-v1",
        "policy_version": "heuristic-v1",
        "system_prompt_version": "external-evaluation-system-v1",
        "system_prompt_sha256": "a" * 64,
        "root_seed": 20260823,
        "code_revision": "external-seal-unit-test",
        "dirty_worktree": False,
        "pixi_lock_hash": "b" * 64,
        "created_at_utc": PLAN_AT,
    }
    draft = ExternalDecisionSealPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ExternalDecisionSealPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _records_and_ratings(
    public: PublicBenchmarkManifest,
) -> tuple[tuple[RecordedGenerationResponse, ...], tuple[FinalPublicAnswerRating, ...]]:
    builder = PublicChannelRequestBuilder(BENCHMARK_ROOT, public)
    records: list[RecordedGenerationResponse] = []
    ratings: list[FinalPublicAnswerRating] = []
    system_prompt = "Unit-test system prompt."
    evidence_response = "```python\ndef submitted_solution(*args, **kwargs):\n    return True\n```"
    provider_metadata = ProviderResponseMetadata(
        provider_id=ROUTE.provider,
        model_id=ROUTE.model,
        resolved_provider_id=ROUTE.provider,
        resolved_model_id=ROUTE.model,
    )
    for ordinal, case in enumerate(public.cases):
        sample_id = f"{RUN_ID}:{case.case_id}:001"
        spec = GenerationRequestSpec(
            run_id=RUN_ID,
            sample_id=sample_id,
            system_prompt_version="external-evaluation-system-v1",
            system_prompt=system_prompt,
            system_prompt_sha256=file_sha256(system_prompt.encode("utf-8")),
            model_route=ROUTE,
            sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=ordinal),
        )
        public_record = create_recorded_response(
            request=builder.build_public(case_id=case.case_id, spec=spec),
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response="A visible public answer for independent rating.",
            provider_metadata=provider_metadata,
            captured_at_utc=PROJECTED_AT + timedelta(minutes=ordinal),
        )
        evidence_record = create_recorded_response(
            request=builder.build_evidence(case_id=case.case_id, spec=spec),
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response=evidence_response,
            provider_metadata=provider_metadata,
            captured_at_utc=PROJECTED_AT + timedelta(minutes=ordinal),
        )
        records.extend((public_record, evidence_record))
        ratings.append(
            FinalPublicAnswerRating(
                case_id=case.case_id,
                response_hash=public_record.response_hash,
                category=EvidenceCategory.CORRECT,
                confidence=0.95,
                resolution="agreement",
                source_hashes=(
                    canonical_sha256({"case": case.case_id, "rater": "a"}),
                    canonical_sha256({"case": case.case_id, "rater": "b"}),
                ),
            )
        )
    return tuple(records), tuple(ratings)
