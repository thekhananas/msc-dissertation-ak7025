# pyright: reportPrivateUsage=false
"""Network-free tests for the post-hoc harness correction analysis."""

from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.generation import GenerationChannel
from socratic_tutor.benchmark.harness_correction_analysis import (
    _case_effects,
    _reconstruct_cases,
)
from socratic_tutor.benchmark.harness_correction_replay import (
    HarnessCorrectionAttempt,
    HarnessCorrectionReplayReport,
    HarnessExecutionSnapshot,
)
from socratic_tutor.benchmark.methodology_clarification import MethodologyClarification
from socratic_tutor.benchmark.public.models import BenchmarkCondition
from socratic_tutor.benchmark.public_rating import (
    FinalPublicAnswerRating,
    PublicAnswerRatingReport,
)
from socratic_tutor.contracts import EvidenceCategory

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
CREATED_AT = datetime(2026, 9, 3, 14, 0, tzinfo=UTC)
HASH = "a" * 64


def test_case_effects_use_loss_of_comparator_minus_valid_evidence() -> None:
    assert _case_effects(1.0, 0.0, 1.0, 0.0) == (1.0, 1.0, 0.0, 0.5)
    assert _case_effects(0.0, 1.0, 0.0, 1.0) == (-1.0, -1.0, 0.0, -0.5)


def test_all_passing_probes_expose_absence_of_relevance_sensitivity() -> None:
    authored = load_and_verify_manifest(BENCHMARK_ROOT / "manifest.yaml")
    public = project_public_manifest(authored, projected_at_utc=CREATED_AT)
    ratings = PublicAnswerRatingReport.model_construct(
        item_count=24,
        report_hash=HASH,
        final_ratings=tuple(
            FinalPublicAnswerRating(
                case_id=case.case_id,
                response_hash=HASH,
                category=EvidenceCategory.CORRECT,
                confidence=1.0,
                resolution="agreement",
                source_hashes=("b" * 64, "c" * 64),
            )
            for case in public.cases
        ),
    )
    completed = HarnessExecutionSnapshot(
        status="completed",
        passed=1,
        failed=0,
        demonstrated_performance=True,
    )
    attempts: list[HarnessCorrectionAttempt] = []
    for ordinal, case in enumerate(public.cases):
        attempts.append(
            HarnessCorrectionAttempt.model_construct(
                case_id=case.case_id,
                channel=GenerationChannel.EVIDENCE,
                original=completed,
                corrected=completed,
                attempt_hash=HASH,
            )
        )
        criterion = (
            HarnessExecutionSnapshot(
                status="invalid_submission",
                passed=0,
                failed=0,
                demonstrated_performance=None,
            )
            if ordinal == 0
            else completed
        )
        attempts.append(
            HarnessCorrectionAttempt.model_construct(
                case_id=case.case_id,
                channel=GenerationChannel.CRITERION,
                original=criterion,
                corrected=criterion,
                attempt_hash=HASH,
            )
        )
    correction = HarnessCorrectionReplayReport.model_construct(attempts=tuple(attempts))
    methodology = MethodologyClarification.model_construct(
        tracker_version="simple-v1",
        policy_threshold=0.6,
    )

    results = _reconstruct_cases(
        correction=correction,
        public=public,
        ratings=ratings,
        methodology=methodology,
        benchmark_root=BENCHMARK_ROOT,
    )

    assert len(results) == 24
    assert sum(result.primary_effect is not None for result in results) == 23
    assert all(
        result.valid_vs_unrelated_effect == 0.0
        for result in results
        if result.primary_effect is not None
    )
    for result in results:
        scores = {score.condition: score for score in result.conditions}
        assert (
            scores[BenchmarkCondition.PROBE_INFORMED].mastery_score
            == scores[BenchmarkCondition.UNRELATED_PROBE].mastery_score
        )
