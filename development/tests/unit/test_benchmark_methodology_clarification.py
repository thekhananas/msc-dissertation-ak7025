"""Tests for the pre-criterion methodology clarification."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.calibration import UncalibratedDecisionReport
from socratic_tutor.benchmark.evaluator.scoring import (
    CalibrationStatus,
    create_calibration_decision,
)
from socratic_tutor.benchmark.external_protocol import ExternalModelExecutionProtocol
from socratic_tutor.benchmark.methodology_clarification import (
    MethodologyClarificationPlan,
    freeze_methodology_clarification,
    load_methodology_clarification_plan,
)
from socratic_tutor.benchmark.public_rating import PublicAnswerRatingReport
from socratic_tutor.contracts import Evidence
from socratic_tutor.tracking.simple import initial_tracker_state, update_tracker

WORKSPACE_ROOT = Path(__file__).parents[2]
PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-methodology-clarification.yaml"
ANALYSIS = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _inputs():
    analysis = load_analysis_specification(ANALYSIS)
    specification_hash = analysis_specification_hash(analysis)
    decision = create_calibration_decision(
        status=CalibrationStatus.UNCALIBRATED_SCORE,
        policy_threshold=0.6,
        calibration_task_family_hash=HASH_A,
        calibration_data_hash=HASH_B,
        tracker_version="simple-v1",
        diagnostics_hash=HASH_C,
        decision_owner="researcher",
        decided_at_utc=datetime(2026, 8, 22, tzinfo=UTC),
    )
    calibration = UncalibratedDecisionReport.model_construct(
        analysis_specification_hash=specification_hash,
        decision=decision,
    )
    protocol = ExternalModelExecutionProtocol.model_construct(
        protocol_hash=HASH_A,
        analysis_specification_hash=specification_hash,
        source_manifest_hash=HASH_B,
        provider_id="test-provider",
        requested_model_id="test-model",
        repeats_per_case=1,
    )
    rating = PublicAnswerRatingReport.model_construct(item_count=24, report_hash=HASH_C)
    plan = MethodologyClarificationPlan(
        clarification_version="pre-criterion-v1",
        decision_owner="researcher",
        frozen_on=date(2026, 9, 1),
        external_protocol_hash=protocol.protocol_hash,
        analysis_specification_hash=specification_hash,
        calibration_decision_hash=decision.decision_hash,
        public_rating_report_hash=rating.report_hash,
        source_manifest_hash=protocol.source_manifest_hash,
    )
    return plan, protocol, analysis, calibration, rating


def _freeze(tmp_path: Path):
    plan, protocol, analysis, calibration, rating = _inputs()
    return freeze_methodology_clarification(
        plan=plan,
        protocol=protocol,
        analysis_specification=analysis,
        calibration_report=calibration,
        public_rating_report=rating,
        output_path=tmp_path / "methodology-clarification.json",
    )


def test_versioned_plan_is_valid_without_loading_local_artifacts() -> None:
    plan = load_methodology_clarification_plan(PLAN)

    assert plan.clarification_version == "pre-criterion-v1"
    assert plan.frozen_on == date(2026, 9, 1)


def test_freezes_exact_scope_tracker_and_claim_boundary(tmp_path: Path) -> None:
    clarification = _freeze(tmp_path)

    assert clarification.authored_case_count == 24
    assert clarification.model_runs_per_case == 1
    assert clarification.population == "one_pinned_external_evaluation_model_route"
    assert clarification.tracker_kind == "authored_additive_heuristic"
    assert clarification.tracker_prior == 0.5
    assert clarification.policy_threshold == 0.6
    assert clarification.evidence_confidence_consumed is False
    assert clarification.sufficient_to_unlock_criterion_access is False
    assert "human_learning_improvement" in clarification.prohibited_claims
    assert "reduced_cognitive_offloading" in clarification.prohibited_claims


def test_declared_tracker_deltas_match_simple_v1_behavior(tmp_path: Path) -> None:
    clarification = _freeze(tmp_path)
    declared = {item.category: item.delta for item in clarification.tracker_category_deltas}
    initial = initial_tracker_state("test-concept")

    for category, delta in declared.items():
        evidence = Evidence(
            category=category,
            confidence=0.01,
            rationale="Parity fixture.",
            observed_signals=("fixture",),
        )
        updated = update_tracker(initial, evidence)
        assert abs(updated.mastery_probability - (0.5 + delta)) < 1e-12


def test_rejects_a_different_rating_report(tmp_path: Path) -> None:
    plan, protocol, analysis, calibration, rating = _inputs()
    changed = plan.model_copy(update={"public_rating_report_hash": HASH_D})

    with pytest.raises(ValueError, match="different public rating report"):
        freeze_methodology_clarification(
            plan=changed,
            protocol=protocol,
            analysis_specification=analysis,
            calibration_report=calibration,
            public_rating_report=rating,
            output_path=tmp_path / "invalid.json",
        )
