"""Hand-calculated tests for the frozen evidence-specificity amendment."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from socratic_tutor.benchmark.analysis_spec import load_analysis_specification
from socratic_tutor.benchmark.calibration import (
    load_uncalibrated_decision_plan,
    record_uncalibrated_decision,
)
from socratic_tutor.benchmark.evaluator.aggregation import (
    CaseAggregate,
    CaseComparison,
    create_case_aggregate,
)
from socratic_tutor.benchmark.evidence_specificity import (
    analyze_evidence_specificity,
    create_external_run_preflight,
    freeze_evidence_specificity_amendment,
    load_evidence_specificity_amendment_plan,
)
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    freeze_external_model_execution_protocol,
    load_external_model_protocol_freeze_plan,
)
from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.benchmark.public_rating import (
    PublicAnswerRatingGuide,
    PublicRatingProtocolAmendment,
    freeze_public_rating_boundary,
    load_public_rating_guide_plan,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
V1_MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks"
ANALYSIS = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
CALIBRATION_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-uncalibrated-decision.yaml"
PROTOCOL_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-external-protocol-cerebras.yaml"
SPECIFICITY_PLAN = (
    WORKSPACE_ROOT / "configs" / "benchmark" / "v2-evidence-specificity-amendment.yaml"
)
RATING_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-public-answer-rating-guide.yaml"
SYSTEM_PROMPT = (
    WORKSPACE_ROOT
    / "data"
    / "benchmark-execution"
    / "v2"
    / "prompts"
    / "external-evaluation-system-v1.md"
)


def _frozen_inputs(tmp_path: Path):
    specification = load_analysis_specification(ANALYSIS)
    protocol = freeze_external_model_execution_protocol(
        plan=load_external_model_protocol_freeze_plan(PROTOCOL_PLAN),
        manifest_path=V1_MANIFEST,
        analysis_specification_path=ANALYSIS,
        system_prompt_path=SYSTEM_PROMPT,
        output_path=tmp_path / "protocol.json",
    )
    calibration = record_uncalibrated_decision(
        plan=load_uncalibrated_decision_plan(CALIBRATION_PLAN),
        heldout_manifest_path=V1_MANIFEST,
        benchmark_root=BENCHMARK_ROOT,
        analysis_specification_path=ANALYSIS,
        output_path=tmp_path / "calibration.json",
    )
    amendment = freeze_evidence_specificity_amendment(
        plan=load_evidence_specificity_amendment_plan(SPECIFICITY_PLAN),
        protocol=protocol,
        analysis_specification=specification,
        calibration_report=calibration,
        output_path=tmp_path / "specificity.json",
    )
    return specification, protocol, calibration, amendment


def _aggregate(
    *,
    case_id: str,
    comparison: CaseComparison,
    dialogue_loss: float | None,
    comparison_loss: float | None,
) -> CaseAggregate:
    eligible = int(dialogue_loss is not None and comparison_loss is not None)
    source_hashes = (
        ()
        if eligible == 0
        else (
            canonical_sha256(f"{case_id}:{comparison.value}:left"),
            canonical_sha256(f"{case_id}:{comparison.value}:right"),
        )
    )
    return create_case_aggregate(
        benchmark_version="v1",
        run_id="specificity-test",
        case_id=case_id,
        model_route_id="cerebras:gpt-oss-120b",
        comparison=comparison,
        eligible_repeat_count=eligible,
        planned_repeat_count=1,
        mean_left_loss=dialogue_loss,
        mean_right_loss=comparison_loss,
        missingness_rule="exclude_missing_criterion_and_report_denominator",
        aggregation_version="case-mean-v1",
        source_repeat_hashes=source_hashes,
        created_at_utc=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
    )


def _case_rows(
    case_id: str,
    *,
    dialogue: float | None,
    valid: float | None,
    unrelated: float | None,
    corrupted: float | None,
) -> tuple[CaseAggregate, ...]:
    return (
        _aggregate(
            case_id=case_id,
            comparison=CaseComparison.PRIMARY_VALID,
            dialogue_loss=dialogue,
            comparison_loss=valid,
        ),
        _aggregate(
            case_id=case_id,
            comparison=CaseComparison.UNRELATED_CONTROL,
            dialogue_loss=dialogue,
            comparison_loss=unrelated,
        ),
        _aggregate(
            case_id=case_id,
            comparison=CaseComparison.CORRUPTED_CONTROL,
            dialogue_loss=dialogue,
            comparison_loss=corrupted,
        ),
    )


def test_amendment_inherits_frozen_primary_rules_without_creating_a_new_gate(
    tmp_path: Path,
) -> None:
    specification, protocol, calibration, amendment = _frozen_inputs(tmp_path)

    assert amendment.base_external_protocol_hash == protocol.protocol_hash
    assert amendment.calibration_decision_hash == calibration.decision.decision_hash
    assert amendment.bootstrap_resamples == specification.bootstrap_resamples
    assert amendment.permutation_resamples == specification.permutation_resamples
    assert amendment.primary_minimum_interpretable_effect == 0.1
    assert amendment.secondary_role == "interpretation_only_cannot_rescue_primary"


def test_specificity_effects_match_hand_calculated_losses_and_exclude_missing_cases(
    tmp_path: Path,
) -> None:
    _, _, _, amendment = _frozen_inputs(tmp_path)
    aggregates = (
        *_case_rows(
            "case-001",
            dialogue=0.8,
            valid=0.2,
            unrelated=0.7,
            corrupted=0.6,
        ),
        *_case_rows(
            "case-002",
            dialogue=0.6,
            valid=0.4,
            unrelated=0.5,
            corrupted=0.3,
        ),
        *_case_rows(
            "case-missing",
            dialogue=None,
            valid=None,
            unrelated=None,
            corrupted=None,
        ),
    )

    report = analyze_evidence_specificity(
        aggregates=aggregates,
        amendment=amendment,
        output_path=tmp_path / "report.json",
    )

    first, second = report.case_effects
    assert abs(first.valid_vs_unrelated_effect - 0.5) < 1e-12
    assert abs(first.valid_vs_corrupted_effect - 0.4) < 1e-12
    assert abs(first.combined_specificity_effect - 0.45) < 1e-12
    assert abs(second.valid_vs_unrelated_effect - 0.1) < 1e-12
    assert abs(second.valid_vs_corrupted_effect - (-0.1)) < 1e-12
    assert abs(second.combined_specificity_effect) < 1e-12
    assert abs(report.combined_specificity_inference.mean_case_effect - 0.225) < 1e-12
    assert report.total_case_count == 3
    assert report.eligible_case_count == 2
    assert report.missing_case_count == 1
    assert report.interpretation == "valid_evidence_is_more_specific_than_controls"
    assert report.confirmatory_gate_allowed is False


def test_specificity_requires_all_three_predeclared_comparisons(tmp_path: Path) -> None:
    _, _, _, amendment = _frozen_inputs(tmp_path)
    incomplete = _case_rows(
        "case-001",
        dialogue=0.8,
        valid=0.2,
        unrelated=0.7,
        corrupted=0.6,
    )[:-1]

    with pytest.raises(ValueError, match="requires valid, unrelated, and corrupted"):
        analyze_evidence_specificity(
            aggregates=incomplete,
            amendment=amendment,
            output_path=tmp_path / "report.json",
        )


def test_external_preflight_requires_both_matching_amendments(tmp_path: Path) -> None:
    specification, protocol, calibration, specificity = _frozen_inputs(tmp_path)
    _, rating_amendment = _freeze_rating(protocol, tmp_path)

    preflight = create_external_run_preflight(
        protocol=protocol,
        analysis_specification=specification,
        calibration_report=calibration,
        public_rating_amendment=rating_amendment,
        evidence_specificity_amendment=specificity,
        output_path=tmp_path / "preflight.json",
    )

    assert preflight.gate_passed is True
    changed = specificity.model_copy(update={"base_external_protocol_hash": "f" * 64})
    with pytest.raises(ValueError, match="another external protocol"):
        create_external_run_preflight(
            protocol=protocol,
            analysis_specification=specification,
            calibration_report=calibration,
            public_rating_amendment=rating_amendment,
            evidence_specificity_amendment=changed,
            output_path=tmp_path / "invalid-preflight.json",
        )


def _freeze_rating(
    protocol: ExternalModelExecutionProtocol,
    tmp_path: Path,
) -> tuple[PublicAnswerRatingGuide, PublicRatingProtocolAmendment]:
    from socratic_tutor.benchmark.public_rating import (
        load_public_answer_rating_guide,
        load_public_rating_protocol_amendment,
    )

    freeze_public_rating_boundary(
        plan=load_public_rating_guide_plan(RATING_PLAN),
        protocol=protocol,
        guide_output_path=tmp_path / "rating-guide.json",
        amendment_output_path=tmp_path / "rating-amendment.json",
    )
    return (
        load_public_answer_rating_guide(tmp_path / "rating-guide.json"),
        load_public_rating_protocol_amendment(tmp_path / "rating-amendment.json"),
    )
