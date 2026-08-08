from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.aggregation import CaseComparison
from socratic_tutor.benchmark.evidence_specificity import (
    EvidenceSpecificityCaseEffect,
    EvidenceSpecificityReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.result_interpretation import (
    ResultInterpretationPlan,
    ResultInterpretationReport,
)
from socratic_tutor.benchmark.secondary_analysis import (
    SecondaryAnalysisReport,
    SecondaryComparisonSummary,
)
from socratic_tutor.benchmark.specificity_figure import (
    SpecificityFigureError,
    render_specificity_figure,
)
from socratic_tutor.benchmark.statistics import paired_case_inference


def test_renders_all_specificity_cases_and_retries_exactly(tmp_path: Path) -> None:
    secondary_path, plan_path, interpretation_path, pixi_lock = _write_sources(tmp_path)
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 16, 0, tzinfo=UTC)
    first = render_specificity_figure(
        secondary_report_path=secondary_path,
        interpretation_plan_path=plan_path,
        interpretation_report_path=interpretation_path,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
        generated_at_utc=generated_at,
    )
    retry = render_specificity_figure(
        secondary_report_path=secondary_path,
        interpretation_plan_path=plan_path,
        interpretation_report_path=interpretation_path,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
    )

    assert first == retry
    assert first.specificity_conclusion == "specific"
    assert not first.confirmatory_gate_allowed
    assert not first.primary_result_rescued
    assert (output_root / "evidence_specificity.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "evidence_specificity.svg").read_text(encoding="utf-8")
    assert "Secondary result: relevant evidence performed better" in svg
    assert "primary benchmark remains inconclusive" in svg
    csv = (output_root / "specificity_case_results.csv").read_text(encoding="utf-8")
    assert len(csv.splitlines()) == 25
    assert "h-c2m2-03,missing" in csv


def test_rejects_an_interpretation_from_another_environment(tmp_path: Path) -> None:
    secondary_path, plan_path, interpretation_path, pixi_lock = _write_sources(tmp_path)
    pixi_lock.write_text("different lock\n", encoding="utf-8")

    with pytest.raises(SpecificityFigureError, match="another Pixi lock"):
        render_specificity_figure(
            secondary_report_path=secondary_path,
            interpretation_plan_path=plan_path,
            interpretation_report_path=interpretation_path,
            pixi_lock_path=pixi_lock,
            output_root=tmp_path / "publication",
            publication_code_revision="a" * 40,
        )


def _write_sources(root: Path) -> tuple[Path, Path, Path, Path]:
    case_ids = tuple(
        f"h-c{concept}m{misconception}-{index:02}"
        for concept in range(1, 5)
        for misconception in range(1, 3)
        for index in range(1, 4)
    )
    missing_id = "h-c2m2-03"
    eligible_ids = tuple(case_id for case_id in case_ids if case_id != missing_id)
    contrast_effects = ((1.0, 1.0),) * 10 + ((0.0, 1.0),) * 10 + ((-1.0, -1.0),) * 3
    effects = tuple(
        _case_effect(case_id, unrelated, corrupted, index)
        for index, (case_id, (unrelated, corrupted)) in enumerate(
            zip(eligible_ids, contrast_effects, strict=True)
        )
    )
    unrelated_inference = paired_case_inference(
        tuple(row.valid_vs_unrelated_effect for row in effects),
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        minimum_interpretable_effect=0.1,
        random_seed=20260813,
    )
    corrupted_inference = paired_case_inference(
        tuple(row.valid_vs_corrupted_effect for row in effects),
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        minimum_interpretable_effect=0.1,
        random_seed=20260813,
    )
    combined_inference = paired_case_inference(
        tuple(row.combined_specificity_effect for row in effects),
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        minimum_interpretable_effect=0.1,
        random_seed=20260813,
    )
    specificity_content = {
        "schema_version": 1,
        "schema_id": "benchmark.evidence_specificity_report.v1",
        "amendment_hash": _sha(1),
        "total_case_count": 24,
        "eligible_case_count": 23,
        "missing_case_count": 1,
        "case_effects": effects,
        "valid_vs_unrelated_inference": unrelated_inference,
        "valid_vs_corrupted_inference": corrupted_inference,
        "combined_specificity_inference": combined_inference,
        "interpretation": "valid_evidence_is_more_specific_than_controls",
        "confirmatory_gate_allowed": False,
    }
    specificity_draft = EvidenceSpecificityReport.model_construct(
        _fields_set=set(specificity_content),
        **specificity_content,
        report_hash="0" * 64,
    )
    specificity = EvidenceSpecificityReport.model_validate(
        {
            **specificity_content,
            "report_hash": model_content_hash(specificity_draft, exclude={"report_hash"}),
        }
    )
    summaries = tuple(
        _comparison_summary(comparison, index, missing_id)
        for index, comparison in enumerate(CaseComparison)
    )
    completed_at = datetime(2026, 9, 3, 15, 0, tzinfo=UTC)
    secondary_content = {
        "schema_version": 1,
        "schema_id": "benchmark.secondary_analysis_report.v1",
        "benchmark_version": "v1",
        "run_id": "specificity-figure-test",
        "analysis_plan_hash": _sha(2),
        "primary_analysis_report_hash": _sha(3),
        "comparison_summaries": summaries,
        "evidence_specificity": specificity,
        "secondary_can_rescue_primary": False,
        "multiple_testing_role": "descriptive_secondary_not_confirmatory",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": completed_at,
    }
    secondary_draft = SecondaryAnalysisReport.model_construct(
        _fields_set=set(secondary_content),
        **secondary_content,
        report_hash="0" * 64,
    )
    secondary = SecondaryAnalysisReport.model_validate(
        {
            **secondary_content,
            "report_hash": model_content_hash(secondary_draft, exclude={"report_hash"}),
        }
    )

    pixi_lock = root / "pixi.lock"
    pixi_lock.write_text("locked\n", encoding="utf-8")
    plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.result_interpretation_plan.v1",
        "benchmark_version": "v1",
        "run_id": secondary.run_id,
        "primary_report_hash": secondary.primary_analysis_report_hash,
        "secondary_report_hash": secondary.report_hash,
        "probe_correction_plan_hash": _sha(4),
        "probe_correction_report_hash": _sha(5),
        "negative_result_audit_plan_hash": _sha(6),
        "negative_result_audit_report_hash": _sha(7),
        "analysis_code_revision": "9" * 40,
        "analysis_pixi_lock_hash": file_sha256(pixi_lock.read_bytes()),
        "record_owner": "dissertation_author",
        "created_at_utc": datetime(2026, 9, 3, 15, 10, tzinfo=UTC),
    }
    plan_draft = ResultInterpretationPlan.model_construct(
        _fields_set=set(plan_content),
        **plan_content,
        plan_hash="0" * 64,
    )
    plan = ResultInterpretationPlan.model_validate(
        {**plan_content, "plan_hash": model_content_hash(plan_draft, exclude={"plan_hash"})}
    )
    interpretation_content = {
        "schema_version": 1,
        "schema_id": "benchmark.result_interpretation_report.v1",
        "benchmark_version": "v1",
        "run_id": secondary.run_id,
        "interpretation_plan_hash": plan.plan_hash,
        "total_case_count": 24,
        "eligible_case_count": 23,
        "missing_case_count": 1,
        "primary_mean_effect": 7 / 23,
        "primary_interval_lower": 0.0,
        "primary_interval_upper": 13 / 23,
        "minimum_interpretable_effect": 0.1,
        "exact_mcnemar_p_value": 0.09228515625,
        "primary_conclusion": "inconclusive",
        "specificity_mean_effect": combined_inference.mean_case_effect,
        "specificity_interval_lower": combined_inference.interval_lower,
        "specificity_interval_upper": combined_inference.interval_upper,
        "specificity_conclusion": "specific",
        "combined_conclusion": "inconclusive",
        "primary_wording": "The primary result remains inconclusive.",
        "specificity_wording": "Valid evidence performed better than the average control.",
        "combined_wording": "Secondary evidence cannot rescue the primary result.",
        "interpretation_rule_status": "post_result_reporting_rule_not_confirmatory_gate",
        "evidence_specificity_role": "secondary_cannot_rescue_primary",
        "post_hoc_correction_role": "sensitivity_cannot_replace_sealed_primary",
        "result_scope": "one_pinned_model_run_on_one_fixed_authored_corpus",
        "human_learning_claim_allowed": False,
        "tutoring_efficacy_claim_allowed": False,
        "reduced_cognitive_offloading_claim_allowed": False,
        "absence_of_significance_means_no_effect": False,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": datetime(2026, 9, 3, 15, 20, tzinfo=UTC),
    }
    interpretation_draft = ResultInterpretationReport.model_construct(
        _fields_set=set(interpretation_content),
        **interpretation_content,
        report_hash="0" * 64,
    )
    interpretation = ResultInterpretationReport.model_validate(
        {
            **interpretation_content,
            "report_hash": model_content_hash(
                interpretation_draft,
                exclude={"report_hash"},
            ),
        }
    )
    secondary_path = root / "secondary.json"
    plan_path = root / "interpretation-plan.json"
    interpretation_path = root / "interpretation.json"
    write_immutable_json(secondary_path, secondary)
    write_immutable_json(plan_path, plan)
    write_immutable_json(interpretation_path, interpretation)
    return secondary_path, plan_path, interpretation_path, pixi_lock


def _case_effect(
    case_id: str,
    unrelated: float,
    corrupted: float,
    index: int,
) -> EvidenceSpecificityCaseEffect:
    content = {
        "benchmark_version": "v1",
        "run_id": "specificity-figure-test",
        "case_id": case_id,
        "model_route_id": "provider/model",
        "valid_vs_unrelated_effect": unrelated,
        "valid_vs_corrupted_effect": corrupted,
        "combined_specificity_effect": (unrelated + corrupted) / 2,
        "source_aggregate_hashes": (
            _sha(100 + index * 3),
            _sha(101 + index * 3),
            _sha(102 + index * 3),
        ),
    }
    draft = EvidenceSpecificityCaseEffect.model_construct(
        _fields_set=set(content),
        **content,
        effect_hash="0" * 64,
    )
    return EvidenceSpecificityCaseEffect.model_validate(
        {**content, "effect_hash": model_content_hash(draft, exclude={"effect_hash"})}
    )


def _comparison_summary(
    comparison: CaseComparison,
    comparison_index: int,
    missing_id: str,
) -> SecondaryComparisonSummary:
    inference = paired_case_inference(
        (0.0,) * 23,
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        minimum_interpretable_effect=0.1,
        random_seed=20260813,
    )
    return SecondaryComparisonSummary(
        comparison=comparison,
        total_case_count=24,
        eligible_case_count=23,
        missing_case_ids=(missing_id,),
        favourable_case_count=0,
        adverse_case_count=0,
        unchanged_case_count=23,
        inference=inference,
        source_aggregate_hashes=tuple(
            _sha(1000 + comparison_index * 24 + index) for index in range(24)
        ),
    )


def _sha(value: int) -> str:
    return f"{value:064x}"
