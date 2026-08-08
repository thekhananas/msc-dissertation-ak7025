from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.primary_analysis import (
    MissingPrimaryCase,
    PrimaryAnalysisReport,
    PrimaryCaseResult,
)
from socratic_tutor.benchmark.primary_figure import render_primary_result_figure
from socratic_tutor.benchmark.result_interpretation import (
    ResultInterpretationPlan,
    ResultInterpretationReport,
    classify_primary,
    combine_conclusions,
)
from socratic_tutor.benchmark.statistics import exact_mcnemar, paired_case_inference


def test_renders_all_cases_and_retries_exactly(tmp_path: Path) -> None:
    primary_path, plan_path, interpretation_path, pixi_lock = _write_sources(tmp_path)
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 14, 0, tzinfo=UTC)
    first = render_primary_result_figure(
        primary_report_path=primary_path,
        interpretation_plan_path=plan_path,
        interpretation_report_path=interpretation_path,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
        generated_at_utc=generated_at,
    )
    retry = render_primary_result_figure(
        primary_report_path=primary_path,
        interpretation_plan_path=plan_path,
        interpretation_report_path=interpretation_path,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
    )

    assert first == retry
    assert first.primary_conclusion == "inconclusive"
    assert not first.human_learning_claim_supported
    assert (output_root / "primary_result.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "primary_result.svg").read_text(encoding="utf-8")
    assert "Primary conclusion: inconclusive" in svg
    assert "does not show" in svg
    csv = (output_root / "primary_case_results.csv").read_text(encoding="utf-8")
    assert len(csv.splitlines()) == 25
    assert "h-c2m2-03,missing" in csv


def _write_sources(root: Path) -> tuple[Path, Path, Path, Path]:
    case_ids = tuple(
        f"h-c{concept}m{misconception}-{index:02}"
        for concept in range(1, 5)
        for misconception in range(1, 3)
        for index in range(1, 4)
    )
    missing_id = "h-c2m2-03"
    eligible_ids = tuple(case_id for case_id in case_ids if case_id != missing_id)
    pairs = ((False, True),) * 10 + ((True, False),) * 3 + ((True, True),) * 10
    case_results = tuple(
        _case_result(case_id, dialogue_correct, evidence_correct)
        for case_id, (dialogue_correct, evidence_correct) in zip(
            eligible_ids,
            pairs,
            strict=True,
        )
    )
    effects = tuple(row.case_effect for row in case_results)
    inference = paired_case_inference(
        effects,
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        minimum_interpretable_effect=0.1,
        random_seed=20260813,
    )
    mcnemar = exact_mcnemar(pairs)
    primary_content = {
        "schema_version": 1,
        "schema_id": "benchmark.primary_analysis_report.v1",
        "benchmark_version": "v1",
        "run_id": "primary-figure-test",
        "analysis_plan_hash": "1" * 64,
        "total_case_count": 24,
        "eligible_case_count": 23,
        "missing_case_count": 1,
        "case_results": case_results,
        "missing_cases": (
            MissingPrimaryCase(
                case_id=missing_id,
                missing_reason="sandbox_error",
                aggregate_record_hash="2" * 64,
                criterion_record_hash="3" * 64,
            ),
        ),
        "inference": inference,
        "mcnemar": mcnemar,
        "valid_evidence_improvement_count": 10,
        "valid_evidence_regression_count": 3,
        "net_improvement_count": 7,
        "paired_effect_from_counts": 7 / 23,
        "interval_threshold_status": "interval_crosses_minimum",
        "mcnemar_rejects_equal_discordance": False,
        "mcnemar_alpha": 0.05,
        "mcnemar_scope": "tests_equality_of_discordant_probabilities",
        "threshold_scope": "tests_whether_mean_reaches_frozen_practical_threshold",
        "significance_is_not_success_gate": True,
        "sign_swap_role": "secondary_sensitivity_only",
        "interval_scope": "conditional_on_this_fixed_authored_corpus",
        "repeat_variability": "not_estimable_with_one_model_run_per_case",
        "completed_at_utc": datetime(2026, 9, 3, 13, 0, tzinfo=UTC),
    }
    primary_draft = PrimaryAnalysisReport.model_construct(
        _fields_set=set(primary_content),
        **primary_content,
        report_hash="0" * 64,
    )
    primary = PrimaryAnalysisReport.model_validate(
        {
            **primary_content,
            "report_hash": model_content_hash(primary_draft, exclude={"report_hash"}),
        }
    )

    pixi_lock = root / "pixi.lock"
    pixi_lock.write_text("locked\n", encoding="utf-8")
    lock_hash = file_sha256(pixi_lock.read_bytes())
    plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.result_interpretation_plan.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "primary_report_hash": primary.report_hash,
        "secondary_report_hash": "4" * 64,
        "probe_correction_plan_hash": "5" * 64,
        "probe_correction_report_hash": "6" * 64,
        "negative_result_audit_plan_hash": "7" * 64,
        "negative_result_audit_report_hash": "8" * 64,
        "analysis_code_revision": "9" * 40,
        "analysis_pixi_lock_hash": lock_hash,
        "record_owner": "dissertation_author",
        "created_at_utc": datetime(2026, 9, 3, 13, 10, tzinfo=UTC),
    }
    plan_draft = ResultInterpretationPlan.model_construct(
        _fields_set=set(plan_content),
        **plan_content,
        plan_hash="0" * 64,
    )
    plan = ResultInterpretationPlan.model_validate(
        {**plan_content, "plan_hash": model_content_hash(plan_draft, exclude={"plan_hash"})}
    )
    primary_conclusion = classify_primary(
        interval_lower=inference.interval_lower,
        interval_upper=inference.interval_upper,
        minimum_interpretable_effect=inference.minimum_interpretable_effect,
    )
    specificity_conclusion = "specific"
    interpretation_content = {
        "schema_version": 1,
        "schema_id": "benchmark.result_interpretation_report.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "interpretation_plan_hash": plan.plan_hash,
        "total_case_count": 24,
        "eligible_case_count": 23,
        "missing_case_count": 1,
        "primary_mean_effect": inference.mean_case_effect,
        "primary_interval_lower": inference.interval_lower,
        "primary_interval_upper": inference.interval_upper,
        "minimum_interpretable_effect": inference.minimum_interpretable_effect,
        "exact_mcnemar_p_value": mcnemar.two_sided_exact_p_value,
        "primary_conclusion": primary_conclusion,
        "specificity_mean_effect": 0.5,
        "specificity_interval_lower": 0.2,
        "specificity_interval_upper": 0.8,
        "specificity_conclusion": specificity_conclusion,
        "combined_conclusion": combine_conclusions(
            primary_conclusion,
            specificity_conclusion,
        ),
        "primary_wording": "The primary result is inconclusive.",
        "specificity_wording": "Valid evidence was more specific than controls.",
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
        "completed_at_utc": datetime(2026, 9, 3, 13, 20, tzinfo=UTC),
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
    primary_path = root / "primary.json"
    plan_path = root / "interpretation-plan.json"
    interpretation_path = root / "interpretation.json"
    write_immutable_json(primary_path, primary)
    write_immutable_json(plan_path, plan)
    write_immutable_json(interpretation_path, interpretation)
    return primary_path, plan_path, interpretation_path, pixi_lock


def _case_result(
    case_id: str,
    dialogue_correct: bool,
    evidence_correct: bool,
) -> PrimaryCaseResult:
    return PrimaryCaseResult(
        case_id=case_id,
        criterion_outcome=True,
        dialogue_binary_decision=dialogue_correct,
        valid_evidence_binary_decision=evidence_correct,
        dialogue_correct=dialogue_correct,
        valid_evidence_correct=evidence_correct,
        dialogue_classification_error=float(not dialogue_correct),
        valid_evidence_classification_error=float(not evidence_correct),
        case_effect=float(evidence_correct) - float(dialogue_correct),
        aggregate_record_hash="a" * 64,
        dialogue_repeat_hash="b" * 64,
        valid_evidence_repeat_hash="c" * 64,
        criterion_record_hash="d" * 64,
    )
