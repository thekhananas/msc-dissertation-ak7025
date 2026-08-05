# pyright: reportPrivateUsage=false
"""End-to-end deterministic scoring tests for a sealed external run."""

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.calibration import (
    CalibrationManifestInventory,
    UncalibratedDecisionReport,
)
from socratic_tutor.benchmark.dependence_figure import render_dependence_figure
from socratic_tutor.benchmark.dependence_sensitivity import run_dependence_sensitivity
from socratic_tutor.benchmark.design import load_design
from socratic_tutor.benchmark.evaluator.scoring import (
    CalibrationDecision,
    CalibrationStatus,
    create_calibration_decision,
)
from socratic_tutor.benchmark.evidence_specificity import EvidenceSpecificityAmendment
from socratic_tutor.benchmark.external_criterion import run_external_criterion
from socratic_tutor.benchmark.external_replay import (
    ExternalReplayPlan,
    ExternalReplayReport,
    replay_published_datasets,
)
from socratic_tutor.benchmark.external_scoring import run_external_scoring
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import InferentialHierarchy
from socratic_tutor.benchmark.missingness_sensitivity import run_missingness_sensitivity
from socratic_tutor.benchmark.primary_analysis import run_primary_analysis
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.global_seal import FilesystemGlobalDecisionSealStore
from socratic_tutor.benchmark.secondary_analysis import run_secondary_analysis
from tests.unit.test_benchmark_external_criterion import (
    BENCHMARK_ROOT,
    CRITERION_AT,
    PROMPT,
    FakeExecutor,
    FakeGateway,
    IncrementingClock,
    _no_sleep,
    _sealed_fixture,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SPECIFICATION = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
DEPENDENCE_GRID = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-dependence-sensitivity.yaml"
DESIGN = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"
PIXI_LOCK = WORKSPACE_ROOT / "pixi.lock"
SCORED_AT = datetime(2026, 9, 1, 13, 0, tzinfo=UTC)


def test_scores_sealed_run_once_and_preserves_missingness(tmp_path: Path) -> None:
    specification = load_analysis_specification(ANALYSIS_SPECIFICATION)
    authored_hash = file_sha256((BENCHMARK_ROOT / "manifest.yaml").read_bytes())
    calibration = create_calibration_decision(
        status=CalibrationStatus.UNCALIBRATED_SCORE,
        policy_threshold=specification.policy_threshold,
        calibration_task_family_hash=canonical_sha256(()),
        calibration_data_hash=canonical_sha256(()),
        tracker_version="simple-v1",
        diagnostics_hash=canonical_sha256({"status": "no_independent_calibration_cases"}),
        decision_owner="researcher",
        decided_at_utc=datetime(2026, 8, 30, 9, 0, tzinfo=UTC),
    )
    calibration_report = _calibration_report(
        calibration=calibration,
        manifest_hash=authored_hash,
        specification_hash=analysis_specification_hash(specification),
    )
    public, evaluator, criterion_plan = _sealed_fixture(
        tmp_path,
        calibration_decision_hash=calibration.decision_hash,
    )
    failed_case = public.cases[0].case_id
    asyncio.run(
        run_external_criterion(
            plan=criterion_plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=FakeGateway(fail_case_id=failed_case),
            executor=FakeExecutor(),
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )
    replay_root = tmp_path / "replay" / "preanalysis-v1"
    replay_plan, replay_report = _replay_evidence(tmp_path, replay_root)
    write_immutable_json(replay_root / "external_replay_plan.json", replay_plan)
    write_immutable_json(replay_root / "external_replay_report.json", replay_report)
    public_path = tmp_path / "public_manifest.json"
    calibration_path = tmp_path / "calibration_report.json"
    write_immutable_json(public_path, public)
    write_immutable_json(calibration_path, calibration_report)

    summary = run_external_scoring(
        replay_report_path=replay_root / "external_replay_report.json",
        analysis_specification_path=ANALYSIS_SPECIFICATION,
        calibration_report_path=calibration_path,
        public_manifest_path=public_path,
        benchmark_root=BENCHMARK_ROOT,
        seal_root=tmp_path,
        pixi_lock_path=PIXI_LOCK,
        scoring_code_revision="external-scoring-unit",
        created_at_utc=SCORED_AT,
    )

    assert summary.gate_passed
    assert summary.criterion_count == 24
    assert summary.eligible_criterion_count == 23
    assert summary.missing_case_ids == (failed_case,)
    assert summary.baseline_count == 48
    assert summary.repeat_metric_count == 144
    assert summary.eligible_repeat_metric_count == 138
    assert summary.case_aggregate_count == 120
    assert summary.eligible_case_aggregate_count == 115
    assert summary.effect_interpretation_status == "not_computed"
    assert summary.network_calls_made == 0
    assert summary.sandbox_calls_made == 0
    assert (
        run_external_scoring(
            replay_report_path=replay_root / "external_replay_report.json",
            analysis_specification_path=ANALYSIS_SPECIFICATION,
            calibration_report_path=calibration_path,
            public_manifest_path=public_path,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            pixi_lock_path=PIXI_LOCK,
            scoring_code_revision="external-scoring-unit",
            created_at_utc=SCORED_AT,
        )
        == summary
    )

    amendment = _specificity_amendment(analysis_specification_hash(specification))
    amendment_path = tmp_path / "evidence_specificity_amendment.json"
    write_immutable_json(amendment_path, amendment)
    design = load_design(DESIGN)
    hierarchy = _inferential_hierarchy(
        analysis_specification_hash(specification),
        amendment.amendment_hash,
        model_content_hash(design),
    )
    hierarchy_path = tmp_path / "inferential_hierarchy.json"
    write_immutable_json(hierarchy_path, hierarchy)
    primary = run_primary_analysis(
        scoring_summary_path=tmp_path / "analysis" / "scoring-v1" / "external_scoring_summary.json",
        analysis_specification_path=ANALYSIS_SPECIFICATION,
        inferential_hierarchy_path=hierarchy_path,
        dataset_root=tmp_path / "datasets",
        pixi_lock_path=PIXI_LOCK,
        output_root=tmp_path / "analysis" / "primary-v1",
        analysis_code_revision="primary-analysis-unit",
        created_at_utc=datetime(2026, 9, 1, 14, 0, tzinfo=UTC),
    )

    assert primary.total_case_count == 24
    assert primary.eligible_case_count == 23
    assert primary.missing_case_count == 1
    assert primary.missing_cases[0].case_id == failed_case
    assert primary.valid_evidence_improvement_count == 23
    assert primary.valid_evidence_regression_count == 0
    assert primary.net_improvement_count == 23
    assert primary.paired_effect_from_counts == 1.0
    assert primary.inference.mean_case_effect == 1.0
    assert primary.mcnemar.right_only_correct_count == 23
    assert primary.significance_is_not_success_gate is True
    assert (
        run_primary_analysis(
            scoring_summary_path=tmp_path
            / "analysis"
            / "scoring-v1"
            / "external_scoring_summary.json",
            analysis_specification_path=ANALYSIS_SPECIFICATION,
            inferential_hierarchy_path=hierarchy_path,
            dataset_root=tmp_path / "datasets",
            pixi_lock_path=PIXI_LOCK,
            output_root=tmp_path / "analysis" / "primary-v1",
            analysis_code_revision="primary-analysis-unit",
            created_at_utc=datetime(2026, 9, 1, 14, 0, tzinfo=UTC),
        )
        == primary
    )

    secondary = run_secondary_analysis(
        scoring_summary_path=tmp_path / "analysis" / "scoring-v1" / "external_scoring_summary.json",
        primary_analysis_plan_path=tmp_path
        / "analysis"
        / "primary-v1"
        / "primary_analysis_plan.json",
        primary_analysis_report_path=tmp_path
        / "analysis"
        / "primary-v1"
        / "primary_analysis_report.json",
        analysis_specification_path=ANALYSIS_SPECIFICATION,
        inferential_hierarchy_path=hierarchy_path,
        evidence_specificity_amendment_path=amendment_path,
        dataset_root=tmp_path / "datasets",
        pixi_lock_path=PIXI_LOCK,
        output_root=tmp_path / "analysis" / "secondary-v1",
        analysis_code_revision="secondary-analysis-unit",
        created_at_utc=datetime(2026, 9, 1, 15, 0, tzinfo=UTC),
    )

    assert len(secondary.comparison_summaries) == 5
    assert all(item.total_case_count == 24 for item in secondary.comparison_summaries)
    assert all(item.eligible_case_count == 23 for item in secondary.comparison_summaries)
    assert all(item.missing_case_ids == (failed_case,) for item in secondary.comparison_summaries)
    assert secondary.evidence_specificity.total_case_count == 24
    assert secondary.evidence_specificity.eligible_case_count == 23
    assert secondary.evidence_specificity.missing_case_count == 1
    assert secondary.secondary_can_rescue_primary is False
    assert secondary.network_calls_made == 0
    assert secondary.sandbox_calls_made == 0

    sensitivity = run_dependence_sensitivity(
        grid_path=DEPENDENCE_GRID,
        primary_analysis_plan_path=tmp_path
        / "analysis"
        / "primary-v1"
        / "primary_analysis_plan.json",
        primary_report_path=tmp_path / "analysis" / "primary-v1" / "primary_analysis_report.json",
        secondary_report_path=tmp_path
        / "analysis"
        / "secondary-v1"
        / "secondary_analysis_report.json",
        inferential_hierarchy_path=hierarchy_path,
        benchmark_design_path=DESIGN,
        dataset_root=tmp_path / "datasets",
        pixi_lock_path=PIXI_LOCK,
        output_root=tmp_path / "analysis" / "dependence-v1",
        analysis_code_revision="dependence-sensitivity-unit",
        created_at_utc=datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
    )

    assert len(sensitivity.concept_effects) == 4
    assert len(sensitivity.misconception_effects) == 8
    assert len(sensitivity.leave_one_concept_out) == 4
    assert len(sensitivity.threshold_sensitivity) == 5
    assert len(sensitivity.update_size_sensitivity) == 5
    assert sensitivity.grid_values_selected_after_reveal is True
    assert sensitivity.secondary_can_rescue_primary is False
    assert sensitivity.repeat_variability_estimable is False

    figure = render_dependence_figure(
        report_path=tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity_report.json",
        pdf_output_path=tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity.pdf",
        svg_output_path=tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity.svg",
        manifest_path=tmp_path / "analysis" / "dependence-v1" / "figure_manifest.json",
        generated_at_utc=datetime(2026, 9, 1, 16, 1, tzinfo=UTC),
    )
    svg = (tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity.svg").read_text()
    assert figure.source_report_hash == sensitivity.report_hash
    assert figure.figure_formats == ("pdf", "svg")
    assert figure.panel_count == 3
    assert (
        (tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity.pdf")
        .read_bytes()
        .startswith(b"%PDF")
    )
    assert (
        "Each condition predicts whether the evaluation model completes a separate coding task"
        in svg
    )
    assert "Probe-informed prediction combines the public answer" in svg
    assert "Public-answer comparison by concept" in svg
    assert "Across decision thresholds" in svg
    assert "Across executable-evidence weights" in svg
    assert "23 of 24 benchmark cases were analysed" in svg
    assert "no population claim" not in svg

    repeated = render_dependence_figure(
        report_path=tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity_report.json",
        pdf_output_path=tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity.pdf",
        svg_output_path=tmp_path / "analysis" / "dependence-v1" / "dependence_sensitivity.svg",
        manifest_path=tmp_path / "analysis" / "dependence-v1" / "figure_manifest.json",
    )
    assert repeated.pdf_sha256 == figure.pdf_sha256
    assert repeated.svg_sha256 == figure.svg_sha256
    assert repeated.manifest_hash == figure.manifest_hash

    missingness = run_missingness_sensitivity(
        primary_analysis_plan_path=tmp_path
        / "analysis"
        / "primary-v1"
        / "primary_analysis_plan.json",
        primary_report_path=tmp_path / "analysis" / "primary-v1" / "primary_analysis_report.json",
        dataset_root=tmp_path / "datasets",
        pixi_lock_path=PIXI_LOCK,
        output_root=tmp_path / "analysis" / "missingness-v1",
        analysis_code_revision="missingness-sensitivity-unit",
        created_at_utc=datetime(2026, 9, 1, 16, 2, tzinfo=UTC),
    )
    missing_bound = missingness.missing_case_bounds[0]
    assert missingness.complete_case_count == 23
    assert missingness.missing_case_count == 1
    assert missing_bound.case_id == failed_case
    assert missing_bound.effect_if_criterion_false == -1
    assert missing_bound.effect_if_criterion_true == 1
    assert missing_bound.lower_bound_assignment == "criterion_false"
    assert missing_bound.upper_bound_assignment == "criterion_true"
    assert missingness.complete_case_effect == 1.0
    assert missingness.lower_bound_full_corpus_effect == 22 / 24
    assert missingness.upper_bound_full_corpus_effect == 1.0
    assert abs(missingness.bound_width - (2 / 24)) < 1e-12
    assert missingness.complete_case_remains_primary is True
    assert missingness.network_calls_made == 0
    assert missingness.sandbox_calls_made == 0
    assert (
        run_missingness_sensitivity(
            primary_analysis_plan_path=tmp_path
            / "analysis"
            / "primary-v1"
            / "primary_analysis_plan.json",
            primary_report_path=tmp_path
            / "analysis"
            / "primary-v1"
            / "primary_analysis_report.json",
            dataset_root=tmp_path / "datasets",
            pixi_lock_path=PIXI_LOCK,
            output_root=tmp_path / "analysis" / "missingness-v1",
            analysis_code_revision="missingness-sensitivity-unit",
            created_at_utc=datetime(2026, 9, 1, 16, 2, tzinfo=UTC),
        )
        == missingness
    )


def _calibration_report(
    *,
    calibration: CalibrationDecision,
    manifest_hash: str,
    specification_hash: str,
) -> UncalibratedDecisionReport:
    inventory = CalibrationManifestInventory(
        manifest_ref="v1/manifest.yaml",
        benchmark_version="v1",
        manifest_hash=manifest_hash,
        calibration_case_count=0,
        calibration_task_families=(),
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.uncalibrated_decision_report.v1",
        "benchmark_version": "v1",
        "heldout_manifest_hash": manifest_hash,
        "analysis_specification_hash": specification_hash,
        "source_manifests": (inventory,),
        "calibration_case_count": 0,
        "reason": "no_independent_calibration_cases",
        "decision": calibration,
    }
    draft = UncalibratedDecisionReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    return UncalibratedDecisionReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )


def _replay_evidence(
    seal_root: Path,
    replay_root: Path,
) -> tuple[ExternalReplayPlan, ExternalReplayReport]:
    conditions = FilesystemConditionCommitStore(seal_root / "decision")
    seal = FilesystemGlobalDecisionSealStore(seal_root / "decision", conditions).load()
    assert seal is not None
    comparison = replay_published_datasets(
        seal_root=seal_root,
        output_dataset_root=replay_root / "datasets",
    )
    replay_plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_replay_plan.v1",
        "run_id": seal.run_id,
        "postcriterion_integrity_report_hash": "a" * 64,
        "global_seal_hash": seal.seal_hash,
        "decision_response_root_hash": "b" * 64,
        "criterion_response_root_hash": "c" * 64,
        "replay_code_revision": "external-replay-unit",
        "replay_pixi_lock_hash": file_sha256(PIXI_LOCK.read_bytes()),
        "created_at_utc": CRITERION_AT,
    }
    plan_draft = ExternalReplayPlan.model_construct(
        _fields_set=set(replay_plan_content), **replay_plan_content, plan_hash="0" * 64
    )
    plan = ExternalReplayPlan.model_validate(
        {
            **replay_plan_content,
            "plan_hash": model_content_hash(plan_draft, exclude={"plan_hash"}),
        }
    )
    report_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_replay_report.v1",
        "run_id": seal.run_id,
        "replay_plan_hash": plan.plan_hash,
        "postcriterion_integrity_report_hash": "a" * 64,
        "replayed_decision_response_count": 48,
        "replayed_criterion_response_count": 23,
        "replayed_request_count": 71,
        "decision_response_root_hash": "b" * 64,
        "criterion_response_root_hash": "c" * 64,
        **comparison,
        "scored_dataset_status": "not_yet_created",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": CRITERION_AT,
        "gate_passed": True,
    }
    report_draft = ExternalReplayReport.model_construct(
        _fields_set=set(report_content), **report_content, report_hash="0" * 64
    )
    report = ExternalReplayReport.model_validate(
        {
            **report_content,
            "report_hash": model_content_hash(report_draft, exclude={"report_hash"}),
        }
    )
    return plan, report


def _inferential_hierarchy(
    specification_hash: str,
    specificity_amendment_hash: str,
    benchmark_design_hash: str,
) -> InferentialHierarchy:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.inferential_hierarchy.v1",
        "hierarchy_version": "inferential-hierarchy-v1",
        "frozen_on": date(2026, 8, 30),
        "decision_owner": "dissertation_author",
        "analysis_specification_hash": specification_hash,
        "methodology_clarification_hash": "1" * 64,
        "binary_sensitivity_report_hash": "2" * 64,
        "public_rating_procedure_hash": "3" * 64,
        "evidence_specificity_amendment_hash": specificity_amendment_hash,
        "benchmark_design_hash": benchmark_design_hash,
        "source_manifest_hash": "6" * 64,
        "external_protocol_hash": "7" * 64,
        "primary_estimand": "mean_paired_classification_error_difference_by_case",
        "primary_effect_direction": "positive_means_valid_evidence_predicts_better",
        "primary_test": "exact_two_sided_mcnemar",
        "primary_test_alpha": 0.05,
        "primary_uncertainty": "paired_case_percentile_bootstrap_interval",
        "bootstrap_resamples": 10_000,
        "minimum_interpretable_effect": 0.1,
        "primary_result_rule": (
            "report_mean_interval_threshold_and_exact_test_without_a_significance_success_gate"
        ),
        "secondary_analyses": (
            "within_case_sign_swap",
            "median_case_effect",
            "wilcoxon_signed_rank",
            "policy_threshold_sensitivity",
            "tracker_update_size_sensitivity",
            "misconception_group_summary",
            "concept_group_summary",
            "leave_one_concept_out",
        ),
        "secondary_can_rescue_primary": False,
        "specificity_role": "secondary_interpretation_only",
        "case_count": 24,
        "concept_count": 4,
        "misconception_count": 8,
        "cases_per_misconception": 3,
        "dependence_statement": ("cases_are_nested_within_eight_misconceptions_and_four_concepts"),
        "case_interval_scope": "conditional_on_this_fixed_authored_corpus",
        "subgroup_role": "descriptive_fragility_checks_not_confirmatory_tests",
        "repeat_variability": "not_estimable_with_one_model_run_per_case",
        "multiple_testing_rule": "no_secondary_p_value_supports_a_confirmatory_claim",
        "required_before_criterion_access": True,
        "all_precriterion_clarifications_complete": True,
    }
    draft = InferentialHierarchy.model_construct(
        _fields_set=set(content), **content, hierarchy_hash="0" * 64
    )
    return InferentialHierarchy.model_validate(
        {**content, "hierarchy_hash": model_content_hash(draft, exclude={"hierarchy_hash"})}
    )


def _specificity_amendment(specification_hash: str) -> EvidenceSpecificityAmendment:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.evidence_specificity_amendment.v1",
        "amendment_version": "evidence-specificity-v1",
        "base_analysis_specification_hash": specification_hash,
        "base_external_protocol_hash": "7" * 64,
        "calibration_decision_hash": "8" * 64,
        "selected_loss": "classification_error",
        "primary_case_effect": "dialogue_loss_minus_valid_probe_loss",
        "valid_vs_unrelated_effect": "unrelated_probe_loss_minus_valid_probe_loss",
        "valid_vs_corrupted_effect": "corrupted_probe_loss_minus_valid_probe_loss",
        "combined_specificity_effect": ("mean_unrelated_and_corrupted_loss_minus_valid_probe_loss"),
        "aggregation_unit": "case",
        "aggregation_version": "case-mean-v1",
        "missingness_rule": "exclude_missing_criterion_and_report_denominator",
        "bootstrap_method": "paired_case_percentile",
        "bootstrap_resamples": 10_000,
        "permutation_method": "within_case_sign_swap",
        "permutation_resamples": 10_000,
        "random_seed": 20260813,
        "primary_minimum_interpretable_effect": 0.1,
        "secondary_role": "interpretation_only_cannot_rescue_primary",
        "positive_mean_interpretation": "valid_evidence_is_more_specific_than_controls",
        "nonpositive_mean_interpretation": "primary_effect_is_not_evidence_specific",
    }
    draft = EvidenceSpecificityAmendment.model_construct(
        _fields_set=set(content), **content, amendment_hash="0" * 64
    )
    return EvidenceSpecificityAmendment.model_validate(
        {**content, "amendment_hash": model_content_hash(draft, exclude={"amendment_hash"})}
    )
