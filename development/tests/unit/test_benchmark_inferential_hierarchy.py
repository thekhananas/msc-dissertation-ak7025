"""Tests for the final pre-criterion inferential hierarchy."""

from datetime import date
from pathlib import Path

import pytest

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.binary_sensitivity import BinarySensitivityReport
from socratic_tutor.benchmark.design import load_design
from socratic_tutor.benchmark.evidence_specificity import EvidenceSpecificityAmendment
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import (
    InferentialHierarchyPlan,
    freeze_inferential_hierarchy,
    load_inferential_hierarchy_plan,
)
from socratic_tutor.benchmark.methodology_clarification import MethodologyClarification
from socratic_tutor.benchmark.public_rating_procedure import PublicRatingProcedureRecord

WORKSPACE_ROOT = Path(__file__).parents[2]
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-inferential-hierarchy.yaml"
ANALYSIS_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
DESIGN_PATH = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"


def _inputs():
    analysis = load_analysis_specification(ANALYSIS_PATH)
    design = load_design(DESIGN_PATH)
    methodology = MethodologyClarification.model_construct(
        clarification_hash="a" * 64,
        analysis_specification_hash=analysis_specification_hash(analysis),
        source_manifest_hash="b" * 64,
        external_protocol_hash="c" * 64,
    )
    sensitivity = BinarySensitivityReport.model_construct(
        report_hash="d" * 64,
        methodology_clarification_hash=methodology.clarification_hash,
        case_count=24,
        model_runs_per_case=1,
    )
    rating = PublicRatingProcedureRecord.model_construct(
        record_hash="e" * 64,
        methodology_clarification_hash=methodology.clarification_hash,
        item_count=24,
    )
    specificity = EvidenceSpecificityAmendment.model_construct(
        amendment_hash="f" * 64,
        base_analysis_specification_hash=analysis_specification_hash(analysis),
        base_external_protocol_hash=methodology.external_protocol_hash,
    )
    plan = InferentialHierarchyPlan(
        hierarchy_version="inferential-hierarchy-v1",
        frozen_on=date(2026, 9, 1),
        decision_owner="dissertation_author",
        analysis_specification_hash=analysis_specification_hash(analysis),
        methodology_clarification_hash=methodology.clarification_hash,
        binary_sensitivity_report_hash=sensitivity.report_hash,
        public_rating_procedure_hash=rating.record_hash,
        evidence_specificity_amendment_hash=specificity.amendment_hash,
        benchmark_design_hash=model_content_hash(design),
        source_manifest_hash=methodology.source_manifest_hash,
        external_protocol_hash=methodology.external_protocol_hash,
    )
    return plan, analysis, methodology, sensitivity, rating, specificity, design


def test_versioned_plan_binds_every_precriterion_source() -> None:
    plan = load_inferential_hierarchy_plan(PLAN_PATH)

    assert plan.hierarchy_version == "inferential-hierarchy-v1"
    assert len(plan.binary_sensitivity_report_hash) == 64
    assert len(plan.public_rating_procedure_hash) == 64


def test_freezes_primary_order_and_secondary_limits(tmp_path: Path) -> None:
    plan, analysis, methodology, sensitivity, rating, specificity, design = _inputs()
    hierarchy = freeze_inferential_hierarchy(
        plan=plan,
        analysis_specification=analysis,
        methodology=methodology,
        sensitivity=sensitivity,
        rating_procedure=rating,
        specificity=specificity,
        design=design,
        output_path=tmp_path / "hierarchy.json",
    )

    assert hierarchy.primary_estimand == "mean_paired_classification_error_difference_by_case"
    assert hierarchy.primary_test == "exact_two_sided_mcnemar"
    assert hierarchy.secondary_can_rescue_primary is False
    assert hierarchy.case_interval_scope == "conditional_on_this_fixed_authored_corpus"
    assert hierarchy.case_count == 24
    assert hierarchy.concept_count == 4
    assert hierarchy.misconception_count == 8
    assert hierarchy.all_precriterion_clarifications_complete is True


def test_rejects_a_sensitivity_report_from_another_methodology(tmp_path: Path) -> None:
    plan, analysis, methodology, sensitivity, rating, specificity, design = _inputs()
    changed = sensitivity.model_copy(update={"methodology_clarification_hash": "0" * 64})

    with pytest.raises(ValueError, match="another methodology"):
        freeze_inferential_hierarchy(
            plan=plan,
            analysis_specification=analysis,
            methodology=methodology,
            sensitivity=changed,
            rating_procedure=rating,
            specificity=specificity,
            design=design,
            output_path=tmp_path / "invalid.json",
        )
