"""Frozen hierarchy for primary inference and dependence checks."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.binary_sensitivity import BinarySensitivityReport
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.design import BenchmarkDesignPlan, DesignStatus
from socratic_tutor.benchmark.evidence_specificity import EvidenceSpecificityAmendment
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.json_io import load_json_model
from socratic_tutor.benchmark.methodology_clarification import MethodologyClarification
from socratic_tutor.benchmark.public_rating_procedure import PublicRatingProcedureRecord
from socratic_tutor.contracts import ContractModel

_SECONDARY_ANALYSES = (
    "within_case_sign_swap",
    "median_case_effect",
    "wilcoxon_signed_rank",
    "policy_threshold_sensitivity",
    "tracker_update_size_sensitivity",
    "misconception_group_summary",
    "concept_group_summary",
    "leave_one_concept_out",
)


class InferentialHierarchyPlan(ContractModel):
    """Reviewed identities for the final pre-criterion analysis clarification."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.inferential_hierarchy_plan.v1"] = (
        "benchmark.inferential_hierarchy_plan.v1"
    )
    hierarchy_version: Literal["inferential-hierarchy-v1"]
    frozen_on: date
    decision_owner: Literal["dissertation_author"]
    analysis_specification_hash: Sha256
    methodology_clarification_hash: Sha256
    binary_sensitivity_report_hash: Sha256
    public_rating_procedure_hash: Sha256
    evidence_specificity_amendment_hash: Sha256
    benchmark_design_hash: Sha256
    source_manifest_hash: Sha256
    external_protocol_hash: Sha256


class InferentialHierarchy(ContractModel):
    """Content-addressed order of evidence that cannot be changed after reveal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.inferential_hierarchy.v1"] = "benchmark.inferential_hierarchy.v1"
    hierarchy_version: Literal["inferential-hierarchy-v1"]
    frozen_on: date
    decision_owner: Literal["dissertation_author"]
    analysis_specification_hash: Sha256
    methodology_clarification_hash: Sha256
    binary_sensitivity_report_hash: Sha256
    public_rating_procedure_hash: Sha256
    evidence_specificity_amendment_hash: Sha256
    benchmark_design_hash: Sha256
    source_manifest_hash: Sha256
    external_protocol_hash: Sha256
    primary_estimand: Literal["mean_paired_classification_error_difference_by_case"]
    primary_effect_direction: Literal["positive_means_valid_evidence_predicts_better"]
    primary_test: Literal["exact_two_sided_mcnemar"]
    primary_test_alpha: float = Field(gt=0.0, lt=0.5)
    primary_uncertainty: Literal["paired_case_percentile_bootstrap_interval"]
    bootstrap_resamples: int = Field(ge=1000)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    primary_result_rule: Literal[
        "report_mean_interval_threshold_and_exact_test_without_a_significance_success_gate"
    ]
    secondary_analyses: tuple[str, ...] = Field(min_length=8, max_length=8)
    secondary_can_rescue_primary: Literal[False] = False
    specificity_role: Literal["secondary_interpretation_only"]
    case_count: Literal[24]
    concept_count: Literal[4]
    misconception_count: Literal[8]
    cases_per_misconception: Literal[3]
    dependence_statement: Literal["cases_are_nested_within_eight_misconceptions_and_four_concepts"]
    case_interval_scope: Literal["conditional_on_this_fixed_authored_corpus"]
    subgroup_role: Literal["descriptive_fragility_checks_not_confirmatory_tests"]
    repeat_variability: Literal["not_estimable_with_one_model_run_per_case"]
    multiple_testing_rule: Literal["no_secondary_p_value_supports_a_confirmatory_claim"]
    required_before_criterion_access: Literal[True] = True
    all_precriterion_clarifications_complete: Literal[True] = True
    hierarchy_hash: Sha256

    @model_validator(mode="after")
    def validate_hierarchy(self) -> InferentialHierarchy:
        if self.primary_test_alpha != 0.05:
            raise ValueError("Inferential hierarchy v1 requires alpha 0.05")
        if self.minimum_interpretable_effect != 0.1:
            raise ValueError("Inferential hierarchy requires the frozen 0.10 effect threshold")
        if self.secondary_analyses != _SECONDARY_ANALYSES:
            raise ValueError("Inferential hierarchy must retain every declared secondary analysis")
        if self.hierarchy_hash != model_content_hash(self, exclude={"hierarchy_hash"}):
            raise ValueError("Inferential hierarchy hash does not match its content")
        return self


def load_inferential_hierarchy_plan(path: Path) -> InferentialHierarchyPlan:
    """Load the reviewed hierarchy identities from YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read inferential hierarchy plan: {path}") from error
    return InferentialHierarchyPlan.model_validate(raw)


def freeze_inferential_hierarchy(
    *,
    plan: InferentialHierarchyPlan,
    analysis_specification: AnalysisSpecification,
    methodology: MethodologyClarification,
    sensitivity: BinarySensitivityReport,
    rating_procedure: PublicRatingProcedureRecord,
    specificity: EvidenceSpecificityAmendment,
    design: BenchmarkDesignPlan,
    output_path: Path,
) -> InferentialHierarchy:
    """Validate every pre-criterion source and publish the final hierarchy."""

    specification_hash = analysis_specification_hash(analysis_specification)
    design_hash = model_content_hash(design)
    _validate_bound_sources(
        plan=plan,
        specification_hash=specification_hash,
        methodology=methodology,
        sensitivity=sensitivity,
        rating_procedure=rating_procedure,
        specificity=specificity,
        design=design,
        design_hash=design_hash,
    )
    misconception_count = sum(len(concept.misconception_ids) for concept in design.concepts)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.inferential_hierarchy.v1",
        "hierarchy_version": plan.hierarchy_version,
        "frozen_on": plan.frozen_on,
        "decision_owner": plan.decision_owner,
        "analysis_specification_hash": specification_hash,
        "methodology_clarification_hash": methodology.clarification_hash,
        "binary_sensitivity_report_hash": sensitivity.report_hash,
        "public_rating_procedure_hash": rating_procedure.record_hash,
        "evidence_specificity_amendment_hash": specificity.amendment_hash,
        "benchmark_design_hash": design_hash,
        "source_manifest_hash": methodology.source_manifest_hash,
        "external_protocol_hash": methodology.external_protocol_hash,
        "primary_estimand": "mean_paired_classification_error_difference_by_case",
        "primary_effect_direction": "positive_means_valid_evidence_predicts_better",
        "primary_test": "exact_two_sided_mcnemar",
        "primary_test_alpha": 0.05,
        "primary_uncertainty": "paired_case_percentile_bootstrap_interval",
        "bootstrap_resamples": analysis_specification.bootstrap_resamples,
        "minimum_interpretable_effect": analysis_specification.minimum_interpretable_effect,
        "primary_result_rule": (
            "report_mean_interval_threshold_and_exact_test_without_a_significance_success_gate"
        ),
        "secondary_analyses": _SECONDARY_ANALYSES,
        "secondary_can_rescue_primary": False,
        "specificity_role": "secondary_interpretation_only",
        "case_count": len(design.held_out_cases),
        "concept_count": len(design.concepts),
        "misconception_count": misconception_count,
        "cases_per_misconception": len(design.held_out_cases) // misconception_count,
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
    hierarchy = InferentialHierarchy.model_validate(
        {**content, "hierarchy_hash": model_content_hash(draft, exclude={"hierarchy_hash"})}
    )
    write_immutable_json(output_path, hierarchy)
    return hierarchy


def load_methodology_clarification(path: Path) -> MethodologyClarification:
    return load_json_model(path, MethodologyClarification)


def load_binary_sensitivity_report(path: Path) -> BinarySensitivityReport:
    return load_json_model(path, BinarySensitivityReport)


def load_public_rating_procedure_record(path: Path) -> PublicRatingProcedureRecord:
    return load_json_model(path, PublicRatingProcedureRecord)


def _validate_bound_sources(
    *,
    plan: InferentialHierarchyPlan,
    specification_hash: Sha256,
    methodology: MethodologyClarification,
    sensitivity: BinarySensitivityReport,
    rating_procedure: PublicRatingProcedureRecord,
    specificity: EvidenceSpecificityAmendment,
    design: BenchmarkDesignPlan,
    design_hash: Sha256,
) -> None:
    expected = {
        "analysis specification": (plan.analysis_specification_hash, specification_hash),
        "methodology clarification": (
            plan.methodology_clarification_hash,
            methodology.clarification_hash,
        ),
        "binary sensitivity report": (
            plan.binary_sensitivity_report_hash,
            sensitivity.report_hash,
        ),
        "public rating procedure": (
            plan.public_rating_procedure_hash,
            rating_procedure.record_hash,
        ),
        "evidence specificity amendment": (
            plan.evidence_specificity_amendment_hash,
            specificity.amendment_hash,
        ),
        "benchmark design": (plan.benchmark_design_hash, design_hash),
        "source manifest": (plan.source_manifest_hash, methodology.source_manifest_hash),
        "external protocol": (plan.external_protocol_hash, methodology.external_protocol_hash),
    }
    for name, (declared, actual) in expected.items():
        if declared != actual:
            raise ValueError(f"Inferential hierarchy belongs to a different {name}")
    if methodology.analysis_specification_hash != specification_hash:
        raise ValueError("Methodology belongs to another analysis specification")
    if sensitivity.methodology_clarification_hash != methodology.clarification_hash:
        raise ValueError("Sensitivity report belongs to another methodology clarification")
    if rating_procedure.methodology_clarification_hash != methodology.clarification_hash:
        raise ValueError("Rating procedure belongs to another methodology clarification")
    if specificity.base_analysis_specification_hash != specification_hash:
        raise ValueError("Specificity amendment belongs to another analysis specification")
    if specificity.base_external_protocol_hash != methodology.external_protocol_hash:
        raise ValueError("Specificity amendment belongs to another external protocol")
    if design.status is not DesignStatus.FROZEN:
        raise ValueError("Inferential hierarchy requires a frozen benchmark design")
    misconception_count = sum(len(concept.misconception_ids) for concept in design.concepts)
    if (len(design.held_out_cases), len(design.concepts), misconception_count) != (24, 4, 8):
        raise ValueError(
            "Inferential hierarchy requires 24 cases, four concepts, and eight misconceptions"
        )
    if sensitivity.case_count != 24 or sensitivity.model_runs_per_case != 1:
        raise ValueError("Sensitivity report does not match the executed 24-case one-run design")
    if rating_procedure.item_count != 24:
        raise ValueError("Rating procedure does not cover the complete case corpus")
