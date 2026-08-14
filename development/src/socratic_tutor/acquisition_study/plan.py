"""Frozen design contracts for the selective executable-evidence study."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel

from .contracts import PolicyId, ProbeClassId


class AcquisitionStudyPlanError(ValueError):
    """A frozen acquisition-study input is missing or inconsistent."""


class EvaluationEnvironmentId(StrEnum):
    MATCHED = "matched"
    LOWER_RELIABILITY = "lower_reliability"
    ASYMMETRIC_ERRORS = "asymmetric_errors"
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"
    INVERTED_EVIDENCE = "inverted_evidence"
    DIFFICULTY_MISSINGNESS = "difficulty_missingness"
    CORRELATED_FAMILY_FAILURES = "correlated_family_failures"


class EnvironmentRole(StrEnum):
    MATCHED_SANITY_CHECK = "matched_sanity_check"
    HELD_OUT_MISMATCH = "held_out_mismatch"


class AblationId(StrEnum):
    REPLACE_LOWER_QUANTILE_WITH_POSTERIOR_MEAN = "replace_lower_quantile_with_posterior_mean"
    REMOVE_BOUNDED_UPDATE_FROM_CANDIDATE = "remove_bounded_update_from_candidate"
    REMOVE_BOTH_RELIABILITY_CONSERVATISM_AND_BOUNDED_UPDATE = (
        "remove_both_reliability_conservatism_and_bounded_update"
    )
    COMPARE_DENSE_AND_SPARSE_EQUAL_EXPECTED_RELIABILITY = (
        "compare_dense_and_sparse_classes_with_equal_expected_reliability"
    )


class ProbeClassCalibration(ContractModel):
    probe_class: ProbeClassId
    sensitivity: float = Field(gt=0.5, lt=1.0)
    specificity: float = Field(gt=0.5, lt=1.0)
    mastered_samples: int = Field(ge=8)
    non_mastered_samples: int = Field(ge=8)
    cases_per_episode: int = Field(ge=1)


class CalibrationDesign(ContractModel):
    environment_id: Literal["calibration_assumed_model"] = "calibration_assumed_model"
    seed: int = Field(ge=0)
    sensitivity_seeds: tuple[int, ...] = Field(min_length=20, max_length=20)
    beta_prior_alpha: float = Field(gt=0.0)
    beta_prior_beta: float = Field(gt=0.0)
    probe_classes: tuple[ProbeClassCalibration, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_probe_classes(self) -> CalibrationDesign:
        if tuple(item.probe_class for item in self.probe_classes) != tuple(ProbeClassId):
            raise ValueError("Calibration must contain each probe class once and in order")
        if self.seed in self.sensitivity_seeds or len(set(self.sensitivity_seeds)) != 20:
            raise ValueError("Primary and sensitivity calibration seeds must be distinct")
        return self


class ReliabilityTransform(ContractModel):
    sensitivity_scale_from_chance: float = Field(ge=0.0, le=1.0)
    specificity_scale_from_chance: float = Field(ge=0.0, le=1.0)
    outcome_inversion_probability: float = Field(ge=0.0, le=1.0)


class MissingnessModel(ContractModel):
    base_probability: float = Field(ge=0.0, lt=1.0)
    prior_difficulty_slope: float = Field(ge=0.0, lt=1.0)
    maximum_probability: float = Field(ge=0.0, lt=1.0)
    selected_probe_consumes_budget_when_missing: Literal[True] = True

    @model_validator(mode="after")
    def validate_missingness(self) -> MissingnessModel:
        if self.maximum_probability < self.base_probability:
            raise ValueError("Maximum missingness cannot be below base missingness")
        return self


class FamilyFaultModel(ContractModel):
    probability: float = Field(ge=0.0, lt=1.0)
    mode: Literal["invert_all_probe_outcomes_in_family"] = "invert_all_probe_outcomes_in_family"


class EvaluationEnvironment(ContractModel):
    environment_id: EvaluationEnvironmentId
    role: EnvironmentRole
    evidence_seed: int = Field(ge=0)
    definition: str = Field(min_length=20)
    reliability: ReliabilityTransform
    missingness: MissingnessModel
    family_fault: FamilyFaultModel

    def is_identity(self) -> bool:
        return (
            math.isclose(self.reliability.sensitivity_scale_from_chance, 1.0)
            and math.isclose(self.reliability.specificity_scale_from_chance, 1.0)
            and math.isclose(self.reliability.outcome_inversion_probability, 0.0)
            and math.isclose(self.missingness.base_probability, 0.0)
            and math.isclose(self.missingness.prior_difficulty_slope, 0.0)
            and math.isclose(self.missingness.maximum_probability, 0.0)
            and math.isclose(self.family_fault.probability, 0.0)
        )


class EpisodeDesign(ContractModel):
    development_episodes_per_environment: int = Field(ge=20)
    evaluation_episodes_per_environment: int = Field(ge=100)
    candidates_per_episode: int = Field(ge=8)
    families_per_episode: int = Field(ge=2)
    cases_per_family: int = Field(ge=2)
    latent_evaluation_seed: int = Field(ge=0)
    prior_beta_alpha: float = Field(gt=0.0)
    prior_beta_beta: float = Field(gt=0.0)
    prior_probability_floor: float = Field(gt=0.0, lt=0.5)
    truth_sampling: Literal["bernoulli_from_prior_probability"] = "bernoulli_from_prior_probability"
    family_assignment: Literal["one_case_per_probe_class"] = "one_case_per_probe_class"
    episode_sampling: Literal["independent_episodes"] = "independent_episodes"
    family_sampling: Literal[
        "independent_families_with_arbitrary_within_family_fault_dependence"
    ] = "independent_families_with_arbitrary_within_family_fault_dependence"
    environment_latent_streams: Literal[
        "distinct_stream_per_environment_derived_from_latent_seed"
    ] = "distinct_stream_per_environment_derived_from_latent_seed"
    prior_difficulty_formula: Literal["one_minus_two_times_absolute_prior_minus_half"] = (
        "one_minus_two_times_absolute_prior_minus_half"
    )

    @model_validator(mode="after")
    def validate_episode_shape(self) -> EpisodeDesign:
        if self.families_per_episode * self.cases_per_family != self.candidates_per_episode:
            raise ValueError("Family layout must cover every candidate exactly once")
        if self.cases_per_family != len(ProbeClassId):
            raise ValueError("Each family must contain one case from every probe class")
        return self


class AcquisitionEnvironmentSpecification(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.environment_specification.v1"] = (
        "acquisition_study.environment_specification.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    status: Literal["frozen_before_policy_implementation"] = "frozen_before_policy_implementation"
    claim_scope: Literal["glass_box_prediction_not_human_learning"] = (
        "glass_box_prediction_not_human_learning"
    )
    external_model_data_used: Literal[False] = False
    human_data_used: Literal[False] = False
    reliability_transform_formula: Literal[
        "chance_plus_scale_times_base_minus_chance_then_optional_inversion"
    ] = "chance_plus_scale_times_base_minus_chance_then_optional_inversion"
    environment_generation_order: tuple[str, ...] = Field(min_length=5, max_length=5)
    calibration: CalibrationDesign
    episodes: EpisodeDesign
    evaluation_environments: tuple[EvaluationEnvironment, ...] = Field(min_length=7, max_length=7)
    specification_hash: Sha256

    @model_validator(mode="after")
    def validate_specification(self) -> AcquisitionEnvironmentSpecification:
        expected_ids = tuple(EvaluationEnvironmentId)
        actual_ids = tuple(item.environment_id for item in self.evaluation_environments)
        if actual_ids != expected_ids:
            raise ValueError("Evaluation environments must contain the frozen ordered set")
        matched = self.evaluation_environments[0]
        if matched.role is not EnvironmentRole.MATCHED_SANITY_CHECK or not matched.is_identity():
            raise ValueError("The matched environment must be the identity sanity check")
        if any(
            item.role is not EnvironmentRole.HELD_OUT_MISMATCH
            for item in self.evaluation_environments[1:]
        ):
            raise ValueError("Every non-matched environment must be held out")
        if any(item.is_identity() for item in self.evaluation_environments[1:]):
            raise ValueError("A held-out environment must differ from the assumed model")
        seeds = [
            self.calibration.seed,
            *self.calibration.sensitivity_seeds,
            self.episodes.latent_evaluation_seed,
            *(item.evidence_seed for item in self.evaluation_environments),
        ]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Calibration, latent, and evidence seeds must be distinct")
        cases_per_episode = sum(item.cases_per_episode for item in self.calibration.probe_classes)
        if cases_per_episode != self.episodes.candidates_per_episode:
            raise ValueError("Probe-class allocation must match the episode size")
        if self.specification_hash != model_content_hash(self, exclude={"specification_hash"}):
            raise ValueError("Environment specification hash does not match its content")
        return self


class PolicyComparison(ContractModel):
    candidate: PolicyId
    matched_budget_comparators: tuple[PolicyId, ...] = Field(min_length=3, max_length=3)
    endpoints: tuple[PolicyId, ...] = Field(min_length=2, max_length=2)
    oracle_reference: PolicyId
    conservative_reliability_quantile: float = Field(gt=0.0, lt=0.5)
    bounded_log_likelihood_kappa: float = Field(gt=0.0)
    plug_in_reliability_summary: Literal["posterior_mean"] = "posterior_mean"
    reliability_draw_count: int = Field(ge=4_096)
    reliability_draw_seed: int = Field(ge=0)
    tie_break: Literal["ascending_case_id"] = "ascending_case_id"
    random_policy_seed: int = Field(ge=0)
    primary_comparison_isolates: Literal["acquisition_ranking_with_common_bounded_update"] = (
        "acquisition_ranking_with_common_bounded_update"
    )

    @model_validator(mode="after")
    def validate_policies(self) -> PolicyComparison:
        if self.candidate is not PolicyId.RELIABILITY_AWARE_BOUNDED:
            raise ValueError("The frozen candidate policy has changed")
        expected_comparators = (
            PolicyId.SEEDED_RANDOM_BOUNDED,
            PolicyId.UNCERTAINTY_ONLY_BOUNDED,
            PolicyId.PLUG_IN_EVSI_BOUNDED,
        )
        if self.matched_budget_comparators != expected_comparators:
            raise ValueError("Matched-budget comparators have changed")
        if self.endpoints != (PolicyId.NEVER_PROBE, PolicyId.ALWAYS_PROBE_BOUNDED):
            raise ValueError("Endpoint policies have changed")
        if self.oracle_reference is not PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED:
            raise ValueError("Oracle reference has changed")
        return self


class PrimaryAnalysis(ContractModel):
    metric: Literal["final_classification_error"] = "final_classification_error"
    classification_threshold: float = Field(gt=0.0, lt=1.0)
    unit_of_analysis: Literal["episode"] = "episode"
    primary_budget_fraction: float = Field(gt=0.0, lt=1.0)
    exact_selected_probes_per_episode: int = Field(ge=1)
    matched_environment: Literal[EvaluationEnvironmentId.MATCHED] = EvaluationEnvironmentId.MATCHED
    held_out_environments: tuple[EvaluationEnvironmentId, ...] = Field(min_length=6, max_length=6)
    aggregation: Literal["equal_mean_across_held_out_environments"] = (
        "equal_mean_across_held_out_environments"
    )
    estimand: Literal["candidate_minus_comparator_paired_episode_error"] = (
        "candidate_minus_comparator_paired_episode_error"
    )
    interval_method: Literal["stratified_paired_episode_percentile_bootstrap"] = (
        "stratified_paired_episode_percentile_bootstrap"
    )
    bootstrap_pairing: Literal[
        "same_episode_indices_within_environment_independent_between_environments"
    ] = "same_episode_indices_within_environment_independent_between_environments"
    bootstrap_repetitions: int = Field(ge=10_000)
    bootstrap_seed: int = Field(ge=0)
    familywise_alpha: float = Field(gt=0.0, lt=0.5)
    multiplicity_method: Literal["bonferroni_three_primary_comparisons"] = (
        "bonferroni_three_primary_comparisons"
    )
    simultaneous_confidence_level: float = Field(gt=0.95, lt=1.0)
    practical_difference_for_description: float = Field(gt=0.0, lt=0.1)
    precision_target_half_width: float = Field(gt=0.0, lt=0.1)
    precision_basis: Literal[
        "ten_independent_families_with_arbitrary_dependence_within_each_family"
    ] = "ten_independent_families_with_arbitrary_dependence_within_each_family"
    comparison_claim_rule: Literal["simultaneous_upper_interval_below_zero"] = (
        "simultaneous_upper_interval_below_zero"
    )
    overall_advantage_rule: Literal["comparison_claim_holds_for_all_three_comparators"] = (
        "comparison_claim_holds_for_all_three_comparators"
    )
    robustness_rule: Literal[
        "overall_advantage_and_no_held_out_point_regression_against_plug_in_evsi"
    ] = "overall_advantage_and_no_held_out_point_regression_against_plug_in_evsi"
    p_value_status: Literal["not_primary_intervals_drive_claims"] = (
        "not_primary_intervals_drive_claims"
    )

    @model_validator(mode="after")
    def validate_primary(self) -> PrimaryAnalysis:
        expected_held_out = tuple(EvaluationEnvironmentId)[1:]
        if self.held_out_environments != expected_held_out:
            raise ValueError("Primary analysis must include every held-out environment once")
        if not math.isclose(self.classification_threshold, 0.5, abs_tol=1e-12):
            raise ValueError("Classification threshold must remain 0.5")
        if not math.isclose(self.primary_budget_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("Primary probe budget must remain 50%")
        if not math.isclose(self.familywise_alpha, 0.05, abs_tol=1e-12):
            raise ValueError("Primary family-wise alpha must remain 0.05")
        expected_confidence = 1.0 - self.familywise_alpha / 3.0
        if not math.isclose(self.simultaneous_confidence_level, expected_confidence, abs_tol=1e-12):
            raise ValueError("Primary interval does not match the multiplicity rule")
        return self


class SecondaryAnalysis(ContractModel):
    budget_fractions: tuple[float, ...] = Field(min_length=5, max_length=5)
    metrics: tuple[str, ...] = Field(min_length=4, max_length=4)
    status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    asymmetric_false_positive_costs: tuple[float, ...] = Field(min_length=3, max_length=3)
    probability_floor: float = Field(gt=0.0, lt=0.01)
    ece_bin_count: int = Field(ge=5)
    calibration_seed_sensitivity: Literal[
        "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"
    ] = "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"

    @model_validator(mode="after")
    def validate_secondary(self) -> SecondaryAnalysis:
        if self.budget_fractions != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Secondary budget curve has changed")
        if self.metrics != (
            "classification_error",
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
        ):
            raise ValueError("Secondary metrics have changed")
        return self


class AnalysisSafeguards(ContractModel):
    paired_episode_key: Literal["environment_id_and_episode_id"] = "environment_id_and_episode_id"
    missing_probe_rule: Literal["consume_budget_keep_case_no_update"] = (
        "consume_budget_keep_case_no_update"
    )
    episode_exclusion_rule: Literal["none_after_generation"] = "none_after_generation"
    software_failure_rule: Literal["invalidate_run_fix_version_and_rerun"] = (
        "invalidate_run_fix_version_and_rerun"
    )
    canonical_evaluation_run_limit: Literal[1] = 1
    exact_retry_allowed: Literal[True] = True
    parameter_selection_after_evaluation_allowed: Literal[False] = False
    evaluation_outcomes_update_reliability_allowed: Literal[False] = False


class ClaimBoundary(ContractModel):
    simulator_prediction_claim_allowed: Literal[True] = True
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    cognitive_offloading_claim_allowed: Literal[False] = False
    deployed_cost_saving_claim_allowed: Literal[False] = False
    state_of_the_art_claim_allowed: Literal[False] = False
    matched_environment_alone_supports_robustness: Literal[False] = False


class AcquisitionAnalysisSpecification(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.analysis_specification.v1"] = (
        "acquisition_study.analysis_specification.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    status: Literal["frozen_before_policy_implementation"] = "frozen_before_policy_implementation"
    frozen_date: Literal["2026-09-05"] = "2026-09-05"
    research_question: str = Field(min_length=80)
    environment_specification_path: Literal["configs/acquisition-study/v1-environments.yaml"] = (
        "configs/acquisition-study/v1-environments.yaml"
    )
    environment_specification_hash: Sha256
    development_results_inspected_before_freeze: Literal[False] = False
    evaluation_results_inspected_before_freeze: Literal[False] = False
    external_model_data_used: Literal[False] = False
    human_data_used: Literal[False] = False
    policy_information_allowed: tuple[str, ...] = Field(min_length=5)
    policy_information_forbidden: tuple[str, ...] = Field(min_length=5)
    policies: PolicyComparison
    primary: PrimaryAnalysis
    secondary: SecondaryAnalysis
    ablations: tuple[AblationId, ...] = Field(min_length=4, max_length=4)
    negative_control: Literal["shuffle_probe_class_reliability_summaries"] = (
        "shuffle_probe_class_reliability_summaries"
    )
    safeguards: AnalysisSafeguards
    claims: ClaimBoundary
    interpretation_order: tuple[str, ...] = Field(min_length=7, max_length=7)
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> AcquisitionAnalysisSpecification:
        if self.ablations != tuple(AblationId):
            raise ValueError("Ablation order or coverage differs from the frozen design")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Analysis plan hash does not match its content")
        return self


def load_acquisition_study_plan(
    environment_path: Path,
    analysis_path: Path,
) -> tuple[AcquisitionEnvironmentSpecification, AcquisitionAnalysisSpecification]:
    environment = _load_yaml(environment_path, AcquisitionEnvironmentSpecification)
    analysis = _load_yaml(analysis_path, AcquisitionAnalysisSpecification)
    if analysis.environment_specification_hash != environment.specification_hash:
        raise AcquisitionStudyPlanError("Analysis plan refers to another environment specification")
    expected_budget = int(
        environment.episodes.candidates_per_episode * analysis.primary.primary_budget_fraction
    )
    if analysis.primary.exact_selected_probes_per_episode != expected_budget:
        raise AcquisitionStudyPlanError("Primary probe count does not equal the frozen budget")
    seeds = {
        environment.calibration.seed,
        environment.episodes.latent_evaluation_seed,
        *(item.evidence_seed for item in environment.evaluation_environments),
    }
    analysis_seeds = {
        analysis.policies.random_policy_seed,
        analysis.policies.reliability_draw_seed,
        analysis.primary.bootstrap_seed,
    }
    if analysis_seeds & seeds:
        raise AcquisitionStudyPlanError("Policy and analysis seeds must not generate study data")
    if len(analysis_seeds) != 3:
        raise AcquisitionStudyPlanError("Policy sampling and bootstrap require separate seeds")
    return environment, analysis


def _load_yaml[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return model.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise AcquisitionStudyPlanError(
            f"Could not verify acquisition-study input: {path}"
        ) from error
