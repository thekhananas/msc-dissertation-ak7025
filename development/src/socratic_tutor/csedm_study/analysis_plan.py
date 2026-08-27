"""Frozen analysis contract for the CSEDM uncertainty companion study."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.inventory import CSEDMInventoryError

PRIMARY_FEATURES = (
    "prior_event_count",
    "prior_submission_count",
    "prior_correctness_rate",
    "prior_hint_request_rate",
    "prior_distinct_problem_count",
    "training_learner_target_problem_success_rate",
)


class LogisticModelSpecification(ContractModel):
    """Fixed implementation choices for the primary learner-history model."""

    implementation: Literal["sklearn.linear_model.LogisticRegression"]
    solver: Literal["lbfgs"]
    penalty: Literal["l2"]
    regularisation_c: float = Field(gt=0.0)
    fit_intercept: Literal[True]
    class_weight: Literal["none"]
    maximum_iterations: Literal[1000]
    tolerance: float = Field(gt=0.0)
    random_seed: int = Field(ge=0)
    classification_threshold: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_numeric_settings(self) -> Self:
        if (self.regularisation_c, self.tolerance, self.classification_threshold) != (
            1.0,
            1e-8,
            0.5,
        ):
            raise ValueError("Logistic model settings differ from the frozen values")
        return self


class PrimaryAnalysisSpecification(ContractModel):
    """Prespecified question, selection rule, comparator, and estimand."""

    label: Literal["FirstCorrect"]
    prediction_scope: Literal["official_learner_separated_out_of_fold"]
    features: tuple[str, ...]
    model: LogisticModelSpecification
    uncertainty_score: Literal["absolute_probability_minus_one_half_ascending"]
    review_budget_fraction: float = Field(gt=0.0, lt=1.0)
    budget_scope: Literal["within_each_official_test_fold"]
    selected_count_rule: Literal["floor_budget_fraction_times_fold_target_count"]
    tie_break: Literal["ascending_sha256_of_fold_subject_problem_start_order"]
    outcome: Literal["captured_error_recall"]
    comparator: Literal["exact_uniform_random_expectation_at_same_fold_budgets"]
    estimand: Literal["uncertainty_ranked_minus_random_expected_captured_error_recall"]
    favourable_direction: Literal["greater_than_zero"]

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        if self.features != PRIMARY_FEATURES:
            raise ValueError("Primary CSEDM features differ from the frozen feature order")
        if self.review_budget_fraction != 0.5:
            raise ValueError("Primary review budget must remain at 50%")
        return self


class BootstrapSpecification(ContractModel):
    """Learner-level interval method for the primary contrast."""

    method: Literal["learner_cluster_percentile_bootstrap"]
    repetitions: Literal[10000]
    seed: int = Field(ge=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    resampling_unit: Literal["learner"]
    preserve_all_rows_per_sampled_learner: Literal[True]
    selected_status: Literal["fixed_from_original_out_of_fold_ranking"]

    @model_validator(mode="after")
    def validate_confidence_level(self) -> Self:
        if self.confidence_level != 0.95:
            raise ValueError("Bootstrap confidence level must remain at 95%")
        return self


class SecondaryAnalysisSpecification(ContractModel):
    """Descriptive checks that cannot replace the primary outcome."""

    prediction_metrics: tuple[
        Literal["accuracy", "brier_score", "log_loss", "expected_calibration_error"], ...
    ]
    log_loss_probability_floor: float = Field(gt=0.0, lt=0.5)
    ece_equal_width_bin_count: Literal[10]
    risk_coverage_fractions: tuple[float, ...]
    fold_sensitivity: Literal["leave_one_official_fold_out"]
    status: Literal["descriptive_not_confirmatory"]

    @model_validator(mode="after")
    def validate_secondary_settings(self) -> Self:
        if self.prediction_metrics != (
            "accuracy",
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        ):
            raise ValueError("Secondary metrics differ from the frozen order")
        if self.risk_coverage_fractions != tuple(index / 10 for index in range(1, 11)):
            raise ValueError("Risk-coverage fractions must be 0.1 through 1.0")
        if self.log_loss_probability_floor != 1e-6:
            raise ValueError("Log-loss probability floor must remain at 1e-6")
        return self


class ReferenceBaselineSpecification(ContractModel):
    """Published example values used to check source alignment, not select a model."""

    source_member: Literal["Example/evaluation_overall.csv"]
    check: Literal["recalculate_metrics_from_supplied_example_predictions"]
    expected_accuracy: float = Field(ge=0.0, le=1.0)
    expected_f1: float = Field(ge=0.0, le=1.0)
    expected_cohen_kappa: float = Field(ge=-1.0, le=1.0)
    absolute_tolerance: float = Field(gt=0.0)
    status: Literal["reference_check_not_primary_model"]

    @model_validator(mode="after")
    def validate_reference_values(self) -> Self:
        if (
            self.expected_accuracy,
            self.expected_f1,
            self.expected_cohen_kappa,
            self.absolute_tolerance,
        ) != (0.731138545953361, 0.699386503067485, 0.466364899386008, 1e-12):
            raise ValueError("Reference baseline values differ from the supplied v1.1 example")
        return self


class SafeguardSpecification(ContractModel):
    """Rules preventing outcome leakage and selective reporting."""

    history_cutoff: Literal["MainTable.Order_strictly_less_than_Predict.StartOrder"]
    forbidden_target_features: tuple[
        Literal["FirstCorrect", "EverCorrect", "UsedHint", "Attempts"], ...
    ]
    problem_statistics_fit_scope: Literal["training_learners_only"]
    missing_history_rule: Literal["training_only_defaults_plus_missingness_indicators"]
    invalid_label_rule: Literal["fail_run"]
    model_convergence_failure_rule: Literal["fail_run"]
    exclusion_rule: Literal["none_after_validated_inventory"]
    parameter_selection_after_results_allowed: Literal[False]


class ClaimBoundary(ContractModel):
    """Claims allowed and forbidden regardless of the observed result."""

    prediction_error_triage_on_this_dataset_allowed: Literal[True]
    executable_probe_benefit_allowed: Literal[False]
    tutoring_effect_allowed: Literal[False]
    human_learning_effect_allowed: Literal[False]
    cognitive_offloading_effect_allowed: Literal[False]
    deployment_cost_saving_allowed: Literal[False]


class CSEDMAnalysisPlan(ContractModel):
    """Complete prespecification for M7C before out-of-fold predictions exist."""

    schema_id: Literal["csedm.analysis_plan.v1"]
    schema_version: Literal[1]
    study_id: Literal["csedm-uncertainty-triage-v1"]
    status: Literal["frozen_before_out_of_fold_predictions"]
    frozen_date: date
    research_question: str = Field(min_length=80)
    inventory_report_hash: Sha256
    inventory_report_file_sha256: Sha256
    archive_sha256: Sha256
    official_fold_ids: tuple[int, ...]
    out_of_fold_results_inspected_before_freeze: Literal[False]
    primary: PrimaryAnalysisSpecification
    bootstrap: BootstrapSpecification
    secondary: SecondaryAnalysisSpecification
    reference_baseline: ReferenceBaselineSpecification
    safeguards: SafeguardSpecification
    claims: ClaimBoundary
    interpretation_order: tuple[str, ...] = Field(min_length=5)
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.official_fold_ids != tuple(range(10)):
            raise ValueError("The analysis must use all ten official folds in order")
        if len(self.interpretation_order) != len(set(self.interpretation_order)):
            raise ValueError("Interpretation steps must be unique")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("CSEDM analysis plan hash does not match its content")
        return self


def load_analysis_plan(path: Path) -> CSEDMAnalysisPlan:
    """Load and verify the frozen M7C analysis plan."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CSEDMInventoryError(f"Could not read CSEDM analysis plan: {path}") from error
    return CSEDMAnalysisPlan.model_validate(payload)
