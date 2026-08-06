"""Prespecified analysis contract for the tracker robustness study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.config import (
    StressCondition,
    TrackerId,
    TrackerStudyConfiguration,
)


class TrackerStudyAnalysisSpecificationError(ValueError):
    """The tracker-study analysis specification is invalid or inconsistent."""


class PrimaryComparison(ContractModel):
    metric: Literal["brier_score"] = "brier_score"
    candidate_tracker: Literal[TrackerId.BOUNDED_CHANNEL_AWARE] = TrackerId.BOUNDED_CHANNEL_AWARE
    reference_tracker: Literal[TrackerId.CHANNEL_AWARE] = TrackerId.CHANNEL_AWARE
    adverse_conditions: tuple[StressCondition, ...] = Field(min_length=5, max_length=5)
    unit_of_analysis: Literal["episode_index"] = "episode_index"
    aggregation: Literal["equal_mean_over_turns_then_adverse_conditions"] = (
        "equal_mean_over_turns_then_adverse_conditions"
    )
    estimand: Literal["candidate_minus_reference_mean_adverse_brier"] = (
        "candidate_minus_reference_mean_adverse_brier"
    )
    favourable_direction: Literal["negative"] = "negative"
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    interval_method: Literal["paired_episode_percentile_bootstrap"] = (
        "paired_episode_percentile_bootstrap"
    )
    bootstrap_repetitions: Literal[10000] = 10000
    bootstrap_seed: int = Field(ge=0)
    test_method: Literal["paired_episode_sign_swap_mean"] = "paired_episode_sign_swap_mean"
    test_repetitions: Literal[100000] = 100000
    test_seed: int = Field(ge=0)
    clean_guardrail_metric: Literal["candidate_minus_reference_clean_brier"] = (
        "candidate_minus_reference_clean_brier"
    )
    maximum_clean_brier_degradation: float = Field(default=0.01, ge=0.0, le=1.0)
    decision_rule: Literal[
        "robust_if_primary_upper_interval_below_zero_and_clean_upper_interval_at_most_guardrail"
    ] = "robust_if_primary_upper_interval_below_zero_and_clean_upper_interval_at_most_guardrail"

    @model_validator(mode="after")
    def validate_primary_comparison(self) -> PrimaryComparison:
        expected = tuple(
            condition for condition in StressCondition if condition is not StressCondition.CLEAN
        )
        if self.adverse_conditions != expected:
            raise ValueError("Primary comparison must contain every adverse condition once")
        if self.bootstrap_seed == self.test_seed:
            raise ValueError("Bootstrap and sign-swap tests require separate seeds")
        if not math.isclose(self.confidence_level, 0.95, abs_tol=1e-12):
            raise ValueError("Primary confidence level must remain 0.95")
        if not math.isclose(self.maximum_clean_brier_degradation, 0.01, abs_tol=1e-12):
            raise ValueError("Clean Brier degradation guardrail must remain 0.01")
        return self


class MetricSpecification(ContractModel):
    reported_metrics: tuple[
        Literal[
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
            "classification_error",
            "mean_absolute_posterior_update",
        ],
        ...,
    ] = Field(min_length=5, max_length=5)
    probability_floor: float = Field(default=0.000001, gt=0.0, lt=0.01)
    classification_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    ece_bin_count: Literal[10] = 10
    ece_definition: Literal[
        "equal_width_probability_bins_weighted_absolute_mean_probability_minus_outcome"
    ] = "equal_width_probability_bins_weighted_absolute_mean_probability_minus_outcome"
    secondary_comparisons_adjusted_for_multiplicity: Literal[False] = False
    secondary_comparisons_status: Literal["descriptive_not_confirmatory"] = (
        "descriptive_not_confirmatory"
    )

    @model_validator(mode="after")
    def validate_metrics(self) -> MetricSpecification:
        expected = (
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
            "classification_error",
            "mean_absolute_posterior_update",
        )
        if self.reported_metrics != expected:
            raise ValueError("Reported metrics differ from the frozen order")
        if not math.isclose(self.probability_floor, 0.000001, abs_tol=1e-15):
            raise ValueError("Analysis probability floor must remain 0.000001")
        if not math.isclose(self.classification_threshold, 0.5, abs_tol=1e-12):
            raise ValueError("Classification threshold must remain 0.5")
        return self


class RuntimeSpecification(ContractModel):
    status: Literal["descriptive_platform_specific_not_inferential"] = (
        "descriptive_platform_specific_not_inferential"
    )
    warmup_repetitions: Literal[3] = 3
    measured_repetitions: Literal[20] = 20
    summary: Literal["median_nanoseconds_per_non_missing_update"] = (
        "median_nanoseconds_per_non_missing_update"
    )
    deterministic_operation_count_reported: Literal[True] = True


class SensitivitySpecification(ContractModel):
    use_configuration_grids_without_selection: Literal[True] = True
    status: Literal["descriptive_no_parameter_selection_after_test"] = (
        "descriptive_no_parameter_selection_after_test"
    )
    trust_weight_grid_source: Literal["configuration.sensitivity.trust_weight_multipliers"] = (
        "configuration.sensitivity.trust_weight_multipliers"
    )
    clipping_grid_source: Literal["configuration.sensitivity.clipping_kappas"] = (
        "configuration.sensitivity.clipping_kappas"
    )
    reliability_grid_source: Literal["configuration.sensitivity.reliability_scales"] = (
        "configuration.sensitivity.reliability_scales"
    )


class TrackerStudyAnalysisSpecification(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.analysis_specification.v1"] = (
        "tracker_study.analysis_specification.v1"
    )
    study_id: Literal["bayesian-evidence-robustness-v1"] = "bayesian-evidence-robustness-v1"
    configuration_hash: Sha256
    specification_status: Literal["prespecified_before_metric_analysis"] = (
        "prespecified_before_metric_analysis"
    )
    claim_scope: Literal["glass_box_simulator_robustness_not_human_cognition"] = (
        "glass_box_simulator_robustness_not_human_cognition"
    )
    development_results_inspected_before_freeze: Literal[False] = False
    held_out_benchmark_outcomes_used: Literal[False] = False
    external_model_data_used: Literal[False] = False
    human_data_used: Literal[False] = False
    primary: PrimaryComparison
    metrics: MetricSpecification
    runtime: RuntimeSpecification
    sensitivity: SensitivitySpecification
    interpretation_order: tuple[
        Literal[
            "primary_adverse_brier",
            "clean_performance_guardrail",
            "condition_level_metrics",
            "calibration_and_classification",
            "sensitivity",
            "runtime_and_operation_count",
            "simulator_limits",
        ],
        ...,
    ] = Field(min_length=7, max_length=7)
    analysis_specification_hash: Sha256

    @model_validator(mode="after")
    def validate_specification(self) -> TrackerStudyAnalysisSpecification:
        expected_order = (
            "primary_adverse_brier",
            "clean_performance_guardrail",
            "condition_level_metrics",
            "calibration_and_classification",
            "sensitivity",
            "runtime_and_operation_count",
            "simulator_limits",
        )
        if self.interpretation_order != expected_order:
            raise ValueError("Interpretation order differs from the prespecified order")
        expected_hash = model_content_hash(
            self,
            exclude={"analysis_specification_hash"},
        )
        if self.analysis_specification_hash != expected_hash:
            raise ValueError("Analysis specification hash does not match its content")
        return self


def load_tracker_study_analysis_specification(
    path: Path,
    configuration: TrackerStudyConfiguration,
) -> TrackerStudyAnalysisSpecification:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        specification = TrackerStudyAnalysisSpecification.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise TrackerStudyAnalysisSpecificationError(
            f"Could not verify tracker-study analysis specification: {path}"
        ) from error
    if specification.configuration_hash != configuration.configuration_hash:
        raise TrackerStudyAnalysisSpecificationError(
            "Analysis specification belongs to another tracker-study configuration"
        )
    if specification.metrics.probability_floor != configuration.trackers.probability_floor:
        raise TrackerStudyAnalysisSpecificationError(
            "Analysis probability floor differs from the tracker configuration"
        )
    if specification.metrics.ece_bin_count != configuration.experiment.ece_bin_count:
        raise TrackerStudyAnalysisSpecificationError(
            "Analysis calibration bins differ from the tracker configuration"
        )
    return specification
