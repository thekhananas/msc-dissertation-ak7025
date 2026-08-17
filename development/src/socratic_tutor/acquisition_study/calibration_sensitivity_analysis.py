"""Development analysis across the frozen calibration-seed sensitivity runs."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean, median
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.calibration_sensitivity import (
    CalibrationSensitivityEpisodeMetric,
    DevelopmentCalibrationSensitivityManifest,
    DevelopmentCalibrationSensitivityMatrix,
    SensitivityCalibrationRecord,
    load_verified_development_calibration_sensitivity,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EnvironmentRole,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    stratified_paired_bootstrap_distributions,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_COMPARATORS = (
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
)
_HELD_OUT_ENVIRONMENTS = tuple(EvaluationEnvironmentId)[1:]


class CalibrationSensitivityAnalysisError(ValueError):
    """A calibration-sensitivity artifact cannot support the frozen analysis."""


class CalibrationSensitivityEnvironmentEffect(ContractModel):
    """Candidate-minus-comparator error under one calibration and environment."""

    environment_id: EvaluationEnvironmentId
    environment_role: EnvironmentRole
    episode_count: int = Field(ge=1)
    candidate_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    comparator_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    mean_paired_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    paired_differences_hash: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        expected_role = (
            EnvironmentRole.MATCHED_SANITY_CHECK
            if self.environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        )
        if self.environment_role is not expected_role:
            raise ValueError("Calibration-sensitivity effect uses the wrong environment role")
        expected = (
            self.candidate_mean_classification_error - self.comparator_mean_classification_error
        )
        if not math.isclose(self.mean_paired_difference, expected, abs_tol=1e-12):
            raise ValueError("Calibration-sensitivity effect does not match its policy means")
        return self


class CalibrationSeedComparatorResult(ContractModel):
    """Frozen primary comparison for one calibration seed and comparator."""

    comparator_policy_id: PolicyId
    environment_effects: tuple[CalibrationSensitivityEnvironmentEffect, ...] = Field(
        min_length=7,
        max_length=7,
    )
    held_out_macro_mean_paired_difference: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    simultaneous_confidence_level: float = Field(gt=0.95, lt=1.0, allow_inf_nan=False)
    interval_lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    interval_upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    simultaneous_interval_rule_met: bool
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_mean_difference: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    bootstrap_distribution_hash: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.comparator_policy_id not in _COMPARATORS:
            raise ValueError("Calibration sensitivity uses an undeclared comparator")
        if tuple(row.environment_id for row in self.environment_effects) != tuple(
            EvaluationEnvironmentId
        ):
            raise ValueError("Calibration sensitivity has incomplete environment coverage")
        held_out = tuple(
            row for row in self.environment_effects if row.environment_id in _HELD_OUT_ENVIRONMENTS
        )
        expected_mean = fmean(row.mean_paired_difference for row in held_out)
        if not math.isclose(
            self.held_out_macro_mean_paired_difference,
            expected_mean,
            abs_tol=1e-12,
        ):
            raise ValueError("Calibration-sensitivity held-out effect is inconsistent")
        worst = max(held_out, key=lambda row: row.mean_paired_difference)
        if self.worst_environment_id is not worst.environment_id or not math.isclose(
            self.worst_environment_mean_difference,
            worst.mean_paired_difference,
            abs_tol=1e-12,
        ):
            raise ValueError("Calibration-sensitivity worst environment is inconsistent")
        if self.interval_lower > self.interval_upper:
            raise ValueError("Calibration-sensitivity interval bounds are reversed")
        if self.simultaneous_interval_rule_met != (self.interval_upper < 0.0):
            raise ValueError("Calibration-sensitivity interval decision is inconsistent")
        return self


class CalibrationSeedAnalysisResult(ContractModel):
    """Complete frozen primary analysis under one alternative calibration sample."""

    calibration: SensitivityCalibrationRecord
    comparator_results: tuple[CalibrationSeedComparatorResult, ...] = Field(
        min_length=3,
        max_length=3,
    )
    all_interval_rules_met: bool
    no_held_out_point_regression_against_plug_in_evsi: bool
    robustness_pattern_met: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if tuple(row.comparator_policy_id for row in self.comparator_results) != _COMPARATORS:
            raise ValueError("Calibration seed result has incomplete comparator coverage")
        expected_intervals = all(
            row.simultaneous_interval_rule_met for row in self.comparator_results
        )
        plug_in = self.comparator_results[-1]
        expected_no_regression = all(
            row.mean_paired_difference <= 0.0
            for row in plug_in.environment_effects
            if row.environment_id in _HELD_OUT_ENVIRONMENTS
        )
        if (
            self.all_interval_rules_met != expected_intervals
            or self.no_held_out_point_regression_against_plug_in_evsi != expected_no_regression
            or self.robustness_pattern_met != (expected_intervals and expected_no_regression)
        ):
            raise ValueError("Calibration seed robustness classification is inconsistent")
        return self


class ComparatorCalibrationSensitivitySummary(ContractModel):
    """Variation in one primary comparison across the 20 calibration samples."""

    comparator_policy_id: PolicyId
    calibration_seed_count: Literal[20] = 20
    mean_held_out_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    median_held_out_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    minimum_held_out_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    maximum_held_out_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    point_advantage_seed_count: int = Field(ge=0, le=20)
    simultaneous_interval_rule_seed_count: int = Field(ge=0, le=20)
    seed_effects_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.comparator_policy_id not in _COMPARATORS:
            raise ValueError("Sensitivity summary uses an undeclared comparator")
        if not (
            self.minimum_held_out_effect
            <= self.median_held_out_effect
            <= self.maximum_held_out_effect
        ):
            raise ValueError("Sensitivity effect range is inconsistent")
        return self


class DevelopmentCalibrationSensitivityAnalysisPlan(ContractModel):
    """Provenance and frozen rules for the calibration-sensitivity analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_calibration_sensitivity_analysis_plan.v1"] = (
        "acquisition_study.development_calibration_sensitivity_analysis_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    source_sensitivity_manifest_hash: Sha256
    source_sensitivity_file_sha256: Sha256
    source_sensitivity_matrix_hash: Sha256
    source_primary_manifest_hash: Sha256
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    calibration_seed_count: Literal[20] = 20
    comparator_policy_ids: tuple[PolicyId, ...] = Field(min_length=3, max_length=3)
    metric: Literal["final_classification_error"] = "final_classification_error"
    aggregation: Literal["equal_mean_across_held_out_environments"] = (
        "equal_mean_across_held_out_environments"
    )
    interval_method: Literal["stratified_paired_episode_percentile_bootstrap"] = (
        "stratified_paired_episode_percentile_bootstrap"
    )
    bootstrap_repetitions: int = Field(ge=1000)
    bootstrap_seed: int = Field(ge=0)
    simultaneous_confidence_level: float = Field(gt=0.95, lt=1.0, allow_inf_nan=False)
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.comparator_policy_ids != _COMPARATORS:
            raise ValueError("Sensitivity analysis plan differs from the primary comparators")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Sensitivity analysis plan hash does not match its content")
        return self


class DevelopmentCalibrationSensitivityAnalysisReport(ContractModel):
    """Development-only calibration sensitivity without canonical claims."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_calibration_sensitivity_analysis.v1"] = (
        "acquisition_study.development_calibration_sensitivity_analysis.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    result_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    analysis_execution_plan_hash: Sha256
    source_sensitivity_manifest_hash: Sha256
    source_sensitivity_matrix_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    calibration_results: tuple[CalibrationSeedAnalysisResult, ...] = Field(
        min_length=20,
        max_length=20,
    )
    comparator_summaries: tuple[ComparatorCalibrationSensitivitySummary, ...] = Field(
        min_length=3,
        max_length=3,
    )
    robustness_pattern_seed_count: int = Field(ge=0, le=20)
    robustness_pattern_seed_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sensitivity_status: Literal[
        "robustness_pattern_present_all_seeds",
        "robustness_pattern_varies_by_seed",
        "robustness_pattern_absent_all_seeds",
    ]
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    privileged_source_reproduction_verified: Literal[True] = True
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if tuple(row.comparator_policy_id for row in self.comparator_summaries) != _COMPARATORS:
            raise ValueError("Sensitivity report has incomplete comparator coverage")
        if self.robustness_pattern_seed_count != sum(
            row.robustness_pattern_met for row in self.calibration_results
        ):
            raise ValueError("Sensitivity robustness count is inconsistent")
        expected_fraction = self.robustness_pattern_seed_count / len(self.calibration_results)
        if not math.isclose(
            self.robustness_pattern_seed_fraction,
            expected_fraction,
            abs_tol=1e-12,
        ):
            raise ValueError("Sensitivity robustness fraction is inconsistent")
        expected_status = (
            "robustness_pattern_present_all_seeds"
            if self.robustness_pattern_seed_count == len(self.calibration_results)
            else "robustness_pattern_absent_all_seeds"
            if self.robustness_pattern_seed_count == 0
            else "robustness_pattern_varies_by_seed"
        )
        if self.sensitivity_status != expected_status:
            raise ValueError("Sensitivity status is inconsistent")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Sensitivity report hash does not match its content")
        return self


def run_development_calibration_sensitivity_analysis(
    *,
    source_manifest_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    analysis_code_revision: str,
) -> DevelopmentCalibrationSensitivityAnalysisReport:
    """Verify, analyse, and publish development calibration sensitivity."""

    source_manifest, matrix = load_verified_development_calibration_sensitivity(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise CalibrationSensitivityAnalysisError(
            f"Could not read Pixi lock: {pixi_lock_path}"
        ) from error
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_calibration_sensitivity_analysis_plan.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "split": matrix.split,
        "source_run_id": source_manifest.run_id,
        "source_code_revision": source_manifest.code_revision,
        "source_sensitivity_manifest_hash": source_manifest.manifest_hash,
        "source_sensitivity_file_sha256": source_manifest.result_file_sha256,
        "source_sensitivity_matrix_hash": matrix.content_hash,
        "source_primary_manifest_hash": matrix.source_manifest_hash,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "calibration_seed_count": len(matrix.calibrations),
        "comparator_policy_ids": _COMPARATORS,
        "metric": analysis.primary.metric,
        "aggregation": analysis.primary.aggregation,
        "interval_method": analysis.primary.interval_method,
        "bootstrap_repetitions": analysis.primary.bootstrap_repetitions,
        "bootstrap_seed": analysis.primary.bootstrap_seed,
        "simultaneous_confidence_level": analysis.primary.simultaneous_confidence_level,
        "analysis_status": analysis.secondary.status,
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentCalibrationSensitivityAnalysisPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    report = analyse_development_calibration_sensitivity(
        matrix,
        analysis=analysis,
        source_manifest=source_manifest,
        plan=plan,
    )
    write_immutable_json(
        output_root / "development_calibration_sensitivity_analysis_plan.json",
        plan,
    )
    write_immutable_json(
        output_root / "development_calibration_sensitivity_analysis_report.json",
        report,
    )
    return report


def analyse_development_calibration_sensitivity(
    matrix: DevelopmentCalibrationSensitivityMatrix,
    *,
    analysis: AcquisitionAnalysisSpecification,
    source_manifest: DevelopmentCalibrationSensitivityManifest,
    plan: DevelopmentCalibrationSensitivityAnalysisPlan,
) -> DevelopmentCalibrationSensitivityAnalysisReport:
    """Repeat the frozen primary analysis for every alternative calibration seed."""

    if matrix.split != "development" or matrix.analysis_plan_hash != analysis.plan_hash:
        raise CalibrationSensitivityAnalysisError("Sensitivity matrix and analysis plan differ")
    if (
        source_manifest.matrix_content_hash != matrix.content_hash
        or source_manifest.analysis_plan_hash != analysis.plan_hash
    ):
        raise CalibrationSensitivityAnalysisError("Sensitivity matrix differs from its manifest")
    if (
        plan.source_sensitivity_manifest_hash != source_manifest.manifest_hash
        or plan.source_sensitivity_matrix_hash != matrix.content_hash
        or plan.frozen_analysis_specification_hash != analysis.plan_hash
    ):
        raise CalibrationSensitivityAnalysisError("Sensitivity plan differs from its sources")

    rows = {
        (
            row.calibration_seed,
            row.metric.environment_id,
            row.metric.episode_index,
            row.metric.policy_id,
        ): row
        for row in matrix.rows
    }
    calibration_results = tuple(
        _analyse_calibration_seed(
            calibration,
            rows,
            analysis=analysis,
            episodes_per_environment=matrix.episodes_per_environment,
        )
        for calibration in matrix.calibrations
    )
    comparator_summaries = tuple(
        _summarise_comparator(calibration_results, comparator) for comparator in _COMPARATORS
    )
    robustness_count = sum(row.robustness_pattern_met for row in calibration_results)
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_calibration_sensitivity_analysis.v1",
        "study_id": matrix.study_id,
        "run_id": plan.run_id,
        "split": matrix.split,
        "result_status": matrix.result_scope,
        "analysis_execution_plan_hash": plan.plan_hash,
        "source_sensitivity_manifest_hash": source_manifest.manifest_hash,
        "source_sensitivity_matrix_hash": matrix.content_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "calibration_results": calibration_results,
        "comparator_summaries": comparator_summaries,
        "robustness_pattern_seed_count": robustness_count,
        "robustness_pattern_seed_fraction": robustness_count / len(calibration_results),
        "sensitivity_status": (
            "robustness_pattern_present_all_seeds"
            if robustness_count == len(calibration_results)
            else "robustness_pattern_absent_all_seeds"
            if robustness_count == 0
            else "robustness_pattern_varies_by_seed"
        ),
        "analysis_status": plan.analysis_status,
        "privileged_source_reproduction_verified": (matrix.privileged_source_reproduction_verified),
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    return DevelopmentCalibrationSensitivityAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def _analyse_calibration_seed(
    calibration: SensitivityCalibrationRecord,
    rows: dict[
        tuple[int, EvaluationEnvironmentId, int, PolicyId],
        CalibrationSensitivityEpisodeMetric,
    ],
    *,
    analysis: AcquisitionAnalysisSpecification,
    episodes_per_environment: int,
) -> CalibrationSeedAnalysisResult:
    effects: dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]] = {}
    environment_results: dict[
        tuple[EvaluationEnvironmentId, PolicyId],
        CalibrationSensitivityEnvironmentEffect,
    ] = {}
    for environment_id in EvaluationEnvironmentId:
        candidate_values = tuple(
            rows[
                (calibration.calibration_seed, environment_id, episode_index, _CANDIDATE)
            ].metric.final_classification_error
            for episode_index in range(episodes_per_environment)
        )
        for comparator in _COMPARATORS:
            comparator_values = tuple(
                rows[
                    (calibration.calibration_seed, environment_id, episode_index, comparator)
                ].metric.final_classification_error
                for episode_index in range(episodes_per_environment)
            )
            differences = tuple(
                candidate - reference
                for candidate, reference in zip(
                    candidate_values,
                    comparator_values,
                    strict=True,
                )
            )
            effects[(environment_id, comparator)] = differences
            environment_results[(environment_id, comparator)] = (
                CalibrationSensitivityEnvironmentEffect(
                    environment_id=environment_id,
                    environment_role=(
                        EnvironmentRole.MATCHED_SANITY_CHECK
                        if environment_id is EvaluationEnvironmentId.MATCHED
                        else EnvironmentRole.HELD_OUT_MISMATCH
                    ),
                    episode_count=episodes_per_environment,
                    candidate_mean_classification_error=fmean(candidate_values),
                    comparator_mean_classification_error=fmean(comparator_values),
                    mean_paired_difference=fmean(differences),
                    paired_differences_hash=canonical_sha256(differences),
                )
            )

    distributions = stratified_paired_bootstrap_distributions(effects, analysis=analysis)
    alpha = 1.0 - analysis.primary.simultaneous_confidence_level
    comparator_results: list[CalibrationSeedComparatorResult] = []
    for comparator in _COMPARATORS:
        environment_effects = tuple(
            environment_results[(environment_id, comparator)]
            for environment_id in EvaluationEnvironmentId
        )
        held_out = tuple(
            row for row in environment_effects if row.environment_id in _HELD_OUT_ENVIRONMENTS
        )
        distribution = distributions[comparator]
        lower, upper = np.quantile(
            distribution,
            (alpha / 2.0, 1.0 - alpha / 2.0),
            method="linear",
        )
        worst = max(held_out, key=lambda row: row.mean_paired_difference)
        comparator_results.append(
            CalibrationSeedComparatorResult(
                comparator_policy_id=comparator,
                environment_effects=environment_effects,
                held_out_macro_mean_paired_difference=fmean(
                    row.mean_paired_difference for row in held_out
                ),
                simultaneous_confidence_level=(analysis.primary.simultaneous_confidence_level),
                interval_lower=float(lower),
                interval_upper=float(upper),
                simultaneous_interval_rule_met=float(upper) < 0.0,
                worst_environment_id=worst.environment_id,
                worst_environment_mean_difference=worst.mean_paired_difference,
                bootstrap_distribution_hash=canonical_sha256(distribution.tolist()),
            )
        )
    results = tuple(comparator_results)
    all_intervals = all(row.simultaneous_interval_rule_met for row in results)
    plug_in = results[-1]
    no_plugin_regression = all(
        row.mean_paired_difference <= 0.0
        for row in plug_in.environment_effects
        if row.environment_id in _HELD_OUT_ENVIRONMENTS
    )
    return CalibrationSeedAnalysisResult(
        calibration=calibration,
        comparator_results=results,
        all_interval_rules_met=all_intervals,
        no_held_out_point_regression_against_plug_in_evsi=no_plugin_regression,
        robustness_pattern_met=all_intervals and no_plugin_regression,
    )


def _summarise_comparator(
    calibration_results: tuple[CalibrationSeedAnalysisResult, ...],
    comparator: PolicyId,
) -> ComparatorCalibrationSensitivitySummary:
    results = tuple(
        next(
            row
            for row in calibration_result.comparator_results
            if row.comparator_policy_id is comparator
        )
        for calibration_result in calibration_results
    )
    effects = tuple(row.held_out_macro_mean_paired_difference for row in results)
    return ComparatorCalibrationSensitivitySummary(
        comparator_policy_id=comparator,
        mean_held_out_effect=fmean(effects),
        median_held_out_effect=median(effects),
        minimum_held_out_effect=min(effects),
        maximum_held_out_effect=max(effects),
        point_advantage_seed_count=sum(value < 0.0 for value in effects),
        simultaneous_interval_rule_seed_count=sum(
            row.simultaneous_interval_rule_met for row in results
        ),
        seed_effects_hash=canonical_sha256(effects),
    )
