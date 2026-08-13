"""Primary paired analysis for the acquisition-study development rehearsal."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

import numpy as np
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EnvironmentRole,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.runner import (
    DevelopmentPolicyMatrix,
    DevelopmentPolicyMatrixManifest,
    EpisodePolicyComparison,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_PRIMARY_COMPARATORS = (
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
)


class AcquisitionPrimaryAnalysisError(ValueError):
    """A source artifact or analysis result violates the frozen study design."""


class EnvironmentPairedEffect(ContractModel):
    """Paired episode-level errors for one environment and comparator."""

    environment_id: EvaluationEnvironmentId
    environment_role: EnvironmentRole
    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_id: PolicyId
    episode_count: int = Field(ge=1)
    candidate_mean_classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    comparator_mean_classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_paired_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    candidate_better_episode_count: int = Field(ge=0)
    tied_episode_count: int = Field(ge=0)
    comparator_better_episode_count: int = Field(ge=0)
    paired_effects_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.comparator_policy_id not in _PRIMARY_COMPARATORS:
            raise ValueError("Environment effect uses a non-primary comparator")
        if self.episode_count != sum(
            (
                self.candidate_better_episode_count,
                self.tied_episode_count,
                self.comparator_better_episode_count,
            )
        ):
            raise ValueError("Episode win, tie, and loss counts do not reconcile")
        expected_effect = (
            self.candidate_mean_classification_error - self.comparator_mean_classification_error
        )
        if not math.isclose(self.mean_paired_effect, expected_effect, abs_tol=1e-12):
            raise ValueError("Mean paired effect does not match the policy errors")
        return self


class ComparatorPrimaryInterval(ContractModel):
    """One multiplicity-adjusted held-out comparison from the frozen plan."""

    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_id: PolicyId
    estimand: Literal["candidate_minus_comparator_paired_episode_error"] = (
        "candidate_minus_comparator_paired_episode_error"
    )
    held_out_environment_count: Literal[6] = 6
    episodes_per_environment: int = Field(ge=1)
    environment_effects: tuple[EnvironmentPairedEffect, ...] = Field(
        min_length=6,
        max_length=6,
    )
    macro_mean_paired_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    confidence_level: float = Field(gt=0.95, lt=1.0, allow_inf_nan=False)
    interval_method: Literal["stratified_paired_episode_percentile_bootstrap"] = (
        "stratified_paired_episode_percentile_bootstrap"
    )
    bootstrap_pairing: Literal[
        "same_episode_indices_within_environment_independent_between_environments"
    ] = "same_episode_indices_within_environment_independent_between_environments"
    bootstrap_repetitions: int = Field(ge=10_000)
    bootstrap_seed: int = Field(ge=0)
    interval_lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    interval_upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    paired_effects_hash: Sha256
    bootstrap_distribution_hash: Sha256
    simultaneous_interval_rule_met_in_development: bool
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_mean_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.comparator_policy_id not in _PRIMARY_COMPARATORS:
            raise ValueError("Primary interval uses a non-primary comparator")
        expected_environments = tuple(EvaluationEnvironmentId)[1:]
        if tuple(row.environment_id for row in self.environment_effects) != expected_environments:
            raise ValueError("Primary interval does not contain the frozen held-out environments")
        if any(
            row.environment_role is not EnvironmentRole.HELD_OUT_MISMATCH
            or row.comparator_policy_id is not self.comparator_policy_id
            or row.episode_count != self.episodes_per_environment
            for row in self.environment_effects
        ):
            raise ValueError("Held-out environment summaries are inconsistent")
        expected_macro = fmean(row.mean_paired_effect for row in self.environment_effects)
        if not math.isclose(self.macro_mean_paired_effect, expected_macro, abs_tol=1e-12):
            raise ValueError("Macro-average effect does not give each environment equal weight")
        if self.interval_lower > self.interval_upper:
            raise ValueError("Bootstrap interval bounds are reversed")
        if self.simultaneous_interval_rule_met_in_development is not (self.interval_upper < 0.0):
            raise ValueError("Development interval flag does not match its upper bound")
        worst = max(self.environment_effects, key=lambda row: row.mean_paired_effect)
        if self.worst_environment_id is not worst.environment_id or not math.isclose(
            self.worst_environment_mean_effect,
            worst.mean_paired_effect,
            abs_tol=1e-12,
        ):
            raise ValueError("Worst-environment result does not match the environment summaries")
        return self


class DevelopmentPrimaryAnalysisPlan(ContractModel):
    """Provenance for one development-only primary analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_primary_analysis_plan.v1"] = (
        "acquisition_study.development_primary_analysis_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    source_manifest_hash: Sha256
    source_comparison_file_sha256: Sha256
    source_matrix_content_hash: Sha256
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    calibration_hash: Sha256
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    interval_method: Literal["stratified_paired_episode_percentile_bootstrap"] = (
        "stratified_paired_episode_percentile_bootstrap"
    )
    bootstrap_repetitions: int = Field(ge=10_000)
    bootstrap_seed: int = Field(ge=0)
    simultaneous_confidence_level: float = Field(gt=0.95, lt=1.0)
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Development primary-analysis plan hash does not match its content")
        return self


class DevelopmentPrimaryAnalysisReport(ContractModel):
    """Primary results that may diagnose code but cannot support the final claim."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_primary_analysis_report.v1"] = (
        "acquisition_study.development_primary_analysis_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str
    split: Literal["development"] = "development"
    result_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    analysis_execution_plan_hash: Sha256
    source_manifest_hash: Sha256
    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_ids: tuple[PolicyId, ...] = Field(min_length=3, max_length=3)
    matched_environment_effects: tuple[EnvironmentPairedEffect, ...] = Field(
        min_length=3,
        max_length=3,
    )
    held_out_primary_intervals: tuple[ComparatorPrimaryInterval, ...] = Field(
        min_length=3,
        max_length=3,
    )
    all_interval_rules_met_in_development: bool
    no_held_out_point_regression_against_plug_in_evsi_in_development: bool
    development_robustness_pattern_met: bool
    primary_decision_status: Literal["not_evaluated_on_development_split"] = (
        "not_evaluated_on_development_split"
    )
    inference_scope: Literal["independent_episodes_from_declared_simulator_only"] = (
        "independent_episodes_from_declared_simulator_only"
    )
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    cognitive_offloading_claim_supported: Literal[False] = False
    deployed_cost_saving_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.comparator_policy_ids != _PRIMARY_COMPARATORS:
            raise ValueError("Report comparator order differs from the frozen plan")
        if tuple(row.comparator_policy_id for row in self.matched_environment_effects) != (
            _PRIMARY_COMPARATORS
        ):
            raise ValueError("Matched summaries differ from the frozen comparator order")
        if any(
            row.environment_id is not EvaluationEnvironmentId.MATCHED
            or row.environment_role is not EnvironmentRole.MATCHED_SANITY_CHECK
            for row in self.matched_environment_effects
        ):
            raise ValueError("Matched summaries must remain a separate sanity check")
        if tuple(row.comparator_policy_id for row in self.held_out_primary_intervals) != (
            _PRIMARY_COMPARATORS
        ):
            raise ValueError("Held-out intervals differ from the frozen comparator order")
        expected_all = all(
            row.simultaneous_interval_rule_met_in_development
            for row in self.held_out_primary_intervals
        )
        if self.all_interval_rules_met_in_development is not expected_all:
            raise ValueError("All-comparison development flag is inconsistent")
        plug_in = next(
            row
            for row in self.held_out_primary_intervals
            if row.comparator_policy_id is PolicyId.PLUG_IN_EVSI_BOUNDED
        )
        expected_no_regression = all(
            row.mean_paired_effect <= 0.0 for row in plug_in.environment_effects
        )
        if (
            self.no_held_out_point_regression_against_plug_in_evsi_in_development
            is not expected_no_regression
        ):
            raise ValueError("Plug-in EVSI environment guardrail is inconsistent")
        if self.development_robustness_pattern_met is not (expected_all and expected_no_regression):
            raise ValueError("Development robustness pattern is inconsistent")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Development primary-analysis report hash does not match its content")
        return self


def run_development_primary_analysis(
    *,
    source_manifest_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    analysis_code_revision: str,
) -> DevelopmentPrimaryAnalysisReport:
    """Verify, analyse, and immutably publish development primary results."""

    manifest, matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    calibration = estimate_probe_reliability(specification)
    if calibration.calibration_hash != manifest.calibration_hash:
        raise AcquisitionPrimaryAnalysisError("Source matrix uses another calibration result")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise AcquisitionPrimaryAnalysisError(
            f"Could not read Pixi lock: {pixi_lock_path}"
        ) from error
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_primary_analysis_plan.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "split": matrix.split,
        "source_run_id": manifest.run_id,
        "source_code_revision": manifest.code_revision,
        "source_manifest_hash": manifest.manifest_hash,
        "source_comparison_file_sha256": manifest.comparison_file_sha256,
        "source_matrix_content_hash": manifest.matrix_content_hash,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "calibration_hash": calibration.calibration_hash,
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "interval_method": analysis.primary.interval_method,
        "bootstrap_repetitions": analysis.primary.bootstrap_repetitions,
        "bootstrap_seed": analysis.primary.bootstrap_seed,
        "simultaneous_confidence_level": analysis.primary.simultaneous_confidence_level,
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentPrimaryAnalysisPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    report = analyse_development_primary_matrix(matrix, analysis=analysis, plan=plan)
    write_immutable_json(output_root / "development_primary_analysis_plan.json", plan)
    write_immutable_json(output_root / "development_primary_analysis_report.json", report)
    return report


def load_verified_development_policy_matrix(
    manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentPolicyMatrixManifest, DevelopmentPolicyMatrix]:
    """Load a development matrix only after reconciling every source hash."""

    try:
        manifest = DevelopmentPolicyMatrixManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        comparison_path = manifest_path.parent / manifest.comparison_file
        comparison_bytes = comparison_path.read_bytes()
        if file_sha256(comparison_bytes) != manifest.comparison_file_sha256:
            raise AcquisitionPrimaryAnalysisError("Development comparison file hash differs")
        lines = comparison_bytes.decode("utf-8").splitlines()
        if len(lines) != manifest.comparison_count or any(not line for line in lines):
            raise AcquisitionPrimaryAnalysisError("Development comparison row count differs")
        comparisons = tuple(
            EpisodePolicyComparison.model_validate(json.loads(line)) for line in lines
        )
        matrix = DevelopmentPolicyMatrix(
            environment_specification_hash=manifest.environment_specification_hash,
            analysis_plan_hash=manifest.analysis_plan_hash,
            calibration_hash=manifest.calibration_hash,
            episodes_per_environment=manifest.episodes_per_environment,
            environment_count=manifest.environment_count,
            comparison_count=manifest.comparison_count,
            policy_count=manifest.policy_count,
            comparisons=comparisons,
        )
    except AcquisitionPrimaryAnalysisError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise AcquisitionPrimaryAnalysisError(
            f"Could not verify development policy matrix: {manifest_path}"
        ) from error
    if manifest.environment_specification_hash != specification.specification_hash:
        raise AcquisitionPrimaryAnalysisError(
            "Source matrix uses another environment specification"
        )
    if manifest.analysis_plan_hash != analysis.plan_hash:
        raise AcquisitionPrimaryAnalysisError("Source matrix uses another frozen analysis plan")
    if matrix.content_hash != manifest.matrix_content_hash:
        raise AcquisitionPrimaryAnalysisError("Development matrix content hash differs")
    return manifest, matrix


def analyse_development_primary_matrix(
    matrix: DevelopmentPolicyMatrix,
    *,
    analysis: AcquisitionAnalysisSpecification,
    plan: DevelopmentPrimaryAnalysisPlan,
) -> DevelopmentPrimaryAnalysisReport:
    """Apply the frozen held-out macro-average analysis to development episodes."""

    if matrix.split != "development" or plan.split != "development":
        raise AcquisitionPrimaryAnalysisError("Development analysis cannot inspect another split")
    if matrix.analysis_plan_hash != analysis.plan_hash:
        raise AcquisitionPrimaryAnalysisError("Matrix and frozen analysis plan differ")
    if analysis.policies.candidate is not _CANDIDATE:
        raise AcquisitionPrimaryAnalysisError("Candidate policy differs from the frozen plan")
    if analysis.policies.matched_budget_comparators != _PRIMARY_COMPARATORS:
        raise AcquisitionPrimaryAnalysisError("Comparator policies differ from the frozen plan")

    effects = _episode_effects(matrix)
    summaries = {
        (environment_id, comparator): _summarise_environment(
            matrix,
            effects[(environment_id, comparator)],
            environment_id=environment_id,
            comparator=comparator,
        )
        for environment_id in EvaluationEnvironmentId
        for comparator in _PRIMARY_COMPARATORS
    }
    distributions = stratified_paired_bootstrap_distributions(effects, analysis=analysis)
    alpha = 1.0 - analysis.primary.simultaneous_confidence_level
    primary_intervals: list[ComparatorPrimaryInterval] = []
    for comparator in _PRIMARY_COMPARATORS:
        environment_rows = tuple(
            summaries[(environment_id, comparator)]
            for environment_id in analysis.primary.held_out_environments
        )
        bootstrap = distributions[comparator]
        lower, upper = np.quantile(
            bootstrap,
            (alpha / 2.0, 1.0 - alpha / 2.0),
            method="linear",
        )
        worst = max(environment_rows, key=lambda row: row.mean_paired_effect)
        primary_intervals.append(
            ComparatorPrimaryInterval(
                comparator_policy_id=comparator,
                episodes_per_environment=matrix.episodes_per_environment,
                environment_effects=environment_rows,
                macro_mean_paired_effect=fmean(row.mean_paired_effect for row in environment_rows),
                confidence_level=analysis.primary.simultaneous_confidence_level,
                bootstrap_repetitions=analysis.primary.bootstrap_repetitions,
                bootstrap_seed=analysis.primary.bootstrap_seed,
                interval_lower=float(lower),
                interval_upper=float(upper),
                paired_effects_hash=canonical_sha256(
                    tuple(
                        (row.environment_id, effects[(row.environment_id, comparator)])
                        for row in environment_rows
                    )
                ),
                bootstrap_distribution_hash=canonical_sha256(bootstrap.tolist()),
                simultaneous_interval_rule_met_in_development=float(upper) < 0.0,
                worst_environment_id=worst.environment_id,
                worst_environment_mean_effect=worst.mean_paired_effect,
            )
        )

    intervals = tuple(primary_intervals)
    all_intervals_met = all(
        item.simultaneous_interval_rule_met_in_development for item in intervals
    )
    plug_in = next(
        item for item in intervals if item.comparator_policy_id is PolicyId.PLUG_IN_EVSI_BOUNDED
    )
    no_plugin_regression = all(row.mean_paired_effect <= 0.0 for row in plug_in.environment_effects)
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_primary_analysis_report.v1",
        "study_id": matrix.study_id,
        "run_id": plan.run_id,
        "split": matrix.split,
        "result_status": matrix.result_scope,
        "analysis_execution_plan_hash": plan.plan_hash,
        "source_manifest_hash": plan.source_manifest_hash,
        "candidate_policy_id": _CANDIDATE,
        "comparator_policy_ids": _PRIMARY_COMPARATORS,
        "matched_environment_effects": tuple(
            summaries[(EvaluationEnvironmentId.MATCHED, comparator)]
            for comparator in _PRIMARY_COMPARATORS
        ),
        "held_out_primary_intervals": intervals,
        "all_interval_rules_met_in_development": all_intervals_met,
        "no_held_out_point_regression_against_plug_in_evsi_in_development": (no_plugin_regression),
        "development_robustness_pattern_met": all_intervals_met and no_plugin_regression,
        "primary_decision_status": "not_evaluated_on_development_split",
        "inference_scope": "independent_episodes_from_declared_simulator_only",
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "cognitive_offloading_claim_supported": False,
        "deployed_cost_saving_claim_supported": False,
    }
    return DevelopmentPrimaryAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def _episode_effects(
    matrix: DevelopmentPolicyMatrix,
) -> dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]]:
    grouped: dict[tuple[EvaluationEnvironmentId, PolicyId], list[float]] = {
        (environment_id, comparator): []
        for environment_id in EvaluationEnvironmentId
        for comparator in _PRIMARY_COMPARATORS
    }
    for comparison in matrix.comparisons:
        by_policy = {result.policy_id: result for result in comparison.results}
        candidate_error = by_policy[_CANDIDATE].final_classification_error
        for comparator in _PRIMARY_COMPARATORS:
            grouped[(comparison.environment_id, comparator)].append(
                candidate_error - by_policy[comparator].final_classification_error
            )
    effects = {key: tuple(values) for key, values in grouped.items()}
    if any(len(values) != matrix.episodes_per_environment for values in effects.values()):
        raise AcquisitionPrimaryAnalysisError("Paired episode effects have incomplete coverage")
    return effects


def _summarise_environment(
    matrix: DevelopmentPolicyMatrix,
    effects: tuple[float, ...],
    *,
    environment_id: EvaluationEnvironmentId,
    comparator: PolicyId,
) -> EnvironmentPairedEffect:
    comparisons = tuple(
        comparison
        for comparison in matrix.comparisons
        if comparison.environment_id is environment_id
    )
    candidate_errors: list[float] = []
    comparator_errors: list[float] = []
    for comparison in comparisons:
        by_policy = {result.policy_id: result for result in comparison.results}
        candidate_errors.append(by_policy[_CANDIDATE].final_classification_error)
        comparator_errors.append(by_policy[comparator].final_classification_error)
    if environment_id is EvaluationEnvironmentId.MATCHED:
        role = EnvironmentRole.MATCHED_SANITY_CHECK
    else:
        role = EnvironmentRole.HELD_OUT_MISMATCH
    return EnvironmentPairedEffect(
        environment_id=environment_id,
        environment_role=role,
        comparator_policy_id=comparator,
        episode_count=len(effects),
        candidate_mean_classification_error=fmean(candidate_errors),
        comparator_mean_classification_error=fmean(comparator_errors),
        mean_paired_effect=fmean(effects),
        candidate_better_episode_count=sum(effect < 0.0 for effect in effects),
        tied_episode_count=sum(effect == 0.0 for effect in effects),
        comparator_better_episode_count=sum(effect > 0.0 for effect in effects),
        paired_effects_hash=canonical_sha256(effects),
    )


def stratified_paired_bootstrap_distributions(
    effects: dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]],
    *,
    analysis: AcquisitionAnalysisSpecification,
    chunk_size: int = 256,
) -> dict[PolicyId, np.ndarray]:
    held_out = analysis.primary.held_out_environments
    episode_count = len(effects[(held_out[0], _PRIMARY_COMPARATORS[0])])
    if episode_count < 1 or chunk_size < 1:
        raise AcquisitionPrimaryAnalysisError("Bootstrap requires episodes and a positive chunk")
    arrays = np.asarray(
        [
            [effects[(environment_id, comparator)] for comparator in _PRIMARY_COMPARATORS]
            for environment_id in held_out
        ],
        dtype=np.float64,
    )
    if arrays.shape != (
        len(held_out),
        len(_PRIMARY_COMPARATORS),
        episode_count,
    ) or not np.all(np.isfinite(arrays)):
        raise AcquisitionPrimaryAnalysisError("Bootstrap effects have an invalid shape or value")
    repetitions = analysis.primary.bootstrap_repetitions
    distributions = np.empty((repetitions, len(_PRIMARY_COMPARATORS)), dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(analysis.primary.bootstrap_seed))
    for start in range(0, repetitions, chunk_size):
        stop = min(repetitions, start + chunk_size)
        size = stop - start
        macro_means = np.zeros((size, len(_PRIMARY_COMPARATORS)), dtype=np.float64)
        for environment_index in range(len(held_out)):
            indices = generator.integers(0, episode_count, size=(size, episode_count))
            sampled = arrays[environment_index][:, indices]
            macro_means += np.mean(sampled, axis=2).T / len(held_out)
        distributions[start:stop] = macro_means
    return {
        comparator: distributions[:, index] for index, comparator in enumerate(_PRIMARY_COMPARATORS)
    }
