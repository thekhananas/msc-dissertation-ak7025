"""Descriptive component analysis for the development acquisition ablations."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.ablation_runner import (
    AblationEpisodeMetric,
    DevelopmentAblationManifest,
    DevelopmentAblationMatrix,
    load_verified_development_ablation_matrix,
)
from socratic_tutor.acquisition_study.ablations import AblationPolicyCellId
from socratic_tutor.acquisition_study.budget_curve_analysis import (
    EnvironmentBudgetSummary,
    summarise_environment_budget,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import ProbeClassId
from socratic_tutor.acquisition_study.plan import (
    AblationId,
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EnvironmentRole,
    EvaluationEnvironmentId,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_HELD_OUT_ENVIRONMENTS = tuple(EvaluationEnvironmentId)[1:]
_PAIRWISE_ABLATIONS = (
    (
        AblationId.REPLACE_LOWER_QUANTILE_WITH_POSTERIOR_MEAN,
        AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED,
    ),
    (
        AblationId.REMOVE_BOUNDED_UPDATE_FROM_CANDIDATE,
        AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED,
    ),
    (
        AblationId.REMOVE_BOTH_RELIABILITY_CONSERVATISM_AND_BOUNDED_UPDATE,
        AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED,
    ),
)
_SELECTION_ABLATION = AblationId.COMPARE_DENSE_AND_SPARSE_EQUAL_EXPECTED_RELIABILITY


class AblationAnalysisError(ValueError):
    """A development ablation artifact cannot support the frozen analysis."""


class ProbeClassSelectionSummary(ContractModel):
    """Aggregated selection frequency for one probe class."""

    probe_class: ProbeClassId
    available_case_count: int = Field(ge=1)
    selected_case_count: int = Field(ge=0)
    selection_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.selected_case_count > self.available_case_count:
            raise ValueError("Selected probe-class count exceeds availability")
        expected = self.selected_case_count / self.available_case_count
        if not math.isclose(self.selection_rate, expected, abs_tol=1e-12):
            raise ValueError("Probe-class selection rate does not match its counts")
        return self


class AblationEnvironmentSummary(ContractModel):
    """One cell's pooled metrics and selections in one environment."""

    cell_id: AblationPolicyCellId
    reliability_summary: Literal["lower_quantile", "posterior_mean"]
    update_rule: Literal["bounded", "unbounded"]
    metrics: EnvironmentBudgetSummary
    selection_by_probe_class: tuple[ProbeClassSelectionSummary, ...] = Field(
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        expected = {
            AblationPolicyCellId.FULL_CANDIDATE: ("lower_quantile", "bounded"),
            AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED: ("posterior_mean", "bounded"),
            AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED: (
                "lower_quantile",
                "unbounded",
            ),
            AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED: (
                "posterior_mean",
                "unbounded",
            ),
        }[self.cell_id]
        if (self.reliability_summary, self.update_rule) != expected:
            raise ValueError("Ablation summary does not match its component cell")
        if tuple(row.probe_class for row in self.selection_by_probe_class) != tuple(ProbeClassId):
            raise ValueError("Ablation summary does not cover every probe class")
        if sum(row.available_case_count for row in self.selection_by_probe_class) != (
            self.metrics.case_prediction_count
        ):
            raise ValueError("Ablation class availability does not match the case count")
        if sum(row.selected_case_count for row in self.selection_by_probe_class) != (
            self.metrics.selected_probe_count
        ):
            raise ValueError("Ablation class selections do not match the probe count")
        return self


class HeldOutAblationCellSummary(ContractModel):
    """Equal-environment metrics for one cell over held-out mismatch settings."""

    cell_id: AblationPolicyCellId
    environment_count: Literal[6] = 6
    episodes_per_environment: int = Field(ge=1)
    macro_mean_classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_mean_brier_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_mean_negative_log_likelihood: float = Field(ge=0.0, allow_inf_nan=False)
    macro_mean_expected_calibration_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    source_summaries_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.worst_environment_id not in _HELD_OUT_ENVIRONMENTS:
            raise ValueError("Worst ablation environment must be held out")
        return self


class EnvironmentAblationEffect(ContractModel):
    """Full-candidate minus ablated error for paired episodes in one environment."""

    ablation_id: AblationId
    ablated_cell_id: AblationPolicyCellId
    environment_id: EvaluationEnvironmentId
    environment_role: EnvironmentRole
    episode_count: int = Field(ge=1)
    full_candidate_mean_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ablated_mean_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_paired_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    full_candidate_better_episode_count: int = Field(ge=0)
    tied_episode_count: int = Field(ge=0)
    ablated_better_episode_count: int = Field(ge=0)
    paired_differences_hash: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        expected_cell = dict(_PAIRWISE_ABLATIONS).get(self.ablation_id)
        if expected_cell is None or self.ablated_cell_id is not expected_cell:
            raise ValueError("Ablation effect uses an undeclared cell comparison")
        expected_role = (
            EnvironmentRole.MATCHED_SANITY_CHECK
            if self.environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        )
        if self.environment_role is not expected_role:
            raise ValueError("Ablation effect uses the wrong environment role")
        if self.episode_count != sum(
            (
                self.full_candidate_better_episode_count,
                self.tied_episode_count,
                self.ablated_better_episode_count,
            )
        ):
            raise ValueError("Ablation win, tie, and loss counts do not reconcile")
        expected_difference = self.full_candidate_mean_error - self.ablated_mean_error
        if not math.isclose(self.mean_paired_difference, expected_difference, abs_tol=1e-12):
            raise ValueError("Ablation effect does not match its cell means")
        return self


class HeldOutAblationEffect(ContractModel):
    """Equal-environment paired effect for one component removal."""

    ablation_id: AblationId
    ablated_cell_id: AblationPolicyCellId
    environment_count: Literal[6] = 6
    episodes_per_environment: int = Field(ge=1)
    paired_episode_count: int = Field(ge=1)
    full_candidate_macro_mean_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    ablated_macro_mean_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_mean_paired_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    full_candidate_better_episode_count: int = Field(ge=0)
    tied_episode_count: int = Field(ge=0)
    ablated_better_episode_count: int = Field(ge=0)
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_mean_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    environment_effects_hash: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        expected_cell = dict(_PAIRWISE_ABLATIONS).get(self.ablation_id)
        if expected_cell is None or self.ablated_cell_id is not expected_cell:
            raise ValueError("Held-out ablation uses an undeclared cell comparison")
        if self.paired_episode_count != self.environment_count * self.episodes_per_environment:
            raise ValueError("Held-out ablation episode count is inconsistent")
        if self.paired_episode_count != sum(
            (
                self.full_candidate_better_episode_count,
                self.tied_episode_count,
                self.ablated_better_episode_count,
            )
        ):
            raise ValueError("Held-out ablation win, tie, and loss counts do not reconcile")
        expected_difference = self.full_candidate_macro_mean_error - self.ablated_macro_mean_error
        if not math.isclose(
            self.macro_mean_paired_difference,
            expected_difference,
            abs_tol=1e-12,
        ):
            raise ValueError("Held-out ablation effect does not match its cell means")
        if self.worst_environment_id not in _HELD_OUT_ENVIRONMENTS:
            raise ValueError("Worst ablation effect must come from a held-out environment")
        return self


class HeldOutFactorialInteraction(ContractModel):
    """Descriptive interaction between reliability summary and update bound."""

    effect_direction: Literal["first_named_cell_minus_second_negative_favours_first"] = (
        "first_named_cell_minus_second_negative_favours_first"
    )
    lower_quantile_effect_when_bounded: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    lower_quantile_effect_when_unbounded: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    bounded_update_effect_with_lower_quantile: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    bounded_update_effect_with_posterior_mean: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    interaction_difference: float = Field(ge=-2.0, le=2.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_interaction(self) -> Self:
        reliability_interaction = (
            self.lower_quantile_effect_when_bounded - self.lower_quantile_effect_when_unbounded
        )
        update_interaction = (
            self.bounded_update_effect_with_lower_quantile
            - self.bounded_update_effect_with_posterior_mean
        )
        if not (
            math.isclose(reliability_interaction, update_interaction, abs_tol=1e-12)
            and math.isclose(
                self.interaction_difference,
                reliability_interaction,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Factorial ablation interaction is inconsistent")
        return self


class DenseSparseSelectionEffect(ContractModel):
    """Selection-rate contrast for equally accurate dense and sparse classes."""

    ablation_id: Literal[AblationId.COMPARE_DENSE_AND_SPARSE_EQUAL_EXPECTED_RELIABILITY] = (
        _SELECTION_ABLATION
    )
    cell_id: AblationPolicyCellId
    environment_count: Literal[7] = 7
    episode_count: int = Field(ge=1)
    dense_available_count: int = Field(ge=1)
    dense_selected_count: int = Field(ge=0)
    dense_selection_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    sparse_available_count: int = Field(ge=1)
    sparse_selected_count: int = Field(ge=0)
    sparse_selection_rate: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    dense_minus_sparse_selection_rate: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        if self.dense_available_count != self.sparse_available_count:
            raise ValueError("Dense and sparse classes do not have equal availability")
        expected_dense = self.dense_selected_count / self.dense_available_count
        expected_sparse = self.sparse_selected_count / self.sparse_available_count
        if not (
            math.isclose(self.dense_selection_rate, expected_dense, abs_tol=1e-12)
            and math.isclose(self.sparse_selection_rate, expected_sparse, abs_tol=1e-12)
            and math.isclose(
                self.dense_minus_sparse_selection_rate,
                expected_dense - expected_sparse,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Dense-sparse selection effect does not match its counts")
        return self


class DevelopmentAblationAnalysisPlan(ContractModel):
    """Provenance and fixed interpretation for one development ablation analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_ablation_analysis_plan.v1"] = (
        "acquisition_study.development_ablation_analysis_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    source_ablation_manifest_hash: Sha256
    source_ablation_file_sha256: Sha256
    source_ablation_matrix_hash: Sha256
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    calibration_hash: Sha256
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    fixed_budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    metric: Literal["final_classification_error"] = "final_classification_error"
    pairwise_ablations: tuple[AblationId, ...] = Field(min_length=3, max_length=3)
    selection_ablation: Literal[AblationId.COMPARE_DENSE_AND_SPARSE_EQUAL_EXPECTED_RELIABILITY] = (
        _SELECTION_ABLATION
    )
    effect_direction: Literal["full_candidate_minus_ablated_negative_favours_full"] = (
        "full_candidate_minus_ablated_negative_favours_full"
    )
    aggregation: Literal["equal_mean_across_held_out_environments"] = (
        "equal_mean_across_held_out_environments"
    )
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if not math.isclose(self.fixed_budget_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("Ablation analysis plan differs from the fixed primary budget")
        if self.pairwise_ablations != tuple(item[0] for item in _PAIRWISE_ABLATIONS):
            raise ValueError("Ablation analysis plan differs from the frozen comparisons")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Ablation analysis plan hash does not match its content")
        return self


class DevelopmentAblationAnalysisReport(ContractModel):
    """Development-only component results without canonical claims."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_ablation_analysis.v1"] = (
        "acquisition_study.development_ablation_analysis.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    result_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    analysis_execution_plan_hash: Sha256
    source_ablation_manifest_hash: Sha256
    source_ablation_matrix_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    cell_ids: tuple[AblationPolicyCellId, ...] = Field(min_length=4, max_length=4)
    environment_summaries: tuple[AblationEnvironmentSummary, ...] = Field(
        min_length=28,
        max_length=28,
    )
    held_out_cell_summaries: tuple[HeldOutAblationCellSummary, ...] = Field(
        min_length=4,
        max_length=4,
    )
    environment_effects: tuple[EnvironmentAblationEffect, ...] = Field(
        min_length=21,
        max_length=21,
    )
    held_out_effects: tuple[HeldOutAblationEffect, ...] = Field(
        min_length=3,
        max_length=3,
    )
    held_out_factorial_interaction: HeldOutFactorialInteraction
    dense_sparse_selection_effects: tuple[DenseSparseSelectionEffect, ...] = Field(
        min_length=4,
        max_length=4,
    )
    effect_direction: Literal["full_candidate_minus_ablated_negative_favours_full"] = (
        "full_candidate_minus_ablated_negative_favours_full"
    )
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    bounded_source_reproduction_verified: Literal[True] = True
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.cell_ids != tuple(AblationPolicyCellId):
            raise ValueError("Ablation report differs from the frozen cells")
        expected_environment_keys = tuple(
            (environment_id, cell_id)
            for environment_id in EvaluationEnvironmentId
            for cell_id in self.cell_ids
        )
        actual_environment_keys = tuple(
            (row.metrics.environment_id, row.cell_id) for row in self.environment_summaries
        )
        if actual_environment_keys != expected_environment_keys:
            raise ValueError("Ablation environment summaries have incomplete coverage")
        if tuple(row.cell_id for row in self.held_out_cell_summaries) != self.cell_ids:
            raise ValueError("Ablation held-out summaries have incomplete coverage")
        expected_effect_keys = tuple(
            (environment_id, ablation_id)
            for environment_id in EvaluationEnvironmentId
            for ablation_id, _ in _PAIRWISE_ABLATIONS
        )
        actual_effect_keys = tuple(
            (row.environment_id, row.ablation_id) for row in self.environment_effects
        )
        if actual_effect_keys != expected_effect_keys:
            raise ValueError("Ablation environment effects have incomplete coverage")
        if tuple(row.ablation_id for row in self.held_out_effects) != tuple(
            item[0] for item in _PAIRWISE_ABLATIONS
        ):
            raise ValueError("Ablation held-out effects have incomplete coverage")
        if tuple(row.cell_id for row in self.dense_sparse_selection_effects) != self.cell_ids:
            raise ValueError("Dense-sparse effects have incomplete cell coverage")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Ablation analysis report hash does not match its content")
        return self


def run_development_ablation_analysis(
    *,
    source_manifest_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    analysis_code_revision: str,
) -> DevelopmentAblationAnalysisReport:
    """Verify, analyse, and immutably publish development ablation results."""

    source_manifest, matrix = load_verified_development_ablation_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    calibration = estimate_probe_reliability(specification)
    if calibration.calibration_hash != source_manifest.calibration_hash:
        raise AblationAnalysisError("Ablation source uses another calibration result")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise AblationAnalysisError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_ablation_analysis_plan.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "split": matrix.split,
        "source_run_id": source_manifest.run_id,
        "source_code_revision": source_manifest.code_revision,
        "source_ablation_manifest_hash": source_manifest.manifest_hash,
        "source_ablation_file_sha256": source_manifest.ablation_file_sha256,
        "source_ablation_matrix_hash": matrix.content_hash,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "calibration_hash": calibration.calibration_hash,
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "fixed_budget_fraction": analysis.primary.primary_budget_fraction,
        "metric": analysis.primary.metric,
        "pairwise_ablations": tuple(item[0] for item in _PAIRWISE_ABLATIONS),
        "selection_ablation": _SELECTION_ABLATION,
        "effect_direction": "full_candidate_minus_ablated_negative_favours_full",
        "aggregation": analysis.primary.aggregation,
        "analysis_status": analysis.secondary.status,
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentAblationAnalysisPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    report = analyse_development_ablation_matrix(
        matrix,
        analysis=analysis,
        source_manifest=source_manifest,
        plan=plan,
    )
    write_immutable_json(output_root / "development_ablation_analysis_plan.json", plan)
    write_immutable_json(output_root / "development_ablation_analysis_report.json", report)
    return report


def analyse_development_ablation_matrix(
    matrix: DevelopmentAblationMatrix,
    *,
    analysis: AcquisitionAnalysisSpecification,
    source_manifest: DevelopmentAblationManifest,
    plan: DevelopmentAblationAnalysisPlan,
) -> DevelopmentAblationAnalysisReport:
    """Summarise frozen development cells without choosing or tuning a method."""

    if matrix.split != "development" or matrix.analysis_plan_hash != analysis.plan_hash:
        raise AblationAnalysisError("Ablation matrix and analysis specification do not match")
    if (
        source_manifest.matrix_content_hash != matrix.content_hash
        or source_manifest.analysis_plan_hash != analysis.plan_hash
    ):
        raise AblationAnalysisError("Ablation matrix does not match its source manifest")
    if (
        plan.source_ablation_manifest_hash != source_manifest.manifest_hash
        or plan.source_ablation_matrix_hash != matrix.content_hash
        or plan.frozen_analysis_specification_hash != analysis.plan_hash
    ):
        raise AblationAnalysisError("Ablation analysis plan does not match its sources")

    grouped: dict[
        tuple[EvaluationEnvironmentId, AblationPolicyCellId],
        list[AblationEpisodeMetric],
    ] = {
        (environment_id, cell_id): []
        for environment_id in EvaluationEnvironmentId
        for cell_id in matrix.cell_ids
    }
    for row in matrix.rows:
        grouped[(row.metric.environment_id, row.cell_id)].append(row)
    environment_summaries = tuple(
        _summarise_environment_cell(
            grouped[(environment_id, cell_id)],
            environment_id=environment_id,
            cell_id=cell_id,
            candidates_per_episode=matrix.candidates_per_episode,
        )
        for environment_id in EvaluationEnvironmentId
        for cell_id in matrix.cell_ids
    )
    summary_by_key = {
        (row.metrics.environment_id, row.cell_id): row for row in environment_summaries
    }
    held_out_cell_summaries = tuple(
        _summarise_held_out_cell(
            summary_by_key,
            cell_id=cell_id,
            episodes_per_environment=matrix.episodes_per_environment,
        )
        for cell_id in matrix.cell_ids
    )
    held_out_by_cell = {row.cell_id: row for row in held_out_cell_summaries}
    row_by_key = {
        (row.metric.environment_id, row.metric.episode_index, row.cell_id): row
        for row in matrix.rows
    }
    environment_effects = tuple(
        _summarise_environment_effect(
            row_by_key,
            summary_by_key,
            environment_id=environment_id,
            ablation_id=ablation_id,
            ablated_cell_id=ablated_cell_id,
            episodes_per_environment=matrix.episodes_per_environment,
        )
        for environment_id in EvaluationEnvironmentId
        for ablation_id, ablated_cell_id in _PAIRWISE_ABLATIONS
    )
    effects_by_key = {(row.environment_id, row.ablation_id): row for row in environment_effects}
    held_out_effects = tuple(
        _summarise_held_out_effect(
            effects_by_key,
            held_out_by_cell,
            ablation_id=ablation_id,
            ablated_cell_id=ablated_cell_id,
            episodes_per_environment=matrix.episodes_per_environment,
        )
        for ablation_id, ablated_cell_id in _PAIRWISE_ABLATIONS
    )
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_ablation_analysis.v1",
        "study_id": matrix.study_id,
        "run_id": plan.run_id,
        "split": matrix.split,
        "result_status": matrix.result_scope,
        "analysis_execution_plan_hash": plan.plan_hash,
        "source_ablation_manifest_hash": source_manifest.manifest_hash,
        "source_ablation_matrix_hash": matrix.content_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "fixed_probe_budget": matrix.fixed_probe_budget,
        "cell_ids": matrix.cell_ids,
        "environment_summaries": environment_summaries,
        "held_out_cell_summaries": held_out_cell_summaries,
        "environment_effects": environment_effects,
        "held_out_effects": held_out_effects,
        "held_out_factorial_interaction": _summarise_factorial_interaction(held_out_by_cell),
        "dense_sparse_selection_effects": _summarise_dense_sparse_selection(
            environment_summaries,
            episodes_per_environment=matrix.episodes_per_environment,
        ),
        "effect_direction": plan.effect_direction,
        "analysis_status": plan.analysis_status,
        "bounded_source_reproduction_verified": (matrix.bounded_source_reproduction_verified),
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    return DevelopmentAblationAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def _summarise_environment_cell(
    rows: list[AblationEpisodeMetric],
    *,
    environment_id: EvaluationEnvironmentId,
    cell_id: AblationPolicyCellId,
    candidates_per_episode: int,
) -> AblationEnvironmentSummary:
    if not rows:
        raise AblationAnalysisError("Ablation cell has no source episodes")
    rows = sorted(rows, key=lambda row: row.metric.episode_index)
    if tuple(row.metric.episode_index for row in rows) != tuple(range(len(rows))):
        raise AblationAnalysisError("Ablation cell episode coverage is incomplete")
    if any(row.cell_id is not cell_id for row in rows):
        raise AblationAnalysisError("Ablation cell grouping is inconsistent")
    first = rows[0]
    return AblationEnvironmentSummary(
        cell_id=cell_id,
        reliability_summary=first.reliability_summary,
        update_rule=first.update_rule,
        metrics=summarise_environment_budget(
            [row.metric for row in rows],
            environment_id=environment_id,
            budget_fraction=0.5,
            policy_id=first.metric.policy_id,
            candidates_per_episode=candidates_per_episode,
        ),
        selection_by_probe_class=tuple(
            ProbeClassSelectionSummary(
                probe_class=probe_class,
                available_case_count=sum(
                    next(
                        item.available_case_count
                        for item in row.selection_by_probe_class
                        if item.probe_class is probe_class
                    )
                    for row in rows
                ),
                selected_case_count=(
                    selected_count := sum(
                        next(
                            item.selected_case_count
                            for item in row.selection_by_probe_class
                            if item.probe_class is probe_class
                        )
                        for row in rows
                    )
                ),
                selection_rate=(
                    selected_count
                    / sum(
                        next(
                            item.available_case_count
                            for item in row.selection_by_probe_class
                            if item.probe_class is probe_class
                        )
                        for row in rows
                    )
                ),
            )
            for probe_class in ProbeClassId
        ),
    )


def _summarise_held_out_cell(
    summaries: dict[
        tuple[EvaluationEnvironmentId, AblationPolicyCellId],
        AblationEnvironmentSummary,
    ],
    *,
    cell_id: AblationPolicyCellId,
    episodes_per_environment: int,
) -> HeldOutAblationCellSummary:
    rows = tuple(summaries[(environment_id, cell_id)] for environment_id in _HELD_OUT_ENVIRONMENTS)
    worst = max(rows, key=lambda row: row.metrics.mean_classification_error)
    return HeldOutAblationCellSummary(
        cell_id=cell_id,
        episodes_per_environment=episodes_per_environment,
        macro_mean_classification_error=fmean(
            row.metrics.mean_classification_error for row in rows
        ),
        macro_mean_brier_score=fmean(row.metrics.brier_score for row in rows),
        macro_mean_negative_log_likelihood=fmean(
            row.metrics.negative_log_likelihood for row in rows
        ),
        macro_mean_expected_calibration_error=fmean(
            row.metrics.expected_calibration_error for row in rows
        ),
        worst_environment_id=worst.metrics.environment_id,
        worst_environment_classification_error=worst.metrics.mean_classification_error,
        source_summaries_hash=canonical_sha256(rows),
    )


def _summarise_environment_effect(
    rows: dict[
        tuple[EvaluationEnvironmentId, int, AblationPolicyCellId],
        AblationEpisodeMetric,
    ],
    summaries: dict[
        tuple[EvaluationEnvironmentId, AblationPolicyCellId],
        AblationEnvironmentSummary,
    ],
    *,
    environment_id: EvaluationEnvironmentId,
    ablation_id: AblationId,
    ablated_cell_id: AblationPolicyCellId,
    episodes_per_environment: int,
) -> EnvironmentAblationEffect:
    effects = tuple(
        rows[
            (environment_id, episode_index, AblationPolicyCellId.FULL_CANDIDATE)
        ].metric.final_classification_error
        - rows[(environment_id, episode_index, ablated_cell_id)].metric.final_classification_error
        for episode_index in range(episodes_per_environment)
    )
    full = summaries[(environment_id, AblationPolicyCellId.FULL_CANDIDATE)]
    ablated = summaries[(environment_id, ablated_cell_id)]
    return EnvironmentAblationEffect(
        ablation_id=ablation_id,
        ablated_cell_id=ablated_cell_id,
        environment_id=environment_id,
        environment_role=full.metrics.environment_role,
        episode_count=episodes_per_environment,
        full_candidate_mean_error=full.metrics.mean_classification_error,
        ablated_mean_error=ablated.metrics.mean_classification_error,
        mean_paired_difference=fmean(effects),
        full_candidate_better_episode_count=sum(effect < 0.0 for effect in effects),
        tied_episode_count=sum(effect == 0.0 for effect in effects),
        ablated_better_episode_count=sum(effect > 0.0 for effect in effects),
        paired_differences_hash=canonical_sha256(effects),
    )


def _summarise_held_out_effect(
    effects: dict[tuple[EvaluationEnvironmentId, AblationId], EnvironmentAblationEffect],
    cell_summaries: dict[AblationPolicyCellId, HeldOutAblationCellSummary],
    *,
    ablation_id: AblationId,
    ablated_cell_id: AblationPolicyCellId,
    episodes_per_environment: int,
) -> HeldOutAblationEffect:
    rows = tuple(
        effects[(environment_id, ablation_id)] for environment_id in _HELD_OUT_ENVIRONMENTS
    )
    full = cell_summaries[AblationPolicyCellId.FULL_CANDIDATE]
    ablated = cell_summaries[ablated_cell_id]
    worst = max(rows, key=lambda row: row.mean_paired_difference)
    return HeldOutAblationEffect(
        ablation_id=ablation_id,
        ablated_cell_id=ablated_cell_id,
        episodes_per_environment=episodes_per_environment,
        paired_episode_count=len(rows) * episodes_per_environment,
        full_candidate_macro_mean_error=full.macro_mean_classification_error,
        ablated_macro_mean_error=ablated.macro_mean_classification_error,
        macro_mean_paired_difference=fmean(row.mean_paired_difference for row in rows),
        full_candidate_better_episode_count=sum(
            row.full_candidate_better_episode_count for row in rows
        ),
        tied_episode_count=sum(row.tied_episode_count for row in rows),
        ablated_better_episode_count=sum(row.ablated_better_episode_count for row in rows),
        worst_environment_id=worst.environment_id,
        worst_environment_mean_difference=worst.mean_paired_difference,
        environment_effects_hash=canonical_sha256(rows),
    )


def _summarise_factorial_interaction(
    summaries: dict[AblationPolicyCellId, HeldOutAblationCellSummary],
) -> HeldOutFactorialInteraction:
    errors = {
        cell_id: summary.macro_mean_classification_error for cell_id, summary in summaries.items()
    }
    lower_when_bounded = (
        errors[AblationPolicyCellId.FULL_CANDIDATE]
        - errors[AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED]
    )
    lower_when_unbounded = (
        errors[AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED]
        - errors[AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED]
    )
    bounded_with_lower = (
        errors[AblationPolicyCellId.FULL_CANDIDATE]
        - errors[AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED]
    )
    bounded_with_mean = (
        errors[AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED]
        - errors[AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED]
    )
    return HeldOutFactorialInteraction(
        lower_quantile_effect_when_bounded=lower_when_bounded,
        lower_quantile_effect_when_unbounded=lower_when_unbounded,
        bounded_update_effect_with_lower_quantile=bounded_with_lower,
        bounded_update_effect_with_posterior_mean=bounded_with_mean,
        interaction_difference=lower_when_bounded - lower_when_unbounded,
    )


def _summarise_dense_sparse_selection(
    summaries: tuple[AblationEnvironmentSummary, ...],
    *,
    episodes_per_environment: int,
) -> tuple[DenseSparseSelectionEffect, ...]:
    by_cell = {
        cell_id: tuple(row for row in summaries if row.cell_id is cell_id)
        for cell_id in AblationPolicyCellId
    }
    effects: list[DenseSparseSelectionEffect] = []
    for cell_id in AblationPolicyCellId:
        dense_rows = tuple(
            next(
                item
                for item in row.selection_by_probe_class
                if item.probe_class is ProbeClassId.HIGH_RELIABILITY_DENSE
            )
            for row in by_cell[cell_id]
        )
        sparse_rows = tuple(
            next(
                item
                for item in row.selection_by_probe_class
                if item.probe_class is ProbeClassId.HIGH_RELIABILITY_SPARSE
            )
            for row in by_cell[cell_id]
        )
        dense_available = sum(row.available_case_count for row in dense_rows)
        dense_selected = sum(row.selected_case_count for row in dense_rows)
        sparse_available = sum(row.available_case_count for row in sparse_rows)
        sparse_selected = sum(row.selected_case_count for row in sparse_rows)
        dense_rate = dense_selected / dense_available
        sparse_rate = sparse_selected / sparse_available
        effects.append(
            DenseSparseSelectionEffect(
                cell_id=cell_id,
                episode_count=len(EvaluationEnvironmentId) * episodes_per_environment,
                dense_available_count=dense_available,
                dense_selected_count=dense_selected,
                dense_selection_rate=dense_rate,
                sparse_available_count=sparse_available,
                sparse_selected_count=sparse_selected,
                sparse_selection_rate=sparse_rate,
                dense_minus_sparse_selection_rate=dense_rate - sparse_rate,
            )
        )
    return tuple(effects)
