"""Descriptive analysis of the development shuffled-reliability control."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.budget_curve_analysis import (
    EnvironmentBudgetSummary,
    HeldOutBudgetSummary,
    summarise_environment_budget,
)
from socratic_tutor.acquisition_study.budget_curve_runner import BudgetCurveEpisodeMetric
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.negative_control import (
    DevelopmentNegativeControlManifest,
    DevelopmentNegativeControlMatrix,
    NegativeControlEpisodeMetric,
    load_verified_development_negative_control,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EnvironmentRole,
    EvaluationEnvironmentId,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_CONTROL = PolicyId.SHUFFLED_RELIABILITY_BOUNDED
_HELD_OUT_ENVIRONMENTS = tuple(EvaluationEnvironmentId)[1:]


class NegativeControlAnalysisError(ValueError):
    """A development negative-control artifact cannot support the declared analysis."""


class NegativeControlEnvironmentEffect(ContractModel):
    """Candidate-minus-shuffled error for paired episodes in one environment."""

    environment_id: EvaluationEnvironmentId
    environment_role: EnvironmentRole
    episode_count: int = Field(ge=1)
    candidate_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    shuffled_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    mean_paired_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    candidate_better_episode_count: int = Field(ge=0)
    tied_episode_count: int = Field(ge=0)
    shuffled_better_episode_count: int = Field(ge=0)
    paired_differences_hash: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        expected_role = (
            EnvironmentRole.MATCHED_SANITY_CHECK
            if self.environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        )
        if self.environment_role is not expected_role:
            raise ValueError("Negative-control effect uses the wrong environment role")
        if self.episode_count != sum(
            (
                self.candidate_better_episode_count,
                self.tied_episode_count,
                self.shuffled_better_episode_count,
            )
        ):
            raise ValueError("Negative-control wins, ties, and losses do not reconcile")
        expected = (
            self.candidate_mean_classification_error - self.shuffled_mean_classification_error
        )
        if not math.isclose(self.mean_paired_difference, expected, abs_tol=1e-12):
            raise ValueError("Negative-control effect does not match its policy means")
        return self


class HeldOutNegativeControlEffect(ContractModel):
    """Equal-environment candidate-minus-shuffled effect over held-out settings."""

    environment_count: Literal[6] = 6
    episodes_per_environment: int = Field(ge=1)
    paired_episode_count: int = Field(ge=1)
    candidate_macro_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    shuffled_macro_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    macro_mean_paired_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    candidate_better_episode_count: int = Field(ge=0)
    tied_episode_count: int = Field(ge=0)
    shuffled_better_episode_count: int = Field(ge=0)
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_mean_difference: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    environment_effects_hash: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        if self.paired_episode_count != self.environment_count * self.episodes_per_environment:
            raise ValueError("Held-out negative-control episode count is inconsistent")
        if self.paired_episode_count != sum(
            (
                self.candidate_better_episode_count,
                self.tied_episode_count,
                self.shuffled_better_episode_count,
            )
        ):
            raise ValueError("Held-out negative-control counts do not reconcile")
        expected = (
            self.candidate_macro_mean_classification_error
            - self.shuffled_macro_mean_classification_error
        )
        if not math.isclose(self.macro_mean_paired_difference, expected, abs_tol=1e-12):
            raise ValueError("Held-out negative-control effect does not match its means")
        if self.worst_environment_id not in _HELD_OUT_ENVIRONMENTS:
            raise ValueError("Worst negative-control environment must be held out")
        return self


class DevelopmentNegativeControlAnalysisPlan(ContractModel):
    """Provenance and fixed interpretation for one development analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_negative_control_analysis_plan.v1"] = (
        "acquisition_study.development_negative_control_analysis_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    source_negative_control_manifest_hash: Sha256
    source_negative_control_file_sha256: Sha256
    source_negative_control_matrix_hash: Sha256
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    calibration_hash: Sha256
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    fixed_budget_fraction: float = Field(ge=0.5, le=0.5, allow_inf_nan=False)
    metric: Literal["final_classification_error"] = "final_classification_error"
    comparison: Literal["candidate_minus_shuffled_reliability"] = (
        "candidate_minus_shuffled_reliability"
    )
    effect_direction: Literal["negative_favours_correct_reliability_mapping"] = (
        "negative_favours_correct_reliability_mapping"
    )
    aggregation: Literal["equal_mean_across_held_out_environments"] = (
        "equal_mean_across_held_out_environments"
    )
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Negative-control analysis plan hash does not match its content")
        return self


class DevelopmentNegativeControlAnalysisReport(ContractModel):
    """Development-only negative-control results without canonical claims."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_negative_control_analysis.v1"] = (
        "acquisition_study.development_negative_control_analysis.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    result_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    analysis_execution_plan_hash: Sha256
    source_negative_control_manifest_hash: Sha256
    source_negative_control_matrix_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    environment_summaries: tuple[EnvironmentBudgetSummary, ...] = Field(
        min_length=14,
        max_length=14,
    )
    held_out_policy_summaries: tuple[HeldOutBudgetSummary, ...] = Field(
        min_length=2,
        max_length=2,
    )
    environment_effects: tuple[NegativeControlEnvironmentEffect, ...] = Field(
        min_length=7,
        max_length=7,
    )
    held_out_effect: HeldOutNegativeControlEffect
    effect_direction: Literal["negative_favours_correct_reliability_mapping"] = (
        "negative_favours_correct_reliability_mapping"
    )
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    candidate_source_reproduction_verified: Literal[True] = True
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        expected_summary_keys = tuple(
            (environment_id, policy_id)
            for environment_id in EvaluationEnvironmentId
            for policy_id in (_CANDIDATE, _CONTROL)
        )
        actual_summary_keys = tuple(
            (row.environment_id, row.policy_id) for row in self.environment_summaries
        )
        if actual_summary_keys != expected_summary_keys:
            raise ValueError("Negative-control environment summaries have incomplete coverage")
        if tuple(row.policy_id for row in self.held_out_policy_summaries) != (
            _CANDIDATE,
            _CONTROL,
        ):
            raise ValueError("Negative-control held-out summaries have incomplete coverage")
        if tuple(row.environment_id for row in self.environment_effects) != tuple(
            EvaluationEnvironmentId
        ):
            raise ValueError("Negative-control environment effects have incomplete coverage")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Negative-control analysis report hash does not match its content")
        return self


def run_development_negative_control_analysis(
    *,
    source_manifest_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    analysis_code_revision: str,
) -> DevelopmentNegativeControlAnalysisReport:
    """Verify, analyse, and immutably publish the development negative control."""

    source_manifest, matrix = load_verified_development_negative_control(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    calibration = estimate_probe_reliability(specification)
    if calibration.calibration_hash != source_manifest.calibration_hash:
        raise NegativeControlAnalysisError("Negative-control source uses another calibration")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise NegativeControlAnalysisError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_negative_control_analysis_plan.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "split": matrix.split,
        "source_run_id": source_manifest.run_id,
        "source_code_revision": source_manifest.code_revision,
        "source_negative_control_manifest_hash": source_manifest.manifest_hash,
        "source_negative_control_file_sha256": source_manifest.result_file_sha256,
        "source_negative_control_matrix_hash": matrix.content_hash,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "calibration_hash": calibration.calibration_hash,
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "fixed_budget_fraction": analysis.primary.primary_budget_fraction,
        "metric": analysis.primary.metric,
        "comparison": "candidate_minus_shuffled_reliability",
        "effect_direction": "negative_favours_correct_reliability_mapping",
        "aggregation": analysis.primary.aggregation,
        "analysis_status": analysis.secondary.status,
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentNegativeControlAnalysisPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    report = analyse_development_negative_control(
        matrix,
        analysis=analysis,
        source_manifest=source_manifest,
        plan=plan,
    )
    write_immutable_json(output_root / "development_negative_control_analysis_plan.json", plan)
    write_immutable_json(output_root / "development_negative_control_analysis_report.json", report)
    return report


def analyse_development_negative_control(
    matrix: DevelopmentNegativeControlMatrix,
    *,
    analysis: AcquisitionAnalysisSpecification,
    source_manifest: DevelopmentNegativeControlManifest,
    plan: DevelopmentNegativeControlAnalysisPlan,
) -> DevelopmentNegativeControlAnalysisReport:
    """Summarise paired development results without tuning or inferential claims."""

    if matrix.split != "development" or matrix.analysis_plan_hash != analysis.plan_hash:
        raise NegativeControlAnalysisError("Negative-control matrix and analysis plan differ")
    if (
        source_manifest.matrix_content_hash != matrix.content_hash
        or source_manifest.analysis_plan_hash != analysis.plan_hash
    ):
        raise NegativeControlAnalysisError("Negative-control matrix does not match its manifest")
    if (
        plan.source_negative_control_manifest_hash != source_manifest.manifest_hash
        or plan.source_negative_control_matrix_hash != matrix.content_hash
        or plan.frozen_analysis_specification_hash != analysis.plan_hash
    ):
        raise NegativeControlAnalysisError(
            "Negative-control analysis plan does not match its sources"
        )

    grouped: dict[
        tuple[EvaluationEnvironmentId, PolicyId],
        list[BudgetCurveEpisodeMetric],
    ] = {
        (environment_id, policy_id): []
        for environment_id in EvaluationEnvironmentId
        for policy_id in (_CANDIDATE, _CONTROL)
    }
    rows_by_environment: dict[EvaluationEnvironmentId, list[NegativeControlEpisodeMetric]] = {
        environment_id: [] for environment_id in EvaluationEnvironmentId
    }
    for row in matrix.rows:
        grouped[(row.environment_id, _CANDIDATE)].append(row.candidate)
        grouped[(row.environment_id, _CONTROL)].append(row.shuffled_control)
        rows_by_environment[row.environment_id].append(row)

    summaries = tuple(
        summarise_environment_budget(
            grouped[(environment_id, policy_id)],
            environment_id=environment_id,
            budget_fraction=matrix.fixed_budget_fraction,
            policy_id=policy_id,
            candidates_per_episode=matrix.candidates_per_episode,
        )
        for environment_id in EvaluationEnvironmentId
        for policy_id in (_CANDIDATE, _CONTROL)
    )
    summaries_by_key = {(row.environment_id, row.policy_id): row for row in summaries}
    held_out_summaries = tuple(
        _summarise_held_out_policy(
            summaries_by_key,
            policy_id=policy_id,
            episodes_per_environment=matrix.episodes_per_environment,
            budget_fraction=matrix.fixed_budget_fraction,
        )
        for policy_id in (_CANDIDATE, _CONTROL)
    )
    held_out_by_policy = {row.policy_id: row for row in held_out_summaries}
    environment_effects = tuple(
        _summarise_environment_effect(
            rows_by_environment[environment_id],
            summaries_by_key,
            environment_id=environment_id,
        )
        for environment_id in EvaluationEnvironmentId
    )
    held_out_effect = _summarise_held_out_effect(
        environment_effects,
        held_out_by_policy,
        episodes_per_environment=matrix.episodes_per_environment,
    )
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_negative_control_analysis.v1",
        "study_id": matrix.study_id,
        "run_id": plan.run_id,
        "split": matrix.split,
        "result_status": matrix.result_scope,
        "analysis_execution_plan_hash": plan.plan_hash,
        "source_negative_control_manifest_hash": source_manifest.manifest_hash,
        "source_negative_control_matrix_hash": matrix.content_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "fixed_probe_budget": matrix.fixed_probe_budget,
        "environment_summaries": summaries,
        "held_out_policy_summaries": held_out_summaries,
        "environment_effects": environment_effects,
        "held_out_effect": held_out_effect,
        "effect_direction": plan.effect_direction,
        "analysis_status": plan.analysis_status,
        "candidate_source_reproduction_verified": (matrix.candidate_source_reproduction_verified),
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    return DevelopmentNegativeControlAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def _summarise_held_out_policy(
    summaries: dict[tuple[EvaluationEnvironmentId, PolicyId], EnvironmentBudgetSummary],
    *,
    policy_id: PolicyId,
    episodes_per_environment: int,
    budget_fraction: float,
) -> HeldOutBudgetSummary:
    rows = tuple(
        summaries[(environment_id, policy_id)] for environment_id in _HELD_OUT_ENVIRONMENTS
    )
    worst = max(rows, key=lambda row: row.mean_classification_error)
    return HeldOutBudgetSummary(
        budget_fraction=budget_fraction,
        policy_id=policy_id,
        episodes_per_environment=episodes_per_environment,
        macro_mean_classification_error=fmean(row.mean_classification_error for row in rows),
        macro_mean_brier_score=fmean(row.brier_score for row in rows),
        macro_mean_negative_log_likelihood=fmean(row.negative_log_likelihood for row in rows),
        macro_mean_expected_calibration_error=fmean(row.expected_calibration_error for row in rows),
        worst_environment_id=worst.environment_id,
        worst_environment_classification_error=worst.mean_classification_error,
        environment_summaries_hash=canonical_sha256(rows),
    )


def _summarise_environment_effect(
    rows: list[NegativeControlEpisodeMetric],
    summaries: dict[tuple[EvaluationEnvironmentId, PolicyId], EnvironmentBudgetSummary],
    *,
    environment_id: EvaluationEnvironmentId,
) -> NegativeControlEnvironmentEffect:
    rows = sorted(rows, key=lambda row: row.episode_index)
    if tuple(row.episode_index for row in rows) != tuple(range(len(rows))):
        raise NegativeControlAnalysisError("Negative-control episode coverage is incomplete")
    differences = tuple(
        row.candidate.final_classification_error - row.shuffled_control.final_classification_error
        for row in rows
    )
    candidate = summaries[(environment_id, _CANDIDATE)]
    shuffled = summaries[(environment_id, _CONTROL)]
    return NegativeControlEnvironmentEffect(
        environment_id=environment_id,
        environment_role=candidate.environment_role,
        episode_count=len(rows),
        candidate_mean_classification_error=candidate.mean_classification_error,
        shuffled_mean_classification_error=shuffled.mean_classification_error,
        mean_paired_difference=fmean(differences),
        candidate_better_episode_count=sum(value < 0.0 for value in differences),
        tied_episode_count=sum(math.isclose(value, 0.0, abs_tol=1e-12) for value in differences),
        shuffled_better_episode_count=sum(value > 0.0 for value in differences),
        paired_differences_hash=canonical_sha256(differences),
    )


def _summarise_held_out_effect(
    effects: tuple[NegativeControlEnvironmentEffect, ...],
    summaries: dict[PolicyId, HeldOutBudgetSummary],
    *,
    episodes_per_environment: int,
) -> HeldOutNegativeControlEffect:
    held_out = tuple(row for row in effects if row.environment_id in _HELD_OUT_ENVIRONMENTS)
    worst = max(held_out, key=lambda row: row.mean_paired_difference)
    candidate = summaries[_CANDIDATE]
    shuffled = summaries[_CONTROL]
    return HeldOutNegativeControlEffect(
        episodes_per_environment=episodes_per_environment,
        paired_episode_count=len(held_out) * episodes_per_environment,
        candidate_macro_mean_classification_error=(candidate.macro_mean_classification_error),
        shuffled_macro_mean_classification_error=(shuffled.macro_mean_classification_error),
        macro_mean_paired_difference=fmean(row.mean_paired_difference for row in held_out),
        candidate_better_episode_count=sum(row.candidate_better_episode_count for row in held_out),
        tied_episode_count=sum(row.tied_episode_count for row in held_out),
        shuffled_better_episode_count=sum(row.shuffled_better_episode_count for row in held_out),
        worst_environment_id=worst.environment_id,
        worst_environment_mean_difference=worst.mean_paired_difference,
        environment_effects_hash=canonical_sha256(held_out),
    )
