"""Descriptive paired analysis of the acquisition-study budget curve."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.budget_curve import BUDGET_CURVE_POLICY_ORDER
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    DevelopmentBudgetCurveManifest,
    DevelopmentBudgetCurveMatrix,
    load_verified_development_budget_curve,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EnvironmentRole,
    EvaluationEnvironmentId,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_HELD_OUT_ENVIRONMENTS = tuple(EvaluationEnvironmentId)[1:]
_BUDGET_REFERENCES = BUDGET_CURVE_POLICY_ORDER[1:]


class BudgetCurveAnalysisError(ValueError):
    """Budget rows cannot support the frozen descriptive analysis."""


class DevelopmentBudgetCurveAnalysisPlan(ContractModel):
    """Provenance for one development budget-curve analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_budget_curve_analysis_plan.v1"] = (
        "acquisition_study.development_budget_curve_analysis_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    source_budget_curve_manifest_hash: Sha256
    source_budget_curve_file_sha256: Sha256
    source_budget_curve_matrix_hash: Sha256
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    budget_fractions: tuple[float, ...] = Field(min_length=5, max_length=5)
    metrics: tuple[str, ...] = Field(min_length=4, max_length=4)
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.budget_fractions != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Budget analysis plan differs from the frozen fractions")
        if self.metrics != (
            "classification_error",
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
        ):
            raise ValueError("Budget analysis plan differs from the frozen metrics")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Budget analysis plan hash does not match its content")
        return self


class EnvironmentBudgetSummary(ContractModel):
    """Metrics pooled over episodes for one environment, budget, and policy."""

    environment_id: EvaluationEnvironmentId
    environment_role: EnvironmentRole
    budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    policy_id: PolicyId
    episode_count: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    selected_probes_per_episode: int = Field(ge=0)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    observed_probe_pass_count: int = Field(ge=0)
    observed_probe_fail_count: int = Field(ge=0)
    missing_probe_result_count: int = Field(ge=0)
    mean_classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    brier_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    negative_log_likelihood: float = Field(ge=0.0, allow_inf_nan=False)
    expected_calibration_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    source_rows_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        expected_role = (
            EnvironmentRole.MATCHED_SANITY_CHECK
            if self.environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        )
        if self.environment_role is not expected_role:
            raise ValueError("Budget summary uses the wrong environment role")
        if self.case_prediction_count != self.episode_count * self.candidates_per_episode:
            raise ValueError("Budget summary case count is inconsistent")
        if self.selected_probe_count != self.episode_count * self.selected_probes_per_episode:
            raise ValueError("Budget summary selected-probe count is inconsistent")
        expected_selected = self.budget_fraction * self.candidates_per_episode
        if not math.isclose(expected_selected, self.selected_probes_per_episode, abs_tol=1e-12):
            raise ValueError("Budget summary fraction does not match its probe count")
        if self.selected_probe_count != sum(
            (
                self.observed_probe_pass_count,
                self.observed_probe_fail_count,
                self.missing_probe_result_count,
            )
        ):
            raise ValueError("Budget summary probe outcomes do not reconcile")
        return self


class HeldOutBudgetSummary(ContractModel):
    """Equal-environment average over the six held-out mismatch settings."""

    budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    policy_id: PolicyId
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
    environment_summaries_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.worst_environment_id not in _HELD_OUT_ENVIRONMENTS:
            raise ValueError("Worst environment must be a held-out mismatch setting")
        return self


class HeldOutBudgetPairedDifference(ContractModel):
    """Candidate-minus-reference error over paired held-out episodes."""

    budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    reference_policy_id: PolicyId
    reference_role: Literal["matched_budget_comparator", "oracle_reference"]
    environment_count: Literal[6] = 6
    episodes_per_environment: int = Field(ge=1)
    paired_episode_count: int = Field(ge=1)
    candidate_macro_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    reference_macro_mean_classification_error: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    macro_mean_paired_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    candidate_better_episode_count: int = Field(ge=0)
    tied_episode_count: int = Field(ge=0)
    reference_better_episode_count: int = Field(ge=0)
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_mean_difference: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    paired_differences_hash: Sha256

    @model_validator(mode="after")
    def validate_difference(self) -> Self:
        if self.reference_policy_id not in _BUDGET_REFERENCES:
            raise ValueError("Budget difference uses an undeclared reference policy")
        expected_role = (
            "oracle_reference"
            if self.reference_policy_id is PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED
            else "matched_budget_comparator"
        )
        if self.reference_role != expected_role:
            raise ValueError("Budget difference mislabels its reference policy")
        if self.paired_episode_count != self.environment_count * self.episodes_per_environment:
            raise ValueError("Budget difference episode count is inconsistent")
        if self.paired_episode_count != sum(
            (
                self.candidate_better_episode_count,
                self.tied_episode_count,
                self.reference_better_episode_count,
            )
        ):
            raise ValueError("Budget difference win, tie, and loss counts do not reconcile")
        expected = (
            self.candidate_macro_mean_classification_error
            - self.reference_macro_mean_classification_error
        )
        if not math.isclose(self.macro_mean_paired_difference, expected, abs_tol=1e-12):
            raise ValueError("Budget difference does not match its policy means")
        if self.worst_environment_id not in _HELD_OUT_ENVIRONMENTS:
            raise ValueError("Worst difference must come from a held-out environment")
        return self


class DevelopmentBudgetCurveAnalysisReport(ContractModel):
    """Development-only descriptive results across the full probe-budget curve."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_budget_curve_analysis.v1"] = (
        "acquisition_study.development_budget_curve_analysis.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["development"] = "development"
    result_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    analysis_execution_plan_hash: Sha256
    source_budget_curve_manifest_hash: Sha256
    source_budget_curve_matrix_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    budget_fractions: tuple[float, ...] = Field(min_length=5, max_length=5)
    policy_ids: tuple[PolicyId, ...] = Field(min_length=5, max_length=5)
    environment_summaries: tuple[EnvironmentBudgetSummary, ...] = Field(
        min_length=175,
        max_length=175,
    )
    held_out_macro_summaries: tuple[HeldOutBudgetSummary, ...] = Field(
        min_length=25,
        max_length=25,
    )
    held_out_paired_differences: tuple[HeldOutBudgetPairedDifference, ...] = Field(
        min_length=20,
        max_length=20,
    )
    analysis_status: Literal["descriptive_not_confirmatory"] = "descriptive_not_confirmatory"
    endpoint_agreement_verified: Literal[True] = True
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    deployed_cost_saving_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.budget_fractions != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Budget analysis fractions differ from the frozen plan")
        if self.policy_ids != BUDGET_CURVE_POLICY_ORDER:
            raise ValueError("Budget analysis policies differ from the frozen plan")
        expected_environment_keys = tuple(
            (environment_id, budget_fraction, policy_id)
            for environment_id in EvaluationEnvironmentId
            for budget_fraction in self.budget_fractions
            for policy_id in self.policy_ids
        )
        actual_environment_keys = tuple(
            (row.environment_id, row.budget_fraction, row.policy_id)
            for row in self.environment_summaries
        )
        if actual_environment_keys != expected_environment_keys:
            raise ValueError("Budget environment summaries have incomplete coverage")
        expected_macro_keys = tuple(
            (budget_fraction, policy_id)
            for budget_fraction in self.budget_fractions
            for policy_id in self.policy_ids
        )
        actual_macro_keys = tuple(
            (row.budget_fraction, row.policy_id) for row in self.held_out_macro_summaries
        )
        if actual_macro_keys != expected_macro_keys:
            raise ValueError("Budget macro summaries have incomplete coverage")
        expected_difference_keys = tuple(
            (budget_fraction, reference_policy_id)
            for budget_fraction in self.budget_fractions
            for reference_policy_id in _BUDGET_REFERENCES
        )
        actual_difference_keys = tuple(
            (row.budget_fraction, row.reference_policy_id)
            for row in self.held_out_paired_differences
        )
        if actual_difference_keys != expected_difference_keys:
            raise ValueError("Budget paired differences have incomplete coverage")
        endpoint_rows = tuple(
            row for row in self.held_out_paired_differences if row.budget_fraction in (0.0, 1.0)
        )
        if any(
            not math.isclose(row.macro_mean_paired_difference, 0.0, abs_tol=1e-12)
            or row.candidate_better_episode_count != 0
            or row.reference_better_episode_count != 0
            for row in endpoint_rows
        ):
            raise ValueError("Budget endpoint policies do not agree")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Budget analysis report hash does not match its content")
        return self


def analyse_development_budget_curve(
    matrix: DevelopmentBudgetCurveMatrix,
    *,
    analysis: AcquisitionAnalysisSpecification,
    source_manifest: DevelopmentBudgetCurveManifest,
    plan: DevelopmentBudgetCurveAnalysisPlan,
) -> DevelopmentBudgetCurveAnalysisReport:
    """Summarise the frozen development curve without inferential claims."""

    if matrix.split != "development" or matrix.analysis_plan_hash != analysis.plan_hash:
        raise BudgetCurveAnalysisError("Budget matrix and analysis plan do not match")
    if (
        source_manifest.matrix_content_hash != matrix.content_hash
        or source_manifest.analysis_plan_hash != analysis.plan_hash
    ):
        raise BudgetCurveAnalysisError("Budget matrix does not match its source manifest")
    if (
        plan.source_budget_curve_manifest_hash != source_manifest.manifest_hash
        or plan.source_budget_curve_matrix_hash != matrix.content_hash
        or plan.frozen_analysis_specification_hash != analysis.plan_hash
    ):
        raise BudgetCurveAnalysisError("Budget analysis plan does not match its sources")
    grouped: dict[
        tuple[EvaluationEnvironmentId, float, PolicyId],
        list[BudgetCurveEpisodeMetric],
    ] = {
        (environment_id, fraction, policy_id): []
        for environment_id in EvaluationEnvironmentId
        for fraction in matrix.budget_fractions
        for policy_id in matrix.policy_ids
    }
    for row in matrix.rows:
        grouped[(row.environment_id, row.budget_fraction, row.policy_id)].append(row)
    summaries = tuple(
        summarise_environment_budget(
            grouped[(environment_id, fraction, policy_id)],
            environment_id=environment_id,
            budget_fraction=fraction,
            policy_id=policy_id,
            candidates_per_episode=matrix.candidates_per_episode,
        )
        for environment_id in EvaluationEnvironmentId
        for fraction in matrix.budget_fractions
        for policy_id in matrix.policy_ids
    )
    summary_by_key = {
        (row.environment_id, row.budget_fraction, row.policy_id): row for row in summaries
    }
    macro_summaries = tuple(
        _summarise_held_out_budget(
            summary_by_key,
            budget_fraction=fraction,
            policy_id=policy_id,
            episodes_per_environment=matrix.episodes_per_environment,
        )
        for fraction in matrix.budget_fractions
        for policy_id in matrix.policy_ids
    )
    macro_by_key = {(row.budget_fraction, row.policy_id): row for row in macro_summaries}
    row_by_key = {
        (row.environment_id, row.episode_index, row.budget_fraction, row.policy_id): row
        for row in matrix.rows
    }
    paired_differences = tuple(
        _summarise_paired_difference(
            row_by_key,
            summary_by_key,
            macro_by_key,
            budget_fraction=fraction,
            reference_policy_id=reference_policy_id,
            episodes_per_environment=matrix.episodes_per_environment,
        )
        for fraction in matrix.budget_fractions
        for reference_policy_id in _BUDGET_REFERENCES
    )
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_budget_curve_analysis.v1",
        "study_id": matrix.study_id,
        "split": matrix.split,
        "result_status": matrix.result_scope,
        "run_id": plan.run_id,
        "analysis_execution_plan_hash": plan.plan_hash,
        "source_budget_curve_manifest_hash": source_manifest.manifest_hash,
        "source_budget_curve_matrix_hash": matrix.content_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "budget_fractions": matrix.budget_fractions,
        "policy_ids": matrix.policy_ids,
        "environment_summaries": summaries,
        "held_out_macro_summaries": macro_summaries,
        "held_out_paired_differences": paired_differences,
        "analysis_status": analysis.secondary.status,
        "endpoint_agreement_verified": True,
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "deployed_cost_saving_claim_supported": False,
    }
    return DevelopmentBudgetCurveAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def run_development_budget_curve_analysis(
    *,
    source_manifest_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    analysis_code_revision: str,
) -> DevelopmentBudgetCurveAnalysisReport:
    """Verify, analyse, and immutably publish the development budget curve."""

    source_manifest, matrix = load_verified_development_budget_curve(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise BudgetCurveAnalysisError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_budget_curve_analysis_plan.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "split": matrix.split,
        "source_run_id": source_manifest.run_id,
        "source_code_revision": source_manifest.code_revision,
        "source_budget_curve_manifest_hash": source_manifest.manifest_hash,
        "source_budget_curve_file_sha256": source_manifest.budget_curve_file_sha256,
        "source_budget_curve_matrix_hash": matrix.content_hash,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "budget_fractions": analysis.secondary.budget_fractions,
        "metrics": analysis.secondary.metrics,
        "analysis_status": analysis.secondary.status,
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentBudgetCurveAnalysisPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    report = analyse_development_budget_curve(
        matrix,
        analysis=analysis,
        source_manifest=source_manifest,
        plan=plan,
    )
    write_immutable_json(output_root / "development_budget_curve_analysis_plan.json", plan)
    write_immutable_json(output_root / "development_budget_curve_analysis_report.json", report)
    return report


def summarise_environment_budget(
    rows: list[BudgetCurveEpisodeMetric],
    *,
    environment_id: EvaluationEnvironmentId,
    budget_fraction: float,
    policy_id: PolicyId,
    candidates_per_episode: int,
) -> EnvironmentBudgetSummary:
    """Pool compact episode metrics without discarding calibration counts."""

    if not rows:
        raise BudgetCurveAnalysisError("Budget summary has no source episodes")
    rows = sorted(rows, key=lambda row: row.episode_index)
    if tuple(row.episode_index for row in rows) != tuple(range(len(rows))):
        raise BudgetCurveAnalysisError("Budget summary episode coverage is incomplete")
    selected_counts = {row.selected_probe_count for row in rows}
    if len(selected_counts) != 1:
        raise BudgetCurveAnalysisError("Budget summary mixes different probe counts")
    bin_counts = tuple(
        sum(row.calibration_bin_counts[index] for row in rows) for index in range(10)
    )
    probability_sums = tuple(
        math.fsum(row.calibration_probability_sums[index] for row in rows) for index in range(10)
    )
    positive_counts = tuple(
        sum(row.calibration_positive_counts[index] for row in rows) for index in range(10)
    )
    case_count = len(rows) * candidates_per_episode
    return EnvironmentBudgetSummary(
        environment_id=environment_id,
        environment_role=(
            EnvironmentRole.MATCHED_SANITY_CHECK
            if environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        ),
        budget_fraction=budget_fraction,
        policy_id=policy_id,
        episode_count=len(rows),
        candidates_per_episode=candidates_per_episode,
        selected_probes_per_episode=selected_counts.pop(),
        case_prediction_count=case_count,
        selected_probe_count=sum(row.selected_probe_count for row in rows),
        observed_probe_pass_count=sum(row.observed_probe_pass_count for row in rows),
        observed_probe_fail_count=sum(row.observed_probe_fail_count for row in rows),
        missing_probe_result_count=sum(row.missing_probe_result_count for row in rows),
        mean_classification_error=(
            sum(row.classification_error_count for row in rows) / case_count
        ),
        brier_score=math.fsum(row.squared_error_sum for row in rows) / case_count,
        negative_log_likelihood=(
            math.fsum(row.negative_log_likelihood_sum for row in rows) / case_count
        ),
        expected_calibration_error=_expected_calibration_error(
            bin_counts,
            probability_sums,
            positive_counts,
        ),
        source_rows_hash=canonical_sha256(rows),
    )


def _summarise_held_out_budget(
    summaries: dict[
        tuple[EvaluationEnvironmentId, float, PolicyId],
        EnvironmentBudgetSummary,
    ],
    *,
    budget_fraction: float,
    policy_id: PolicyId,
    episodes_per_environment: int,
) -> HeldOutBudgetSummary:
    rows = tuple(
        summaries[(environment_id, budget_fraction, policy_id)]
        for environment_id in _HELD_OUT_ENVIRONMENTS
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


def _summarise_paired_difference(
    rows: dict[
        tuple[EvaluationEnvironmentId, int, float, PolicyId],
        BudgetCurveEpisodeMetric,
    ],
    environment_summaries: dict[
        tuple[EvaluationEnvironmentId, float, PolicyId],
        EnvironmentBudgetSummary,
    ],
    macro_summaries: dict[tuple[float, PolicyId], HeldOutBudgetSummary],
    *,
    budget_fraction: float,
    reference_policy_id: PolicyId,
    episodes_per_environment: int,
) -> HeldOutBudgetPairedDifference:
    effects_by_environment: dict[EvaluationEnvironmentId, tuple[float, ...]] = {}
    for environment_id in _HELD_OUT_ENVIRONMENTS:
        effects_by_environment[environment_id] = tuple(
            rows[
                (environment_id, episode_index, budget_fraction, _CANDIDATE)
            ].final_classification_error
            - rows[
                (environment_id, episode_index, budget_fraction, reference_policy_id)
            ].final_classification_error
            for episode_index in range(episodes_per_environment)
        )
    all_effects = tuple(
        effect
        for environment_id in _HELD_OUT_ENVIRONMENTS
        for effect in effects_by_environment[environment_id]
    )
    environment_mean_effects = {
        environment_id: fmean(effects) for environment_id, effects in effects_by_environment.items()
    }
    worst_environment_id = max(
        _HELD_OUT_ENVIRONMENTS,
        key=environment_mean_effects.__getitem__,
    )
    candidate_macro = macro_summaries[(budget_fraction, _CANDIDATE)]
    reference_macro = macro_summaries[(budget_fraction, reference_policy_id)]
    expected_macro_effect = fmean(
        environment_summaries[
            (environment_id, budget_fraction, _CANDIDATE)
        ].mean_classification_error
        - environment_summaries[
            (environment_id, budget_fraction, reference_policy_id)
        ].mean_classification_error
        for environment_id in _HELD_OUT_ENVIRONMENTS
    )
    return HeldOutBudgetPairedDifference(
        budget_fraction=budget_fraction,
        reference_policy_id=reference_policy_id,
        reference_role=(
            "oracle_reference"
            if reference_policy_id is PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED
            else "matched_budget_comparator"
        ),
        episodes_per_environment=episodes_per_environment,
        paired_episode_count=len(all_effects),
        candidate_macro_mean_classification_error=(candidate_macro.macro_mean_classification_error),
        reference_macro_mean_classification_error=(reference_macro.macro_mean_classification_error),
        macro_mean_paired_difference=expected_macro_effect,
        candidate_better_episode_count=sum(effect < 0.0 for effect in all_effects),
        tied_episode_count=sum(effect == 0.0 for effect in all_effects),
        reference_better_episode_count=sum(effect > 0.0 for effect in all_effects),
        worst_environment_id=worst_environment_id,
        worst_environment_mean_difference=environment_mean_effects[worst_environment_id],
        paired_differences_hash=canonical_sha256(
            tuple(
                (environment_id, effects_by_environment[environment_id])
                for environment_id in _HELD_OUT_ENVIRONMENTS
            )
        ),
    )


def _expected_calibration_error(
    counts: tuple[int, ...],
    probability_sums: tuple[float, ...],
    positive_counts: tuple[int, ...],
) -> float:
    total = sum(counts)
    if total < 1:
        raise BudgetCurveAnalysisError("Calibration summary has no predictions")
    return math.fsum(
        count / total * abs(probability_sum / count - positive_count / count)
        for count, probability_sum, positive_count in zip(
            counts,
            probability_sums,
            positive_counts,
            strict=True,
        )
        if count
    )
