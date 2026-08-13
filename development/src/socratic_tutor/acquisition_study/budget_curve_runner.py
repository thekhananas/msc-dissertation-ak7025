"""Verified development matrix for the frozen acquisition budget curve."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.budget_curve import (
    BUDGET_CURVE_POLICY_ORDER,
    BudgetPolicyResult,
    run_episode_budget_curve,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.evaluation import PolicyEpisodeResult
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    load_verified_development_policy_matrix,
)
from socratic_tutor.acquisition_study.simulation import SimulatedProbeOutcome
from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel


class DevelopmentBudgetCurveError(ValueError):
    """A budget-curve source or result violates the frozen development design."""


class BudgetCurveEpisodeMetric(ContractModel):
    """Compact sufficient statistics for one policy, budget, and episode."""

    environment_id: EvaluationEnvironmentId
    episode_index: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    policy_id: PolicyId
    candidate_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    observed_probe_pass_count: int = Field(ge=0)
    observed_probe_fail_count: int = Field(ge=0)
    missing_probe_result_count: int = Field(ge=0)
    classification_error_count: int = Field(ge=0)
    final_classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    squared_error_sum: float = Field(ge=0.0, allow_inf_nan=False)
    brier_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    negative_log_likelihood_sum: float = Field(ge=0.0, allow_inf_nan=False)
    negative_log_likelihood: float = Field(ge=0.0, allow_inf_nan=False)
    ece_bin_count: Literal[10] = 10
    calibration_bin_counts: tuple[int, ...] = Field(min_length=10, max_length=10)
    calibration_probability_sums: tuple[float, ...] = Field(min_length=10, max_length=10)
    calibration_positive_counts: tuple[int, ...] = Field(min_length=10, max_length=10)
    decision_hash: Sha256
    case_results_hash: Sha256

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        if self.episode_id != f"episode-{self.episode_index:06d}":
            raise ValueError("Budget metric episode identifier does not match its index")
        expected_selected = self.budget_fraction * self.candidate_count
        if not math.isclose(expected_selected, self.selected_probe_count, abs_tol=1e-12):
            raise ValueError("Budget metric fraction does not match its selected count")
        observed_count = (
            self.observed_probe_pass_count
            + self.observed_probe_fail_count
            + self.missing_probe_result_count
        )
        if observed_count != self.selected_probe_count:
            raise ValueError("Observed probe outcomes do not reconcile with the budget")
        if self.classification_error_count > self.candidate_count:
            raise ValueError("Classification-error count exceeds the case count")
        if not math.isclose(
            self.final_classification_error,
            self.classification_error_count / self.candidate_count,
            abs_tol=1e-12,
        ):
            raise ValueError("Classification-error mean does not match its count")
        if not math.isclose(
            self.brier_score,
            self.squared_error_sum / self.candidate_count,
            abs_tol=1e-12,
        ):
            raise ValueError("Brier score does not match its sufficient statistic")
        if not math.isclose(
            self.negative_log_likelihood,
            self.negative_log_likelihood_sum / self.candidate_count,
            abs_tol=1e-12,
        ):
            raise ValueError("Log loss does not match its sufficient statistic")
        if sum(self.calibration_bin_counts) != self.candidate_count:
            raise ValueError("Calibration bins do not cover every case")
        if any(
            positive_count > count
            for positive_count, count in zip(
                self.calibration_positive_counts,
                self.calibration_bin_counts,
                strict=True,
            )
        ):
            raise ValueError("Calibration positives exceed their bin counts")
        if any(
            probability_sum < 0.0 or probability_sum > count or not math.isfinite(probability_sum)
            for probability_sum, count in zip(
                self.calibration_probability_sums,
                self.calibration_bin_counts,
                strict=True,
            )
        ):
            raise ValueError("Calibration probability sums are invalid")
        return self


class DevelopmentBudgetCurveMatrix(ContractModel):
    """Compact paired budget curves over every development episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_budget_curve_matrix.v1"] = (
        "acquisition_study.development_budget_curve_matrix.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    source_manifest_hash: Sha256
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    environment_count: Literal[7] = 7
    budget_fractions: tuple[float, ...] = Field(min_length=5, max_length=5)
    policy_ids: tuple[PolicyId, ...] = Field(min_length=5, max_length=5)
    row_count: int = Field(ge=1)
    rows: tuple[BudgetCurveEpisodeMetric, ...]
    primary_budget_reproduction_verified: Literal[True] = True
    endpoint_agreement_verified: Literal[True] = True
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.environment_count != len(EvaluationEnvironmentId):
            raise ValueError("Budget matrix does not contain every environment")
        if self.budget_fractions != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Budget matrix differs from the frozen fractions")
        if self.policy_ids != BUDGET_CURVE_POLICY_ORDER:
            raise ValueError("Budget matrix differs from the frozen policy order")
        expected_count = (
            self.environment_count
            * self.episodes_per_environment
            * len(self.budget_fractions)
            * len(self.policy_ids)
        )
        if self.row_count != expected_count or len(self.rows) != expected_count:
            raise ValueError("Budget matrix row count is inconsistent")
        if any(row.candidate_count != self.candidates_per_episode for row in self.rows):
            raise ValueError("Budget matrix rows use another candidate count")
        expected_keys = tuple(
            (environment_id, episode_index, budget_fraction, policy_id)
            for environment_id in EvaluationEnvironmentId
            for episode_index in range(self.episodes_per_environment)
            for budget_fraction in self.budget_fractions
            for policy_id in self.policy_ids
        )
        actual_keys = tuple(
            (
                row.environment_id,
                row.episode_index,
                row.budget_fraction,
                row.policy_id,
            )
            for row in self.rows
        )
        if actual_keys != expected_keys:
            raise ValueError("Budget matrix order or coverage differs")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


class DevelopmentBudgetCurveManifest(ContractModel):
    """Self-verifying record of one published development budget matrix."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_budget_curve_manifest.v1"] = (
        "acquisition_study.development_budget_curve_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    source_manifest_hash: Sha256
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    pixi_lock_sha256: Sha256
    budget_curve_file: Literal["development_budget_curve.jsonl"] = "development_budget_curve.jsonl"
    budget_curve_file_sha256: Sha256
    matrix_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    environment_count: Literal[7] = 7
    budget_count: Literal[5] = 5
    policy_count: Literal[5] = 5
    row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected_rows = (
            self.environment_count
            * self.episodes_per_environment
            * self.budget_count
            * self.policy_count
        )
        if self.row_count != expected_rows:
            raise ValueError("Budget manifest row count is inconsistent")
        if self.case_prediction_count != expected_rows * self.candidates_per_episode:
            raise ValueError("Budget manifest case count is inconsistent")
        if self.matrix_content_hash != self.replay_content_hash:
            raise ValueError("Budget matrix replay hash differs")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Budget manifest hash does not match its content")
        return self


def run_verified_development_budget_curve(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentBudgetCurveMatrix, Sha256]:
    """Run the compact development budget matrix twice and require exact agreement."""

    first = run_development_budget_curve(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    replay = run_development_budget_curve(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    if first.content_hash != replay.content_hash:
        raise DevelopmentBudgetCurveError("Development budget-curve replay differs")
    return first, replay.content_hash


def load_verified_development_budget_curve(
    manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentBudgetCurveManifest, DevelopmentBudgetCurveMatrix]:
    """Load compact budget rows only after checking their complete manifest."""

    try:
        manifest = DevelopmentBudgetCurveManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        curve_path = manifest_path.parent / manifest.budget_curve_file
        curve_bytes = curve_path.read_bytes()
        if file_sha256(curve_bytes) != manifest.budget_curve_file_sha256:
            raise DevelopmentBudgetCurveError("Development budget-curve file hash differs")
        lines = curve_bytes.decode("utf-8").splitlines()
        if len(lines) != manifest.row_count or any(not line for line in lines):
            raise DevelopmentBudgetCurveError("Development budget-curve row count differs")
        rows = tuple(BudgetCurveEpisodeMetric.model_validate(json.loads(line)) for line in lines)
        matrix = DevelopmentBudgetCurveMatrix(
            source_manifest_hash=manifest.source_manifest_hash,
            environment_specification_hash=manifest.environment_specification_hash,
            analysis_plan_hash=manifest.analysis_plan_hash,
            calibration_hash=manifest.calibration_hash,
            episodes_per_environment=manifest.episodes_per_environment,
            candidates_per_episode=manifest.candidates_per_episode,
            environment_count=manifest.environment_count,
            budget_fractions=analysis.secondary.budget_fractions,
            policy_ids=BUDGET_CURVE_POLICY_ORDER,
            row_count=manifest.row_count,
            rows=rows,
        )
    except DevelopmentBudgetCurveError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise DevelopmentBudgetCurveError(
            f"Could not verify development budget curve: {manifest_path}"
        ) from error
    if manifest.environment_specification_hash != specification.specification_hash:
        raise DevelopmentBudgetCurveError("Budget curve uses another environment specification")
    if manifest.analysis_plan_hash != analysis.plan_hash:
        raise DevelopmentBudgetCurveError("Budget curve uses another frozen analysis plan")
    if matrix.content_hash != manifest.matrix_content_hash:
        raise DevelopmentBudgetCurveError("Development budget-curve content hash differs")
    return manifest, matrix


def run_development_budget_curve(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> DevelopmentBudgetCurveMatrix:
    """Regenerate paired episodes and evaluate the complete development budget curve."""

    source_manifest, source_matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    calibration = estimate_probe_reliability(specification)
    if calibration.calibration_hash != source_manifest.calibration_hash:
        raise DevelopmentBudgetCurveError("Budget curve uses another calibration result")

    rows: list[BudgetCurveEpisodeMetric] = []
    for source_comparison in source_matrix.comparisons:
        curve = run_episode_budget_curve(
            specification,
            analysis,
            calibration,
            environment_id=source_comparison.environment_id,
            episode_index=source_comparison.episode_index,
        )
        if (
            curve.shared_candidate_hash != source_comparison.shared_candidate_hash
            or curve.privileged_episode_hash != source_comparison.privileged_episode_hash
        ):
            raise DevelopmentBudgetCurveError("Regenerated budget episode differs from its source")
        _verify_primary_budget(curve.results, source_comparison.results)
        rows.extend(_compact_metric(row, analysis.secondary.ece_bin_count) for row in curve.results)

    return DevelopmentBudgetCurveMatrix(
        source_manifest_hash=source_manifest.manifest_hash,
        environment_specification_hash=specification.specification_hash,
        analysis_plan_hash=analysis.plan_hash,
        calibration_hash=calibration.calibration_hash,
        episodes_per_environment=source_matrix.episodes_per_environment,
        candidates_per_episode=specification.episodes.candidates_per_episode,
        budget_fractions=analysis.secondary.budget_fractions,
        policy_ids=BUDGET_CURVE_POLICY_ORDER,
        row_count=len(rows),
        rows=tuple(rows),
    )


def publish_development_budget_curve(
    matrix: DevelopmentBudgetCurveMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
    pixi_lock_path: Path,
) -> DevelopmentBudgetCurveManifest:
    """Write compact restricted budget rows and their immutable manifest."""

    if matrix.content_hash != replay_content_hash:
        raise DevelopmentBudgetCurveError("Budget matrix and replay hashes differ")
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in matrix.rows)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise DevelopmentBudgetCurveError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_budget_curve_manifest.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "code_revision": code_revision,
        "split": matrix.split,
        "result_scope": matrix.result_scope,
        "access_scope": matrix.access_scope,
        "source_manifest_hash": matrix.source_manifest_hash,
        "environment_specification_hash": matrix.environment_specification_hash,
        "analysis_plan_hash": matrix.analysis_plan_hash,
        "calibration_hash": matrix.calibration_hash,
        "pixi_lock_sha256": pixi_lock_hash,
        "budget_curve_file": "development_budget_curve.jsonl",
        "budget_curve_file_sha256": file_sha256(rows_bytes),
        "matrix_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "deterministic_replay_verified": True,
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "environment_count": matrix.environment_count,
        "budget_count": len(matrix.budget_fractions),
        "policy_count": len(matrix.policy_ids),
        "row_count": matrix.row_count,
        "case_prediction_count": sum(row.candidate_count for row in matrix.rows),
        "selected_probe_count": sum(row.selected_probe_count for row in matrix.rows),
        "external_model_call_count": matrix.external_model_call_count,
        "sandbox_call_count": matrix.sandbox_call_count,
        "human_record_count": matrix.human_record_count,
        "canonical_claim_allowed": matrix.canonical_claim_allowed,
    }
    manifest = DevelopmentBudgetCurveManifest.model_validate(
        {**content, "manifest_hash": canonical_sha256(content)}
    )
    write_immutable_bytes(output_root / manifest.budget_curve_file, rows_bytes)
    write_immutable_json(output_root / "development_budget_curve_manifest.json", manifest)
    return manifest


def _verify_primary_budget(
    curve_results: tuple[BudgetPolicyResult, ...],
    source_results: tuple[PolicyEpisodeResult, ...],
) -> None:
    source_by_policy = {result.policy_id: result for result in source_results}
    primary_rows = tuple(row for row in curve_results if row.budget_fraction == 0.5)
    if tuple(row.result.policy_id for row in primary_rows) != BUDGET_CURVE_POLICY_ORDER:
        raise DevelopmentBudgetCurveError("Primary budget policy order differs")
    if any(row.result != source_by_policy.get(row.result.policy_id) for row in primary_rows):
        raise DevelopmentBudgetCurveError("Primary budget does not reproduce the source matrix")


def _compact_metric(row: BudgetPolicyResult, ece_bin_count: int) -> BudgetCurveEpisodeMetric:
    if ece_bin_count != 10:
        raise DevelopmentBudgetCurveError("Development budget records require ten ECE bins")
    result = row.result
    bin_counts = [0] * ece_bin_count
    probability_sums = [0.0] * ece_bin_count
    positive_counts = [0] * ece_bin_count
    for case in result.case_results:
        bin_index = min(int(case.final_success_probability * ece_bin_count), ece_bin_count - 1)
        bin_counts[bin_index] += 1
        probability_sums[bin_index] += case.final_success_probability
        positive_counts[bin_index] += int(case.latent_success)
    return BudgetCurveEpisodeMetric(
        environment_id=result.environment_id,
        episode_index=int(result.episode_id.removeprefix("episode-")),
        episode_id=result.episode_id,
        budget_fraction=row.budget_fraction,
        policy_id=result.policy_id,
        candidate_count=len(result.case_results),
        selected_probe_count=result.burden.selected_probe_count,
        observed_probe_pass_count=sum(
            case.observed_probe_outcome is SimulatedProbeOutcome.PASS
            for case in result.case_results
        ),
        observed_probe_fail_count=sum(
            case.observed_probe_outcome is SimulatedProbeOutcome.FAIL
            for case in result.case_results
        ),
        missing_probe_result_count=result.burden.missing_probe_result_count,
        classification_error_count=sum(case.classification_error for case in result.case_results),
        final_classification_error=result.final_classification_error,
        squared_error_sum=math.fsum(case.squared_error for case in result.case_results),
        brier_score=result.brier_score,
        negative_log_likelihood_sum=math.fsum(
            case.negative_log_likelihood for case in result.case_results
        ),
        negative_log_likelihood=result.negative_log_likelihood,
        calibration_bin_counts=tuple(bin_counts),
        calibration_probability_sums=tuple(probability_sums),
        calibration_positive_counts=tuple(positive_counts),
        decision_hash=model_content_hash(result.decision),
        case_results_hash=canonical_sha256(result.case_results),
    )
