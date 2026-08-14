"""Verified development matrix for the frozen acquisition-policy ablations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.ablations import (
    AblationPolicyCellId,
    EpisodeAblationComparison,
    ProbeClassSelectionCount,
    run_episode_ablation_comparison,
)
from socratic_tutor.acquisition_study.budget_curve import BudgetPolicyResult
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    compact_budget_policy_result,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import PolicyId, ProbeClassId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    load_verified_development_policy_matrix,
)
from socratic_tutor.acquisition_study.runner import EpisodePolicyComparison
from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel


class DevelopmentAblationError(ValueError):
    """An ablation source or result violates the frozen development design."""


class AblationEpisodeMetric(ContractModel):
    """Compact metrics and selections for one ablation cell and episode."""

    cell_id: AblationPolicyCellId
    reliability_summary: Literal["lower_quantile", "posterior_mean"]
    update_rule: Literal["bounded", "unbounded"]
    metric: BudgetCurveEpisodeMetric
    selection_by_probe_class: tuple[ProbeClassSelectionCount, ...] = Field(
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        expected = {
            AblationPolicyCellId.FULL_CANDIDATE: (
                PolicyId.RELIABILITY_AWARE_BOUNDED,
                "lower_quantile",
                "bounded",
            ),
            AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED: (
                PolicyId.PLUG_IN_EVSI_BOUNDED,
                "posterior_mean",
                "bounded",
            ),
            AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED: (
                PolicyId.ABLATION_QUANTILE_UNBOUNDED,
                "lower_quantile",
                "unbounded",
            ),
            AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED: (
                PolicyId.ABLATION_MEAN_UNBOUNDED,
                "posterior_mean",
                "unbounded",
            ),
        }[self.cell_id]
        if (self.metric.policy_id, self.reliability_summary, self.update_rule) != expected:
            raise ValueError("Ablation metric does not match its component cell")
        if not math.isclose(self.metric.budget_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("Ablation metric does not use the fixed primary budget")
        if tuple(row.probe_class for row in self.selection_by_probe_class) != tuple(ProbeClassId):
            raise ValueError("Ablation metric does not cover every probe class")
        if sum(row.available_case_count for row in self.selection_by_probe_class) != (
            self.metric.candidate_count
        ):
            raise ValueError("Ablation class availability does not cover every case")
        if sum(row.selected_case_count for row in self.selection_by_probe_class) != (
            self.metric.selected_probe_count
        ):
            raise ValueError("Ablation class selections do not match the probe count")
        return self


class DevelopmentAblationMatrix(ContractModel):
    """Compact component comparisons over every development episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_ablation_matrix.v1"] = (
        "acquisition_study.development_ablation_matrix.v1"
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
    fixed_probe_budget: int = Field(ge=1)
    fixed_budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    environment_count: Literal[7] = 7
    cell_ids: tuple[AblationPolicyCellId, ...] = Field(min_length=4, max_length=4)
    row_count: int = Field(ge=1)
    rows: tuple[AblationEpisodeMetric, ...]
    bounded_source_reproduction_verified: Literal[True] = True
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.environment_count != len(EvaluationEnvironmentId):
            raise ValueError("Ablation matrix does not contain every environment")
        if self.cell_ids != tuple(AblationPolicyCellId):
            raise ValueError("Ablation matrix differs from the frozen component cells")
        if not math.isclose(self.fixed_budget_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("Ablation matrix differs from the primary budget fraction")
        if self.fixed_probe_budget != self.candidates_per_episode * self.fixed_budget_fraction:
            raise ValueError("Ablation matrix probe count differs from its budget fraction")
        expected_count = self.environment_count * self.episodes_per_environment * len(self.cell_ids)
        if self.row_count != expected_count or len(self.rows) != expected_count:
            raise ValueError("Ablation matrix row count is inconsistent")
        expected_keys = tuple(
            (environment_id, episode_index, cell_id)
            for environment_id in EvaluationEnvironmentId
            for episode_index in range(self.episodes_per_environment)
            for cell_id in self.cell_ids
        )
        actual_keys = tuple(
            (row.metric.environment_id, row.metric.episode_index, row.cell_id) for row in self.rows
        )
        if actual_keys != expected_keys:
            raise ValueError("Ablation matrix order or coverage differs")
        if any(
            row.metric.candidate_count != self.candidates_per_episode
            or row.metric.selected_probe_count != self.fixed_probe_budget
            for row in self.rows
        ):
            raise ValueError("Ablation matrix rows use another episode shape")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


class DevelopmentAblationManifest(ContractModel):
    """Self-verifying record of one published development ablation matrix."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_ablation_manifest.v1"] = (
        "acquisition_study.development_ablation_manifest.v1"
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
    ablation_file: Literal["development_ablation_matrix.jsonl"] = (
        "development_ablation_matrix.jsonl"
    )
    ablation_file_sha256: Sha256
    matrix_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    bounded_source_reproduction_verified: Literal[True] = True
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    environment_count: Literal[7] = 7
    cell_count: Literal[4] = 4
    row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=1)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected_rows = self.environment_count * self.episodes_per_environment * self.cell_count
        if self.row_count != expected_rows:
            raise ValueError("Ablation manifest row count is inconsistent")
        if self.case_prediction_count != expected_rows * self.candidates_per_episode:
            raise ValueError("Ablation manifest case count is inconsistent")
        if self.selected_probe_count != expected_rows * self.fixed_probe_budget:
            raise ValueError("Ablation manifest probe count is inconsistent")
        if self.matrix_content_hash != self.replay_content_hash:
            raise ValueError("Ablation matrix replay hash differs")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Ablation manifest hash does not match its content")
        return self


def run_verified_development_ablation_matrix(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentAblationMatrix, Sha256]:
    """Run the development ablation matrix twice and require exact agreement."""

    first = run_development_ablation_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    replay = run_development_ablation_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    if first.content_hash != replay.content_hash:
        raise DevelopmentAblationError("Development ablation replay differs")
    return first, replay.content_hash


def run_development_ablation_matrix(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> DevelopmentAblationMatrix:
    """Evaluate all component cells over the published development episodes."""

    source_manifest, source_matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    calibration = estimate_probe_reliability(specification)
    if calibration.calibration_hash != source_manifest.calibration_hash:
        raise DevelopmentAblationError("Ablations use another calibration result")
    rows: list[AblationEpisodeMetric] = []
    for source_comparison in source_matrix.comparisons:
        comparison = run_episode_ablation_comparison(
            specification,
            analysis,
            calibration,
            environment_id=source_comparison.environment_id,
            episode_index=source_comparison.episode_index,
        )
        _verify_source_episode(comparison, source_comparison)
        for cell in comparison.cells:
            rows.append(
                AblationEpisodeMetric(
                    cell_id=cell.cell_id,
                    reliability_summary=cell.reliability_summary,
                    update_rule=cell.update_rule,
                    metric=compact_budget_policy_result(
                        BudgetPolicyResult(
                            budget_fraction=analysis.primary.primary_budget_fraction,
                            selected_probe_count=cell.result.burden.selected_probe_count,
                            result=cell.result,
                        ),
                        analysis.secondary.ece_bin_count,
                    ),
                    selection_by_probe_class=cell.selection_by_probe_class,
                )
            )
    return DevelopmentAblationMatrix(
        source_manifest_hash=source_manifest.manifest_hash,
        environment_specification_hash=specification.specification_hash,
        analysis_plan_hash=analysis.plan_hash,
        calibration_hash=calibration.calibration_hash,
        episodes_per_environment=source_matrix.episodes_per_environment,
        candidates_per_episode=specification.episodes.candidates_per_episode,
        fixed_probe_budget=analysis.primary.exact_selected_probes_per_episode,
        fixed_budget_fraction=analysis.primary.primary_budget_fraction,
        cell_ids=tuple(AblationPolicyCellId),
        row_count=len(rows),
        rows=tuple(rows),
    )


def publish_development_ablation_matrix(
    matrix: DevelopmentAblationMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
    pixi_lock_path: Path,
) -> DevelopmentAblationManifest:
    """Write compact restricted ablation rows and their immutable manifest."""

    if matrix.content_hash != replay_content_hash:
        raise DevelopmentAblationError("Ablation matrix and replay hashes differ")
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in matrix.rows)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise DevelopmentAblationError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_ablation_manifest.v1",
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
        "ablation_file": "development_ablation_matrix.jsonl",
        "ablation_file_sha256": file_sha256(rows_bytes),
        "matrix_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "deterministic_replay_verified": True,
        "bounded_source_reproduction_verified": matrix.bounded_source_reproduction_verified,
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "fixed_probe_budget": matrix.fixed_probe_budget,
        "environment_count": matrix.environment_count,
        "cell_count": len(matrix.cell_ids),
        "row_count": matrix.row_count,
        "case_prediction_count": sum(row.metric.candidate_count for row in matrix.rows),
        "selected_probe_count": sum(row.metric.selected_probe_count for row in matrix.rows),
        "external_model_call_count": matrix.external_model_call_count,
        "sandbox_call_count": matrix.sandbox_call_count,
        "human_record_count": matrix.human_record_count,
        "canonical_claim_allowed": matrix.canonical_claim_allowed,
    }
    manifest = DevelopmentAblationManifest.model_validate(
        {**content, "manifest_hash": canonical_sha256(content)}
    )
    write_immutable_bytes(output_root / manifest.ablation_file, rows_bytes)
    write_immutable_json(output_root / "development_ablation_manifest.json", manifest)
    return manifest


def load_verified_development_ablation_matrix(
    manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentAblationManifest, DevelopmentAblationMatrix]:
    """Load compact ablation rows only after checking the complete manifest."""

    try:
        manifest = DevelopmentAblationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        matrix_path = manifest_path.parent / manifest.ablation_file
        matrix_bytes = matrix_path.read_bytes()
        if file_sha256(matrix_bytes) != manifest.ablation_file_sha256:
            raise DevelopmentAblationError("Development ablation file hash differs")
        lines = matrix_bytes.decode("utf-8").splitlines()
        if len(lines) != manifest.row_count or any(not line for line in lines):
            raise DevelopmentAblationError("Development ablation row count differs")
        rows = tuple(AblationEpisodeMetric.model_validate(json.loads(line)) for line in lines)
        matrix = DevelopmentAblationMatrix(
            source_manifest_hash=manifest.source_manifest_hash,
            environment_specification_hash=manifest.environment_specification_hash,
            analysis_plan_hash=manifest.analysis_plan_hash,
            calibration_hash=manifest.calibration_hash,
            episodes_per_environment=manifest.episodes_per_environment,
            candidates_per_episode=manifest.candidates_per_episode,
            fixed_probe_budget=manifest.fixed_probe_budget,
            fixed_budget_fraction=analysis.primary.primary_budget_fraction,
            environment_count=manifest.environment_count,
            cell_ids=tuple(AblationPolicyCellId),
            row_count=manifest.row_count,
            rows=rows,
        )
    except DevelopmentAblationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise DevelopmentAblationError(
            f"Could not verify development ablation matrix: {manifest_path}"
        ) from error
    if manifest.environment_specification_hash != specification.specification_hash:
        raise DevelopmentAblationError("Ablations use another environment specification")
    if manifest.analysis_plan_hash != analysis.plan_hash:
        raise DevelopmentAblationError("Ablations use another frozen analysis plan")
    if matrix.content_hash != manifest.matrix_content_hash:
        raise DevelopmentAblationError("Development ablation matrix content hash differs")
    return manifest, matrix


def _verify_source_episode(
    comparison: EpisodeAblationComparison,
    source: EpisodePolicyComparison,
) -> None:
    if (
        comparison.shared_candidate_hash != source.shared_candidate_hash
        or comparison.privileged_episode_hash != source.privileged_episode_hash
    ):
        raise DevelopmentAblationError("Regenerated ablation episode differs from its source")
    source_by_policy = {result.policy_id: result for result in source.results}
    bounded_cells = comparison.cells[:2]
    if any(cell.result != source_by_policy.get(cell.result.policy_id) for cell in bounded_cells):
        raise DevelopmentAblationError("Bounded ablation cells do not reproduce source results")
