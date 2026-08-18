"""Compact one-episode-at-a-time output for the acquisition policy study."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.budget_curve import BudgetPolicyResult
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    compact_budget_policy_result,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.evaluation import PolicyEpisodeResult
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    load_verified_development_policy_matrix,
)
from socratic_tutor.acquisition_study.runner import (
    DevelopmentPolicyMatrix,
    EpisodePolicyComparison,
    run_episode_policy_comparison,
)
from socratic_tutor.benchmark.artifacts import ArtifactConflictError, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel


class CompactPolicyStreamError(ValueError):
    """Compact output differs from the frozen policy study."""


class AcquisitionStudyPartition(StrEnum):
    """Disjoint episode ranges for development and canonical evaluation."""

    DEVELOPMENT = "development"
    EVALUATION = "evaluation"


class CompactPolicyEpisodeRecord(ContractModel):
    """Sufficient statistics and integrity hashes for one policy and episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.compact_policy_episode_record.v1"] = (
        "acquisition_study.compact_policy_episode_record.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    partition: AcquisitionStudyPartition
    shared_candidate_hash: Sha256
    privileged_episode_hash: Sha256
    policy_result_hash: Sha256
    metric: BudgetCurveEpisodeMetric
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"


class DevelopmentCompactParityPlan(ContractModel):
    """Source lineage for the development streaming-parity check."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_compact_parity_plan.v1"] = (
        "acquisition_study.development_compact_parity_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_manifest_hash: Sha256
    source_manifest_file_sha256: Sha256
    source_matrix_content_hash: Sha256
    source_comparison_file_sha256: Sha256
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    compact_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    generation_order: Literal["environment_then_episode_then_frozen_policy"] = (
        "environment_then_episode_then_frozen_policy"
    )
    generation_memory_shape: Literal["one_episode_comparison_at_a_time"] = (
        "one_episode_comparison_at_a_time"
    )
    development_episode_index_start: Literal[0] = 0
    development_episode_index_stop_exclusive: int = Field(ge=1)
    evaluation_episode_index_start: int = Field(ge=1)
    evaluation_episode_index_stop_exclusive: int = Field(ge=1)
    disjoint_episode_index_ranges_verified: Literal[True] = True
    comparison_rule: Literal[
        "exact_jsonl_bytes_from_source_projection_fresh_generation_and_replay"
    ] = "exact_jsonl_bytes_from_source_projection_fresh_generation_and_replay"
    result_scope: Literal["engineering_parity_not_scientific_result"] = (
        "engineering_parity_not_scientific_result"
    )
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if (
            self.development_episode_index_stop_exclusive != self.evaluation_episode_index_start
            or self.evaluation_episode_index_stop_exclusive <= self.evaluation_episode_index_start
        ):
            raise ValueError("Development and evaluation episode ranges are not disjoint")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Compact parity plan hash does not match its content")
        return self


class DevelopmentCompactParityReport(ContractModel):
    """Proof that compact streaming preserves the development policy outputs."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_compact_parity_report.v1"] = (
        "acquisition_study.development_compact_parity_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    parity_plan_hash: Sha256
    compact_file: Literal["development_compact_policy_metrics.jsonl"] = (
        "development_compact_policy_metrics.jsonl"
    )
    source_projection_file_sha256: Sha256
    compact_file_sha256: Sha256
    replay_file_sha256: Sha256
    exact_source_projection_parity_verified: Literal[True] = True
    deterministic_replay_verified: Literal[True] = True
    episodes_per_environment: int = Field(ge=1)
    environment_count: Literal[7] = 7
    policy_count: Literal[7] = 7
    episode_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    compact_file_bytes: int = Field(ge=1)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    result_status: Literal["development_streaming_parity_verified"] = (
        "development_streaming_parity_verified"
    )
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if (
            len(
                {
                    self.source_projection_file_sha256,
                    self.compact_file_sha256,
                    self.replay_file_sha256,
                }
            )
            != 1
        ):
            raise ValueError("Compact output does not exactly match source and replay")
        expected_episodes = self.environment_count * self.episodes_per_environment
        if self.episode_count != expected_episodes:
            raise ValueError("Compact parity episode count does not reconcile")
        if self.row_count != self.episode_count * self.policy_count:
            raise ValueError("Compact parity row count does not reconcile")
        if self.case_prediction_count != self.row_count * 40:
            raise ValueError("Compact parity prediction count does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Compact parity report hash does not match its content")
        return self


def iter_compact_policy_records(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    partition: AcquisitionStudyPartition,
    development_episode_limit: int | None = None,
) -> Iterator[CompactPolicyEpisodeRecord]:
    """Generate compact records without retaining earlier episodes."""

    episode_indices = acquisition_episode_indices(specification, partition=partition)
    if development_episode_limit is not None:
        if partition is not AcquisitionStudyPartition.DEVELOPMENT:
            raise CompactPolicyStreamError(
                "Canonical evaluation cannot use a reduced episode count"
            )
        if development_episode_limit < 1 or development_episode_limit > len(episode_indices):
            raise CompactPolicyStreamError(
                f"Development episode limit must lie in [1, {len(episode_indices)}]"
            )
        episode_indices = episode_indices[:development_episode_limit]
    calibration = estimate_probe_reliability(specification)
    for environment_id in EvaluationEnvironmentId:
        for episode_index in episode_indices:
            comparison = run_episode_policy_comparison(
                specification,
                analysis,
                calibration,
                environment_id=environment_id,
                episode_index=episode_index,
            )
            yield from _compact_comparison(comparison, analysis, partition=partition)


def acquisition_episode_indices(
    specification: AcquisitionEnvironmentSpecification,
    *,
    partition: AcquisitionStudyPartition,
) -> tuple[int, ...]:
    """Return the frozen, non-overlapping episode indices for one partition."""

    development_count = specification.episodes.development_episodes_per_environment
    if partition is AcquisitionStudyPartition.DEVELOPMENT:
        return tuple(range(development_count))
    evaluation_count = specification.episodes.evaluation_episodes_per_environment
    return tuple(range(development_count, development_count + evaluation_count))


def run_development_compact_parity(
    *,
    source_manifest_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    compact_code_revision: str,
) -> DevelopmentCompactParityReport:
    """Publish compact development rows only when source and replay bytes agree."""

    source_manifest, source_matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    try:
        source_manifest_file_hash = file_sha256(source_manifest_path.read_bytes())
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise CompactPolicyStreamError("Could not hash a compact-parity source") from error
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_compact_parity_plan.v1",
        "study_id": source_matrix.study_id,
        "run_id": run_id,
        "split": source_matrix.split,
        "source_run_id": source_manifest.run_id,
        "source_manifest_hash": source_manifest.manifest_hash,
        "source_manifest_file_sha256": source_manifest_file_hash,
        "source_matrix_content_hash": source_manifest.matrix_content_hash,
        "source_comparison_file_sha256": source_manifest.comparison_file_sha256,
        "source_code_revision": source_manifest.code_revision,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "compact_code_revision": compact_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "generation_order": "environment_then_episode_then_frozen_policy",
        "generation_memory_shape": "one_episode_comparison_at_a_time",
        "development_episode_index_start": 0,
        "development_episode_index_stop_exclusive": (
            specification.episodes.development_episodes_per_environment
        ),
        "evaluation_episode_index_start": (
            specification.episodes.development_episodes_per_environment
        ),
        "evaluation_episode_index_stop_exclusive": (
            specification.episodes.development_episodes_per_environment
            + specification.episodes.evaluation_episodes_per_environment
        ),
        "disjoint_episode_index_ranges_verified": True,
        "comparison_rule": ("exact_jsonl_bytes_from_source_projection_fresh_generation_and_replay"),
        "result_scope": "engineering_parity_not_scientific_result",
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentCompactParityPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    source_hash, source_rows, source_bytes = _hash_records(
        _project_source_matrix(source_matrix, analysis)
    )
    compact_path = output_root / "development_compact_policy_metrics.jsonl"
    compact_hash, compact_rows, compact_bytes = _write_immutable_record_stream(
        compact_path,
        iter_compact_policy_records(
            specification,
            analysis,
            partition=AcquisitionStudyPartition.DEVELOPMENT,
            development_episode_limit=source_matrix.episodes_per_environment,
        ),
    )
    replay_hash, replay_rows, replay_bytes = _hash_records(
        iter_compact_policy_records(
            specification,
            analysis,
            partition=AcquisitionStudyPartition.DEVELOPMENT,
            development_episode_limit=source_matrix.episodes_per_environment,
        )
    )
    if (source_rows, source_bytes) != (compact_rows, compact_bytes) or (
        compact_rows,
        compact_bytes,
    ) != (replay_rows, replay_bytes):
        raise CompactPolicyStreamError("Compact output row count or size differs")
    if len({source_hash, compact_hash, replay_hash}) != 1:
        raise CompactPolicyStreamError("Compact output differs from source or replay")
    report_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_compact_parity_report.v1",
        "study_id": source_matrix.study_id,
        "run_id": run_id,
        "split": source_matrix.split,
        "parity_plan_hash": plan.plan_hash,
        "compact_file": "development_compact_policy_metrics.jsonl",
        "source_projection_file_sha256": source_hash,
        "compact_file_sha256": compact_hash,
        "replay_file_sha256": replay_hash,
        "exact_source_projection_parity_verified": True,
        "deterministic_replay_verified": True,
        "episodes_per_environment": source_matrix.episodes_per_environment,
        "environment_count": source_matrix.environment_count,
        "policy_count": source_matrix.policy_count,
        "episode_count": source_matrix.comparison_count,
        "row_count": compact_rows,
        "case_prediction_count": sum(
            len(result.case_results)
            for comparison in source_matrix.comparisons
            for result in comparison.results
        ),
        "compact_file_bytes": compact_bytes,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        "canonical_claim_allowed": False,
        "result_status": "development_streaming_parity_verified",
    }
    report = DevelopmentCompactParityReport.model_validate(
        {**report_content, "report_hash": canonical_sha256(report_content)}
    )
    write_immutable_json(output_root / "development_compact_parity_plan.json", plan)
    write_immutable_json(output_root / "development_compact_parity_report.json", report)
    return report


def _project_source_matrix(
    matrix: DevelopmentPolicyMatrix,
    analysis: AcquisitionAnalysisSpecification,
) -> Iterator[CompactPolicyEpisodeRecord]:
    for comparison in matrix.comparisons:
        yield from _compact_comparison(
            comparison,
            analysis,
            partition=AcquisitionStudyPartition.DEVELOPMENT,
        )


def _compact_comparison(
    comparison: EpisodePolicyComparison,
    analysis: AcquisitionAnalysisSpecification,
    *,
    partition: AcquisitionStudyPartition,
) -> Iterator[CompactPolicyEpisodeRecord]:
    for result in comparison.results:
        yield _compact_result(comparison, result, analysis, partition=partition)


def _compact_result(
    comparison: EpisodePolicyComparison,
    result: PolicyEpisodeResult,
    analysis: AcquisitionAnalysisSpecification,
    *,
    partition: AcquisitionStudyPartition,
) -> CompactPolicyEpisodeRecord:
    candidate_count = result.burden.candidate_count
    budget_fraction = result.burden.selected_probe_count / candidate_count
    metric = compact_budget_policy_result(
        BudgetPolicyResult(
            budget_fraction=budget_fraction,
            selected_probe_count=result.burden.selected_probe_count,
            result=result,
        ),
        analysis.secondary.ece_bin_count,
    )
    return CompactPolicyEpisodeRecord(
        partition=partition,
        shared_candidate_hash=comparison.shared_candidate_hash,
        privileged_episode_hash=comparison.privileged_episode_hash,
        policy_result_hash=model_content_hash(result),
        metric=metric,
    )


def _hash_records(
    records: Iterable[CompactPolicyEpisodeRecord],
) -> tuple[Sha256, int, int]:
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    for record in records:
        content = canonical_json_bytes(record) + b"\n"
        digest.update(content)
        row_count += 1
        byte_count += len(content)
    return digest.hexdigest(), row_count, byte_count


def _write_immutable_record_stream(
    path: Path,
    records: Iterable[CompactPolicyEpisodeRecord],
) -> tuple[Sha256, int, int]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    try:
        with temporary.open("xb") as handle:
            for record in records:
                content = canonical_json_bytes(record) + b"\n"
                handle.write(content)
                digest.update(content)
                row_count += 1
                byte_count += len(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if not _files_equal(path, temporary):
                raise ArtifactConflictError(f"Immutable artifact already differs: {path}")
        else:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest(), row_count, byte_count


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
