"""Bounded-memory public and restricted streams for the canonical study."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.budget_curve import BudgetPolicyResult
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    compact_budget_policy_result,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.compact_stream import (
    AcquisitionStudyPartition,
    acquisition_episode_indices,
)
from socratic_tutor.acquisition_study.contracts import AcquisitionDecision, PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.runner import EpisodePolicyRun, run_episode_policy_bundle
from socratic_tutor.acquisition_study.simulation import (
    PolicyEpisodeRecord,
    PrivilegedEpisodeRecord,
)
from socratic_tutor.benchmark.artifacts import ArtifactConflictError
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel

_PUBLIC_FILE = "canonical_public_policy_metrics.jsonl"
_RESTRICTED_FILE = "canonical_restricted_episodes.jsonl"
_POLICY_ORDER = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
    PolicyId.NEVER_PROBE,
    PolicyId.ALWAYS_PROBE_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)


class CanonicalStreamError(ValueError):
    """Canonical output conflicts with the frozen study design."""


class CanonicalPublicPolicyMetric(ContractModel):
    """Aggregate policy result that contains no simulator-owned case truth."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_public_policy_metric.v1"] = (
        "acquisition_study.canonical_public_policy_metric.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    result_scope: Literal["canonical_glass_box_evaluation"] = "canonical_glass_box_evaluation"
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    shared_candidate_hash: Sha256
    policy_result_hash: Sha256
    metric: BudgetCurveEpisodeMetric
    access_scope: Literal["public_aggregate"] = "public_aggregate"
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False


class CanonicalRestrictedEpisode(ContractModel):
    """Complete audit record for one hidden episode and all policy decisions."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_restricted_episode.v1"] = (
        "acquisition_study.canonical_restricted_episode.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    result_scope: Literal["canonical_glass_box_evaluation"] = "canonical_glass_box_evaluation"
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    policy_episode: PolicyEpisodeRecord
    privileged_episode: PrivilegedEpisodeRecord
    shared_candidate_hash: Sha256
    privileged_episode_hash: Sha256
    decisions: tuple[AcquisitionDecision, ...] = Field(min_length=7, max_length=7)
    policy_result_hashes: tuple[Sha256, ...] = Field(min_length=7, max_length=7)
    comparison_hash: Sha256
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_episode(self) -> Self:
        episode_ids = {
            self.policy_episode.episode_id,
            self.privileged_episode.episode_id,
        }
        if len(episode_ids) != 1:
            raise ValueError("Canonical episode projections use different identifiers")
        if (
            self.environment_specification_hash
            != self.policy_episode.environment_specification_hash
        ):
            raise ValueError("Safe episode uses another environment specification")
        if (
            self.environment_specification_hash
            != self.privileged_episode.environment_specification_hash
        ):
            raise ValueError("Privileged episode uses another environment specification")
        if self.calibration_hash != self.policy_episode.calibration_hash:
            raise ValueError("Safe episode uses another calibration result")
        if tuple(decision.policy_id for decision in self.decisions) != _POLICY_ORDER:
            raise ValueError("Canonical decisions differ from the frozen policy order")
        candidate_ids = {candidate.case_id for candidate in self.policy_episode.request.candidates}
        truth_ids = {case.case_id for case in self.privileged_episode.cases}
        if candidate_ids != truth_ids:
            raise ValueError("Canonical safe and privileged case sets differ")
        expected_decision_sizes = (
            self.policy_episode.request.remaining_probe_budget,
            self.policy_episode.request.remaining_probe_budget,
            self.policy_episode.request.remaining_probe_budget,
            self.policy_episode.request.remaining_probe_budget,
            0,
            len(candidate_ids),
            self.policy_episode.request.remaining_probe_budget,
        )
        if tuple(len(decision.selected_case_ids) for decision in self.decisions) != (
            expected_decision_sizes
        ):
            raise ValueError("Canonical decisions violate the frozen probe budgets")
        if any(
            not set(decision.selected_case_ids).issubset(candidate_ids)
            for decision in self.decisions
        ):
            raise ValueError("Canonical decision selects an unknown case")
        if self.shared_candidate_hash != canonical_sha256(self.policy_episode.request.candidates):
            raise ValueError("Canonical candidate hash does not match the safe request")
        if self.privileged_episode_hash != model_content_hash(self.privileged_episode):
            raise ValueError("Canonical privileged hash does not match simulator truth")
        return self


class CanonicalStreamReport(ContractModel):
    """Counts and exact hashes for one canonical write and deterministic retry."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_stream_report.v1"] = (
        "acquisition_study.canonical_stream_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    public_schema_hash: Sha256
    restricted_schema_hash: Sha256
    public_file: Literal["canonical_public_policy_metrics.jsonl"] = _PUBLIC_FILE
    public_file_sha256: Sha256
    public_replay_sha256: Sha256
    public_file_bytes: int = Field(ge=1)
    restricted_file: Literal["canonical_restricted_episodes.jsonl"] = _RESTRICTED_FILE
    restricted_file_sha256: Sha256
    restricted_replay_sha256: Sha256
    restricted_file_bytes: int = Field(ge=1)
    environment_count: Literal[7] = 7
    environment_ids: tuple[EvaluationEnvironmentId, ...] = Field(min_length=7, max_length=7)
    episodes_per_environment: int = Field(ge=1)
    evaluation_episode_index_start: int = Field(ge=1)
    evaluation_episode_index_stop_exclusive: int = Field(ge=2)
    episode_count: int = Field(ge=1)
    policy_count: Literal[7] = 7
    policy_ids: tuple[PolicyId, ...] = Field(min_length=7, max_length=7)
    candidates_per_episode: int = Field(ge=1)
    matched_probe_budget: int = Field(ge=1)
    public_row_count: int = Field(ge=1)
    restricted_row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    missing_probe_result_count: int = Field(ge=0)
    deterministic_replay_verified: Literal[True] = True
    canonical_estimate_count: Literal[1] = 1
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.public_file_sha256 != self.public_replay_sha256:
            raise ValueError("Public canonical retry differs from the written stream")
        if self.restricted_file_sha256 != self.restricted_replay_sha256:
            raise ValueError("Restricted canonical retry differs from the written stream")
        expected_episodes = self.environment_count * self.episodes_per_environment
        if self.environment_ids != tuple(EvaluationEnvironmentId):
            raise ValueError("Canonical report differs from the frozen environment order")
        if self.policy_ids != _POLICY_ORDER:
            raise ValueError("Canonical report differs from the frozen policy order")
        if (
            self.evaluation_episode_index_stop_exclusive - self.evaluation_episode_index_start
            != self.episodes_per_environment
        ):
            raise ValueError("Canonical evaluation index range does not reconcile")
        if self.episode_count != expected_episodes:
            raise ValueError("Canonical episode count does not reconcile")
        if self.restricted_row_count != self.episode_count:
            raise ValueError("Canonical restricted row count does not reconcile")
        if self.public_row_count != self.episode_count * self.policy_count:
            raise ValueError("Canonical public row count does not reconcile")
        if self.case_prediction_count != self.public_row_count * self.candidates_per_episode:
            raise ValueError("Canonical case-prediction count does not reconcile")
        expected_selected = self.episode_count * (
            5 * self.matched_probe_budget + self.candidates_per_episode
        )
        if self.selected_probe_count != expected_selected:
            raise ValueError("Canonical selected-probe count violates the frozen budgets")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Canonical stream report hash does not match its content")
        return self


def canonical_stream_schema_hashes() -> tuple[Sha256, Sha256]:
    """Return stable hashes for the public and restricted record contracts."""

    return (
        canonical_sha256(CanonicalPublicPolicyMetric.model_json_schema()),
        canonical_sha256(CanonicalRestrictedEpisode.model_json_schema()),
    )


def project_canonical_episode(
    run: EpisodePolicyRun,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[CanonicalRestrictedEpisode, tuple[CanonicalPublicPolicyMetric, ...]]:
    """Split one simulator run into restricted evidence and public metrics."""

    comparison = run.comparison
    restricted = CanonicalRestrictedEpisode(
        environment_specification_hash=run.policy_episode.environment_specification_hash,
        analysis_plan_hash=analysis.plan_hash,
        calibration_hash=run.policy_episode.calibration_hash,
        policy_episode=run.policy_episode,
        privileged_episode=run.privileged_episode,
        shared_candidate_hash=comparison.shared_candidate_hash,
        privileged_episode_hash=comparison.privileged_episode_hash,
        decisions=tuple(result.decision for result in comparison.results),
        policy_result_hashes=tuple(model_content_hash(result) for result in comparison.results),
        comparison_hash=model_content_hash(comparison),
    )
    public = tuple(
        CanonicalPublicPolicyMetric(
            environment_specification_hash=run.policy_episode.environment_specification_hash,
            analysis_plan_hash=analysis.plan_hash,
            calibration_hash=run.policy_episode.calibration_hash,
            shared_candidate_hash=comparison.shared_candidate_hash,
            policy_result_hash=model_content_hash(result),
            metric=compact_budget_policy_result(
                BudgetPolicyResult(
                    budget_fraction=(
                        result.burden.selected_probe_count / result.burden.candidate_count
                    ),
                    selected_probe_count=result.burden.selected_probe_count,
                    result=result,
                ),
                analysis.secondary.ece_bin_count,
            ),
        )
        for result in comparison.results
    )
    return restricted, public


def write_verified_canonical_streams(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    output_root: Path,
) -> CanonicalStreamReport:
    """Write the full evaluation once, then verify it by an exact seeded retry."""

    _validate_plan_pair(specification, analysis)
    episode_indices = _evaluation_episode_indices(specification)
    first, replay = _materialise_verified_streams(
        specification,
        analysis,
        episode_indices=episode_indices,
        output_root=output_root,
    )

    public_schema_hash, restricted_schema_hash = canonical_stream_schema_hashes()
    calibration = estimate_probe_reliability(specification)
    public_hash = first.public_hash
    restricted_hash = first.restricted_hash
    public_rows = first.public_rows
    restricted_rows = first.restricted_rows
    candidates_per_episode = specification.episodes.candidates_per_episode
    report_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_stream_report.v1",
        "study_id": specification.study_id,
        "split": "evaluation",
        "environment_specification_hash": specification.specification_hash,
        "analysis_plan_hash": analysis.plan_hash,
        "calibration_hash": calibration.calibration_hash,
        "public_schema_hash": public_schema_hash,
        "restricted_schema_hash": restricted_schema_hash,
        "public_file": _PUBLIC_FILE,
        "public_file_sha256": public_hash,
        "public_replay_sha256": replay.public_hash,
        "public_file_bytes": first.public_bytes,
        "restricted_file": _RESTRICTED_FILE,
        "restricted_file_sha256": restricted_hash,
        "restricted_replay_sha256": replay.restricted_hash,
        "restricted_file_bytes": first.restricted_bytes,
        "environment_count": len(EvaluationEnvironmentId),
        "environment_ids": tuple(EvaluationEnvironmentId),
        "episodes_per_environment": len(episode_indices),
        "evaluation_episode_index_start": episode_indices[0],
        "evaluation_episode_index_stop_exclusive": episode_indices[-1] + 1,
        "episode_count": restricted_rows,
        "policy_count": 7,
        "policy_ids": _POLICY_ORDER,
        "candidates_per_episode": candidates_per_episode,
        "matched_probe_budget": analysis.primary.exact_selected_probes_per_episode,
        "public_row_count": public_rows,
        "restricted_row_count": restricted_rows,
        "case_prediction_count": public_rows * candidates_per_episode,
        "selected_probe_count": first.selected_probe_count,
        "missing_probe_result_count": first.missing_probe_result_count,
        "deterministic_replay_verified": True,
        "canonical_estimate_count": 1,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    return CanonicalStreamReport.model_validate(
        {**report_content, "report_hash": canonical_sha256(report_content)}
    )


def _iter_episode_projections(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    episode_indices: tuple[int, ...],
) -> Iterator[tuple[CanonicalRestrictedEpisode, tuple[CanonicalPublicPolicyMetric, ...]]]:
    calibration = estimate_probe_reliability(specification)
    for environment_id in EvaluationEnvironmentId:
        for episode_index in episode_indices:
            yield project_canonical_episode(
                run_episode_policy_bundle(
                    specification,
                    analysis,
                    calibration,
                    environment_id=environment_id,
                    episode_index=episode_index,
                ),
                analysis,
            )


def _evaluation_episode_indices(
    specification: AcquisitionEnvironmentSpecification,
) -> tuple[int, ...]:
    return acquisition_episode_indices(
        specification,
        partition=AcquisitionStudyPartition.EVALUATION,
    )


def _materialise_verified_streams(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    episode_indices: tuple[int, ...],
    output_root: Path,
) -> tuple[_StreamSummary, _StreamSummary]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    public_path = output_root / _PUBLIC_FILE
    restricted_path = output_root / _RESTRICTED_FILE
    public_temporary = public_path.with_name(f".{public_path.name}.{uuid4().hex}.tmp")
    restricted_temporary = restricted_path.with_name(f".{restricted_path.name}.{uuid4().hex}.tmp")
    try:
        with (
            public_temporary.open("xb") as public_handle,
            restricted_temporary.open("xb") as restricted_handle,
        ):
            result = _consume_projections(
                _iter_episode_projections(
                    specification,
                    analysis,
                    episode_indices=episode_indices,
                ),
                public_handle=public_handle,
                restricted_handle=restricted_handle,
            )
            public_handle.flush()
            restricted_handle.flush()
            os.fsync(public_handle.fileno())
            os.fsync(restricted_handle.fileno())
        replay = _hash_streams(
            _iter_episode_projections(
                specification,
                analysis,
                episode_indices=episode_indices,
            )
        )
        if result != replay:
            raise CanonicalStreamError("Canonical deterministic retry differs from the first run")
        _verify_existing_or_absent(public_path, public_temporary)
        _verify_existing_or_absent(restricted_path, restricted_temporary)
        if not public_path.exists():
            os.replace(public_temporary, public_path)
        if not restricted_path.exists():
            os.replace(restricted_temporary, restricted_path)
        _fsync_directory(output_root)
        return result, replay
    finally:
        public_temporary.unlink(missing_ok=True)
        restricted_temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _StreamSummary:
    public_hash: Sha256
    public_rows: int
    public_bytes: int
    restricted_hash: Sha256
    restricted_rows: int
    restricted_bytes: int
    selected_probe_count: int
    missing_probe_result_count: int


def _consume_projections(
    projections: Iterator[
        tuple[CanonicalRestrictedEpisode, tuple[CanonicalPublicPolicyMetric, ...]]
    ],
    *,
    public_handle: BinaryIO | None = None,
    restricted_handle: BinaryIO | None = None,
) -> _StreamSummary:
    public_digest = hashlib.sha256()
    restricted_digest = hashlib.sha256()
    public_rows = public_bytes = restricted_rows = restricted_bytes = 0
    selected_probe_count = missing_probe_result_count = 0
    for restricted, public_records in projections:
        restricted_content = canonical_json_bytes(restricted) + b"\n"
        if restricted_handle is not None:
            restricted_handle.write(restricted_content)
        restricted_digest.update(restricted_content)
        restricted_rows += 1
        restricted_bytes += len(restricted_content)
        for public in public_records:
            public_content = canonical_json_bytes(public) + b"\n"
            if public_handle is not None:
                public_handle.write(public_content)
            public_digest.update(public_content)
            public_rows += 1
            public_bytes += len(public_content)
            selected_probe_count += public.metric.selected_probe_count
            missing_probe_result_count += public.metric.missing_probe_result_count
    return _StreamSummary(
        public_hash=public_digest.hexdigest(),
        public_rows=public_rows,
        public_bytes=public_bytes,
        restricted_hash=restricted_digest.hexdigest(),
        restricted_rows=restricted_rows,
        restricted_bytes=restricted_bytes,
        selected_probe_count=selected_probe_count,
        missing_probe_result_count=missing_probe_result_count,
    )


def _hash_streams(
    projections: Iterator[
        tuple[CanonicalRestrictedEpisode, tuple[CanonicalPublicPolicyMetric, ...]]
    ],
) -> _StreamSummary:
    return _consume_projections(projections)


def _validate_plan_pair(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> None:
    if specification.specification_hash != analysis.environment_specification_hash:
        raise CanonicalStreamError("Analysis and environment specifications do not match")


def _verify_existing_or_absent(path: Path, temporary: Path) -> None:
    if path.exists() and not _files_equal(path, temporary):
        raise ArtifactConflictError(f"Immutable artifact already differs: {path}")


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
