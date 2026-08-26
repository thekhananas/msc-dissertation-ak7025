"""Verified secondary budget curve over the sealed canonical episodes."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.budget_curve import (
    BUDGET_CURVE_POLICY_ORDER,
    BudgetPolicyResult,
    run_episode_budget_curve,
)
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    compact_budget_policy_result,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.canonical_execution import (
    load_canonical_execution_plan,
)
from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    CanonicalPrimaryAnalysisManifest,
    CanonicalPrimaryAnalysisReport,
    CanonicalPrimarySourcePlan,
    load_canonical_primary_source_plan,
)
from socratic_tutor.acquisition_study.canonical_stream import CanonicalPublicPolicyMetric
from socratic_tutor.acquisition_study.compact_stream import (
    AcquisitionStudyPartition,
    acquisition_episode_indices,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.benchmark.artifacts import (
    ArtifactConflictError,
    immutable_json_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.filesystem import files_equal, fsync_directory

_SOURCE_POLICY_ORDER = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
    PolicyId.NEVER_PROBE,
    PolicyId.ALWAYS_PROBE_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)
_BUDGET_FILE = "canonical_budget_curve_metrics.jsonl"
_PLAN_FILE = "canonical_budget_curve_execution_plan.json"
_REPORT_FILE = "canonical_budget_curve_report.json"
_MANIFEST_FILE = "canonical_budget_curve_manifest.json"


class CanonicalBudgetCurveError(ValueError):
    """A source or generated row differs from the frozen secondary design."""


class CanonicalBudgetCurveRecord(ContractModel):
    """Public aggregate metrics for one policy, budget and sealed episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_budget_curve_metric.v1"] = (
        "acquisition_study.canonical_budget_curve_metric.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    result_scope: Literal["canonical_secondary_descriptive"] = "canonical_secondary_descriptive"
    shared_candidate_hash: Sha256
    policy_result_hash: Sha256
    metric: BudgetCurveEpisodeMetric
    access_scope: Literal["public_aggregate"] = "public_aggregate"
    primary_claim_rescue_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False


class CanonicalBudgetCurveExecutionPlan(ContractModel):
    """Exact lineage and fixed rules for the post-primary secondary execution."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_budget_curve_execution_plan.v1"] = (
        "acquisition_study.canonical_budget_curve_execution_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    source_plan_hash: Sha256
    source_plan_file_sha256: Sha256
    canonical_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    secondary_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    canonical_run_manifest_hash: Sha256
    canonical_stream_report_hash: Sha256
    source_public_stream_sha256: Sha256
    primary_analysis_manifest_hash: Sha256
    primary_analysis_manifest_file_sha256: Sha256
    primary_analysis_report_hash: Sha256
    primary_analysis_report_file_sha256: Sha256
    frozen_analysis_specification_hash: Sha256
    pixi_lock_sha256: Sha256
    environment_ids: tuple[EvaluationEnvironmentId, ...] = Field(min_length=7, max_length=7)
    source_policy_ids: tuple[PolicyId, ...] = Field(min_length=7, max_length=7)
    budget_policy_ids: tuple[PolicyId, ...] = Field(min_length=5, max_length=5)
    budget_fractions: tuple[float, ...] = Field(min_length=5, max_length=5)
    evaluation_episode_index_start: int = Field(ge=1)
    evaluation_episode_index_stop_exclusive: int = Field(ge=2)
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    source_public_row_count: int = Field(ge=1)
    expected_budget_row_count: int = Field(ge=1)
    primary_result_known_before_secondary_execution: Literal[True] = True
    budget_fractions_and_metrics_frozen_before_primary_result: Literal[True] = True
    post_primary_policy_tuning_allowed: Literal[False] = False
    secondary_analysis_role: Literal["descriptive_cannot_rescue_primary"] = (
        "descriptive_cannot_rescue_primary"
    )
    restricted_stream_access_allowed: Literal[False] = False
    external_calls_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.environment_ids != tuple(EvaluationEnvironmentId):
            raise ValueError("Budget plan differs from the frozen environment order")
        if self.source_policy_ids != _SOURCE_POLICY_ORDER:
            raise ValueError("Budget plan differs from the sealed source policy order")
        if self.budget_policy_ids != BUDGET_CURVE_POLICY_ORDER:
            raise ValueError("Budget plan differs from the frozen budget policy order")
        if self.budget_fractions != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Budget plan differs from the frozen fractions")
        if (
            self.evaluation_episode_index_stop_exclusive - self.evaluation_episode_index_start
            != self.episodes_per_environment
        ):
            raise ValueError("Budget plan episode range does not reconcile")
        expected_source = (
            len(self.environment_ids) * self.episodes_per_environment * len(self.source_policy_ids)
        )
        expected_budget = (
            len(self.environment_ids)
            * self.episodes_per_environment
            * len(self.budget_fractions)
            * len(self.budget_policy_ids)
        )
        if self.source_public_row_count != expected_source:
            raise ValueError("Budget plan source row count does not reconcile")
        if self.expected_budget_row_count != expected_budget:
            raise ValueError("Budget plan output row count does not reconcile")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Budget execution-plan hash does not match its content")
        return self


class CanonicalBudgetCurveReport(ContractModel):
    """Integrity and workload summary for the canonical secondary stream."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_budget_curve_report.v1"] = (
        "acquisition_study.canonical_budget_curve_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    result_status: Literal["canonical_budget_curve_complete_analysis_pending"] = (
        "canonical_budget_curve_complete_analysis_pending"
    )
    execution_plan_hash: Sha256
    source_public_stream_sha256: Sha256
    source_public_replay_sha256: Sha256
    budget_curve_file: Literal["canonical_budget_curve_metrics.jsonl"] = _BUDGET_FILE
    budget_curve_file_sha256: Sha256
    budget_curve_replay_sha256: Sha256
    budget_curve_file_bytes: int = Field(ge=1)
    environment_count: Literal[7] = 7
    episodes_per_environment: int = Field(ge=1)
    episode_count: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    budget_count: Literal[5] = 5
    policy_count: Literal[5] = 5
    source_public_row_count: int = Field(ge=1)
    budget_curve_row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    missing_probe_result_count: int = Field(ge=0)
    primary_budget_parity_check_count: int = Field(ge=1)
    primary_budget_reproduction_verified: Literal[True] = True
    endpoint_agreement_verified: Literal[True] = True
    deterministic_replay_verified: Literal[True] = True
    primary_claim_rescue_allowed: Literal[False] = False
    restricted_stream_accessed: Literal[False] = False
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.source_public_stream_sha256 != self.source_public_replay_sha256:
            raise ValueError("Budget execution read different source bytes on retry")
        if self.budget_curve_file_sha256 != self.budget_curve_replay_sha256:
            raise ValueError("Budget-curve deterministic retry differs")
        expected_episodes = self.environment_count * self.episodes_per_environment
        if self.episode_count != expected_episodes:
            raise ValueError("Budget-curve episode count does not reconcile")
        if self.source_public_row_count != self.episode_count * len(_SOURCE_POLICY_ORDER):
            raise ValueError("Budget-curve source row count does not reconcile")
        expected_rows = self.episode_count * self.budget_count * self.policy_count
        if self.budget_curve_row_count != expected_rows:
            raise ValueError("Budget-curve output row count does not reconcile")
        if self.case_prediction_count != (
            self.budget_curve_row_count * self.candidates_per_episode
        ):
            raise ValueError("Budget-curve case count does not reconcile")
        if self.primary_budget_parity_check_count != self.episode_count * self.policy_count:
            raise ValueError("Budget-curve primary parity count does not reconcile")
        expected_selected = (
            self.episode_count
            * self.budget_count
            * self.policy_count
            * self.candidates_per_episode
            // 2
        )
        if self.selected_probe_count != expected_selected:
            raise ValueError("Budget-curve selected-probe count does not reconcile")
        if self.missing_probe_result_count > self.selected_probe_count:
            raise ValueError("Budget-curve missing outcomes exceed selected probes")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Budget-curve report hash does not match its content")
        return self


class CanonicalBudgetCurveManifest(ContractModel):
    """File inventory for one immutable canonical budget-curve execution."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_budget_curve_manifest.v1"] = (
        "acquisition_study.canonical_budget_curve_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    execution_plan_file: Literal["canonical_budget_curve_execution_plan.json"] = _PLAN_FILE
    execution_plan_file_sha256: Sha256
    execution_plan_hash: Sha256
    report_file: Literal["canonical_budget_curve_report.json"] = _REPORT_FILE
    report_file_sha256: Sha256
    report_hash: Sha256
    budget_curve_file: Literal["canonical_budget_curve_metrics.jsonl"] = _BUDGET_FILE
    budget_curve_file_sha256: Sha256
    source_public_stream_sha256: Sha256
    completion_status: Literal["canonical_budget_curve_complete_analysis_pending"] = (
        "canonical_budget_curve_complete_analysis_pending"
    )
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Budget-curve manifest hash does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class _StreamSummary:
    source_hash: Sha256
    source_rows: int
    budget_hash: Sha256
    budget_rows: int
    budget_bytes: int
    selected_probe_count: int
    missing_probe_result_count: int
    primary_parity_checks: int


class _ByteDigest(Protocol):
    def update(self, value: bytes, /) -> None: ...


def run_canonical_budget_curve(
    source_plan_path: Path,
    primary_manifest_path: Path,
    *,
    output_root: Path,
    secondary_code_revision: str,
) -> CanonicalBudgetCurveManifest:
    """Verify canonical lineage, reproduce the 50% rows, and publish the full curve."""

    source_plan = load_canonical_primary_source_plan(source_plan_path)
    project_root = source_plan_path.resolve().parents[2]
    execution = load_canonical_execution_plan(
        _project_path(project_root, source_plan.canonical_execution_plan_path)
    )
    specification, analysis = load_acquisition_study_plan(
        _project_path(project_root, execution.environment_specification_path),
        _project_path(project_root, source_plan.analysis_specification_path),
    )
    primary_manifest = _load_model(primary_manifest_path, CanonicalPrimaryAnalysisManifest)
    primary_report_path = primary_manifest_path.parent / primary_manifest.report_file
    primary_report = _load_model(primary_report_path, CanonicalPrimaryAnalysisReport)

    _validate_lineage(
        source_plan,
        execution_hash=execution.plan_hash,
        specification=specification,
        analysis=analysis,
        pixi_lock_hash=_hash_path(
            _project_path(project_root, source_plan.pixi_lock_path),
            "Pixi lock",
        ),
        primary_manifest=primary_manifest,
        primary_report=primary_report,
        primary_report_file_sha256=_hash_path(primary_report_path, "primary report"),
    )
    episode_indices = acquisition_episode_indices(
        specification,
        partition=AcquisitionStudyPartition.EVALUATION,
    )
    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_budget_curve_execution_plan.v1",
        "study_id": source_plan.study_id,
        "split": "evaluation",
        "source_plan_hash": source_plan.plan_hash,
        "source_plan_file_sha256": _hash_path(source_plan_path, "source plan"),
        "canonical_code_revision": source_plan.canonical_code_revision,
        "secondary_code_revision": secondary_code_revision,
        "canonical_run_manifest_hash": source_plan.canonical_run_manifest_hash,
        "canonical_stream_report_hash": source_plan.canonical_stream_report_hash,
        "source_public_stream_sha256": source_plan.public_stream_sha256,
        "primary_analysis_manifest_hash": primary_manifest.manifest_hash,
        "primary_analysis_manifest_file_sha256": _hash_path(
            primary_manifest_path, "primary manifest"
        ),
        "primary_analysis_report_hash": primary_report.report_hash,
        "primary_analysis_report_file_sha256": _hash_path(primary_report_path, "primary report"),
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "pixi_lock_sha256": source_plan.pixi_lock_sha256,
        "environment_ids": tuple(EvaluationEnvironmentId),
        "source_policy_ids": _SOURCE_POLICY_ORDER,
        "budget_policy_ids": BUDGET_CURVE_POLICY_ORDER,
        "budget_fractions": analysis.secondary.budget_fractions,
        "evaluation_episode_index_start": episode_indices[0],
        "evaluation_episode_index_stop_exclusive": episode_indices[-1] + 1,
        "episodes_per_environment": len(episode_indices),
        "candidates_per_episode": specification.episodes.candidates_per_episode,
        "source_public_row_count": source_plan.expected_public_row_count,
        "expected_budget_row_count": (
            len(EvaluationEnvironmentId)
            * len(episode_indices)
            * len(analysis.secondary.budget_fractions)
            * len(BUDGET_CURVE_POLICY_ORDER)
        ),
        "primary_result_known_before_secondary_execution": True,
        "budget_fractions_and_metrics_frozen_before_primary_result": True,
        "post_primary_policy_tuning_allowed": False,
        "secondary_analysis_role": "descriptive_cannot_rescue_primary",
        "restricted_stream_access_allowed": False,
        "external_calls_allowed": False,
    }
    plan = CanonicalBudgetCurveExecutionPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    first, replay = write_verified_canonical_budget_curve_stream(
        specification,
        analysis,
        source_public_path=_project_path(project_root, source_plan.public_stream_path),
        expected_source_hash=source_plan.public_stream_sha256,
        episode_indices=episode_indices,
        output_path=output_root / _BUDGET_FILE,
    )
    if first != replay:
        raise CanonicalBudgetCurveError("Canonical budget-curve retry differs")
    report = build_canonical_budget_curve_report(
        first,
        replay,
        execution_plan_hash=plan.plan_hash,
        episodes_per_environment=len(episode_indices),
        candidates_per_episode=specification.episodes.candidates_per_episode,
    )
    manifest_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_budget_curve_manifest.v1",
        "study_id": source_plan.study_id,
        "execution_plan_file": _PLAN_FILE,
        "execution_plan_file_sha256": file_sha256(immutable_json_bytes(plan)),
        "execution_plan_hash": plan.plan_hash,
        "report_file": _REPORT_FILE,
        "report_file_sha256": file_sha256(immutable_json_bytes(report)),
        "report_hash": report.report_hash,
        "budget_curve_file": _BUDGET_FILE,
        "budget_curve_file_sha256": first.budget_hash,
        "source_public_stream_sha256": first.source_hash,
        "completion_status": "canonical_budget_curve_complete_analysis_pending",
    }
    manifest = CanonicalBudgetCurveManifest.model_validate(
        {**manifest_content, "manifest_hash": canonical_sha256(manifest_content)}
    )
    write_immutable_json(output_root / _PLAN_FILE, plan)
    write_immutable_json(output_root / _REPORT_FILE, report)
    write_immutable_json(output_root / _MANIFEST_FILE, manifest)
    return manifest


def build_canonical_budget_curve_report(
    first: _StreamSummary,
    replay: _StreamSummary,
    *,
    execution_plan_hash: Sha256,
    episodes_per_environment: int,
    candidates_per_episode: int,
) -> CanonicalBudgetCurveReport:
    """Build the reconciled report after the stream and retry agree."""

    if first != replay:
        raise CanonicalBudgetCurveError("Canonical budget-curve retry differs")
    report_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_budget_curve_report.v1",
        "study_id": "reliability-aware-probing-v1",
        "split": "evaluation",
        "result_status": "canonical_budget_curve_complete_analysis_pending",
        "execution_plan_hash": execution_plan_hash,
        "source_public_stream_sha256": first.source_hash,
        "source_public_replay_sha256": replay.source_hash,
        "budget_curve_file": _BUDGET_FILE,
        "budget_curve_file_sha256": first.budget_hash,
        "budget_curve_replay_sha256": replay.budget_hash,
        "budget_curve_file_bytes": first.budget_bytes,
        "environment_count": len(EvaluationEnvironmentId),
        "episodes_per_environment": episodes_per_environment,
        "episode_count": len(EvaluationEnvironmentId) * episodes_per_environment,
        "candidates_per_episode": candidates_per_episode,
        "budget_count": 5,
        "policy_count": len(BUDGET_CURVE_POLICY_ORDER),
        "source_public_row_count": first.source_rows,
        "budget_curve_row_count": first.budget_rows,
        "case_prediction_count": first.budget_rows * candidates_per_episode,
        "selected_probe_count": first.selected_probe_count,
        "missing_probe_result_count": first.missing_probe_result_count,
        "primary_budget_parity_check_count": first.primary_parity_checks,
        "primary_budget_reproduction_verified": True,
        "endpoint_agreement_verified": True,
        "deterministic_replay_verified": True,
        "primary_claim_rescue_allowed": False,
        "restricted_stream_accessed": False,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
    }
    return CanonicalBudgetCurveReport.model_validate(
        {**report_content, "report_hash": canonical_sha256(report_content)}
    )


def write_verified_canonical_budget_curve_stream(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    source_public_path: Path,
    expected_source_hash: Sha256,
    episode_indices: tuple[int, ...],
    output_path: Path,
) -> tuple[_StreamSummary, _StreamSummary]:
    """Write the canonical curve and require exact source and replay agreement."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        replay = _consume_budget_stream(
            specification,
            analysis,
            source_public_path=source_public_path,
            expected_source_hash=expected_source_hash,
            episode_indices=episode_indices,
        )
        existing_hash, existing_bytes = _hash_file(output_path)
        if existing_hash != replay.budget_hash or existing_bytes != replay.budget_bytes:
            raise ArtifactConflictError(f"Immutable artifact already differs: {output_path}")
        return replay, replay
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output_handle:
            first = _consume_budget_stream(
                specification,
                analysis,
                source_public_path=source_public_path,
                expected_source_hash=expected_source_hash,
                episode_indices=episode_indices,
                output_handle=output_handle,
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        replay = _consume_budget_stream(
            specification,
            analysis,
            source_public_path=source_public_path,
            expected_source_hash=expected_source_hash,
            episode_indices=episode_indices,
        )
        if first != replay:
            raise CanonicalBudgetCurveError("Canonical budget-curve retry differs")
        if output_path.exists() and not files_equal(output_path, temporary):
            raise ArtifactConflictError(f"Immutable artifact already differs: {output_path}")
        if not output_path.exists():
            os.replace(temporary, output_path)
            fsync_directory(output_path.parent)
        return first, replay
    finally:
        temporary.unlink(missing_ok=True)


def _consume_budget_stream(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    source_public_path: Path,
    expected_source_hash: Sha256,
    episode_indices: tuple[int, ...],
    output_handle: BinaryIO | None = None,
) -> _StreamSummary:
    calibration = estimate_probe_reliability(specification)
    source_digest = hashlib.sha256()
    budget_digest = hashlib.sha256()
    source_rows = budget_rows = budget_bytes = 0
    selected_probe_count = missing_probe_result_count = primary_parity_checks = 0
    try:
        with source_public_path.open("rb") as source_handle:
            for environment_id in EvaluationEnvironmentId:
                for episode_index in episode_indices:
                    source_by_policy, raw_count = _read_source_episode(
                        source_handle,
                        source_digest,
                        specification=specification,
                        analysis=analysis,
                        calibration_hash=calibration.calibration_hash,
                        environment_id=environment_id,
                        episode_index=episode_index,
                    )
                    source_rows += raw_count
                    curve = run_episode_budget_curve(
                        specification,
                        analysis,
                        calibration,
                        environment_id=environment_id,
                        episode_index=episode_index,
                    )
                    if (
                        curve.shared_candidate_hash
                        != next(iter(source_by_policy.values())).shared_candidate_hash
                    ):
                        raise CanonicalBudgetCurveError(
                            "Regenerated budget episode differs from the sealed source"
                        )
                    primary_parity_checks += _verify_primary_budget(
                        curve.results,
                        source_by_policy,
                        ece_bin_count=analysis.secondary.ece_bin_count,
                    )
                    for result in curve.results:
                        metric = compact_budget_policy_result(
                            result,
                            analysis.secondary.ece_bin_count,
                        )
                        record = CanonicalBudgetCurveRecord(
                            shared_candidate_hash=curve.shared_candidate_hash,
                            policy_result_hash=model_content_hash(result.result),
                            metric=metric,
                        )
                        content = canonical_json_bytes(record) + b"\n"
                        if output_handle is not None:
                            output_handle.write(content)
                        budget_digest.update(content)
                        budget_rows += 1
                        budget_bytes += len(content)
                        selected_probe_count += metric.selected_probe_count
                        missing_probe_result_count += metric.missing_probe_result_count
            if source_handle.read(1):
                raise CanonicalBudgetCurveError("Sealed public stream contains extra rows")
    except CanonicalBudgetCurveError:
        raise
    except (OSError, ValidationError) as error:
        raise CanonicalBudgetCurveError(
            f"Could not verify canonical public stream: {source_public_path}"
        ) from error
    source_hash = source_digest.hexdigest()
    if source_hash != expected_source_hash:
        raise CanonicalBudgetCurveError("Sealed public stream hash differs")
    return _StreamSummary(
        source_hash=source_hash,
        source_rows=source_rows,
        budget_hash=budget_digest.hexdigest(),
        budget_rows=budget_rows,
        budget_bytes=budget_bytes,
        selected_probe_count=selected_probe_count,
        missing_probe_result_count=missing_probe_result_count,
        primary_parity_checks=primary_parity_checks,
    )


def _read_source_episode(
    handle: BinaryIO,
    digest: _ByteDigest,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    calibration_hash: Sha256,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
) -> tuple[dict[PolicyId, CanonicalPublicPolicyMetric], int]:
    rows: dict[PolicyId, CanonicalPublicPolicyMetric] = {}
    shared_candidate_hash: Sha256 | None = None
    for policy_id in _SOURCE_POLICY_ORDER:
        raw_line = handle.readline()
        if not raw_line:
            raise CanonicalBudgetCurveError("Sealed public stream ended early")
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            raise CanonicalBudgetCurveError("Sealed public stream contains invalid JSONL")
        digest.update(raw_line)
        row = CanonicalPublicPolicyMetric.model_validate_json(raw_line)
        metric = row.metric
        if (
            metric.environment_id is not environment_id
            or metric.episode_index != episode_index
            or metric.policy_id is not policy_id
        ):
            raise CanonicalBudgetCurveError("Sealed public stream order differs")
        if (
            row.environment_specification_hash != specification.specification_hash
            or row.analysis_plan_hash != analysis.plan_hash
            or row.calibration_hash != calibration_hash
            or metric.candidate_count != specification.episodes.candidates_per_episode
        ):
            raise CanonicalBudgetCurveError("Sealed public row uses another frozen input")
        if shared_candidate_hash is None:
            shared_candidate_hash = row.shared_candidate_hash
        elif row.shared_candidate_hash != shared_candidate_hash:
            raise CanonicalBudgetCurveError("Sealed public rows break episode pairing")
        rows[policy_id] = row
    return rows, len(_SOURCE_POLICY_ORDER)


def _verify_primary_budget(
    curve_results: tuple[BudgetPolicyResult, ...],
    source_by_policy: dict[PolicyId, CanonicalPublicPolicyMetric],
    *,
    ece_bin_count: int,
) -> int:
    primary_rows = tuple(
        row for row in curve_results if getattr(row, "budget_fraction", None) == 0.5
    )
    if tuple(row.result.policy_id for row in primary_rows) != BUDGET_CURVE_POLICY_ORDER:
        raise CanonicalBudgetCurveError("Regenerated primary policy order differs")
    for row in primary_rows:
        source = source_by_policy[row.result.policy_id]
        metric = compact_budget_policy_result(row, ece_bin_count)
        if model_content_hash(row.result) != source.policy_result_hash or metric != source.metric:
            raise CanonicalBudgetCurveError(
                "Regenerated 50% result differs from the sealed primary stream"
            )
    return len(primary_rows)


def _validate_lineage(
    source_plan: CanonicalPrimarySourcePlan,
    *,
    execution_hash: Sha256,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_hash: Sha256,
    primary_manifest: CanonicalPrimaryAnalysisManifest,
    primary_report: CanonicalPrimaryAnalysisReport,
    primary_report_file_sha256: Sha256,
) -> None:
    if execution_hash != source_plan.canonical_execution_plan_hash:
        raise CanonicalBudgetCurveError("Canonical execution plan differs")
    if specification.specification_hash != analysis.environment_specification_hash:
        raise CanonicalBudgetCurveError("Environment and analysis specifications differ")
    if analysis.plan_hash != source_plan.analysis_specification_hash:
        raise CanonicalBudgetCurveError("Frozen analysis specification differs")
    if pixi_lock_hash != source_plan.pixi_lock_sha256:
        raise CanonicalBudgetCurveError("Pixi lock differs from the sealed source")
    if (
        primary_manifest.source_plan_hash != source_plan.plan_hash
        or primary_manifest.public_stream_sha256 != source_plan.public_stream_sha256
        or primary_report.source_plan_hash != source_plan.plan_hash
        or primary_report.public_stream_sha256 != source_plan.public_stream_sha256
        or primary_report.report_hash != primary_manifest.report_hash
        or primary_report_file_sha256 != primary_manifest.report_file_sha256
    ):
        raise CanonicalBudgetCurveError("Primary analysis does not belong to the sealed run")
    if (
        not primary_manifest.exact_analysis_retry_verified
        or primary_report.result_status != "canonical_primary_complete_secondary_pending"
    ):
        raise CanonicalBudgetCurveError("Primary analysis status differs from the observed run")


def _load_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CanonicalBudgetCurveError(f"Could not load canonical source: {path}") from error


def _hash_path(path: Path, label: str) -> Sha256:
    try:
        return file_sha256(path.read_bytes())
    except OSError as error:
        raise CanonicalBudgetCurveError(f"Could not read {label}: {path}") from error


def _project_path(project_root: Path, relative: str) -> Path:
    path = (project_root / relative).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise CanonicalBudgetCurveError(
            f"Canonical source escapes project root: {relative}"
        ) from error
    return path


def _hash_file(path: Path) -> tuple[Sha256, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise CanonicalBudgetCurveError(f"Could not verify existing output: {path}") from error
    return digest.hexdigest(), byte_count
