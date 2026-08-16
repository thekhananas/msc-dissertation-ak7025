"""Paired development runner for the shuffled-reliability negative control."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.budget_curve import BudgetPolicyResult
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    compact_budget_policy_result,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import (
    AcquisitionCandidate,
    PolicyId,
    ProbeClassId,
)
from socratic_tutor.acquisition_study.evaluation import (
    PolicyEpisodeResult,
    evaluate_policy_episode,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.policies import (
    FIXED_RELIABILITY_SHUFFLE,
    ReliabilityAwareEVSIPolicy,
    ShuffledReliabilityEVSIPolicy,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    load_verified_development_policy_matrix,
)
from socratic_tutor.acquisition_study.runner import EpisodePolicyComparison
from socratic_tutor.acquisition_study.simulation import (
    PrivilegedEpisodeRecord,
    generate_acquisition_episode,
)
from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_CONTROL = PolicyId.SHUFFLED_RELIABILITY_BOUNDED
_CONTROL_ID = "shuffle_probe_class_reliability_summaries"


class DevelopmentNegativeControlError(ValueError):
    """A negative-control source or result violates the frozen design."""


class NegativeControlEpisodeMetric(ContractModel):
    """Candidate and shuffled-control results on one shared episode."""

    environment_id: EvaluationEnvironmentId
    episode_index: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    candidate: BudgetCurveEpisodeMetric
    shuffled_control: BudgetCurveEpisodeMetric

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.episode_id != f"episode-{self.episode_index:06d}":
            raise ValueError("Negative-control episode identifier does not match its index")
        rows = (self.candidate, self.shuffled_control)
        if tuple(row.policy_id for row in rows) != (_CANDIDATE, _CONTROL):
            raise ValueError("Negative-control pair uses the wrong policies")
        if any(
            row.environment_id is not self.environment_id
            or row.episode_index != self.episode_index
            or row.episode_id != self.episode_id
            for row in rows
        ):
            raise ValueError("Negative-control policies do not share one episode")
        if (
            self.candidate.candidate_count != self.shuffled_control.candidate_count
            or self.candidate.selected_probe_count != self.shuffled_control.selected_probe_count
            or not math.isclose(
                self.candidate.budget_fraction,
                self.shuffled_control.budget_fraction,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Negative-control policies do not share one probe budget")
        return self


class DevelopmentNegativeControlMatrix(ContractModel):
    """Paired negative-control metrics over every development episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_negative_control_matrix.v1"] = (
        "acquisition_study.development_negative_control_matrix.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    negative_control_id: Literal["shuffle_probe_class_reliability_summaries"] = _CONTROL_ID
    reliability_shuffle: tuple[tuple[ProbeClassId, ProbeClassId], ...] = Field(
        min_length=4,
        max_length=4,
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
    row_count: int = Field(ge=1)
    rows: tuple[NegativeControlEpisodeMetric, ...]
    candidate_source_reproduction_verified: Literal[True] = True
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.reliability_shuffle != FIXED_RELIABILITY_SHUFFLE:
            raise ValueError("Negative-control reliability shuffle differs from the frozen mapping")
        if self.environment_count != len(EvaluationEnvironmentId):
            raise ValueError("Negative-control matrix does not contain every environment")
        if not math.isclose(self.fixed_budget_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("Negative-control matrix differs from the primary budget fraction")
        if self.fixed_probe_budget != self.candidates_per_episode * self.fixed_budget_fraction:
            raise ValueError("Negative-control probe count differs from its budget fraction")
        expected_count = self.environment_count * self.episodes_per_environment
        if self.row_count != expected_count or len(self.rows) != expected_count:
            raise ValueError("Negative-control row count is inconsistent")
        expected_keys = tuple(
            (environment_id, episode_index)
            for environment_id in EvaluationEnvironmentId
            for episode_index in range(self.episodes_per_environment)
        )
        actual_keys = tuple((row.environment_id, row.episode_index) for row in self.rows)
        if actual_keys != expected_keys:
            raise ValueError("Negative-control matrix order or coverage differs")
        if any(
            row.candidate.candidate_count != self.candidates_per_episode
            or row.candidate.selected_probe_count != self.fixed_probe_budget
            for row in self.rows
        ):
            raise ValueError("Negative-control rows use another episode shape")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


class DevelopmentNegativeControlManifest(ContractModel):
    """Self-verifying record of one published development negative control."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_negative_control_manifest.v1"] = (
        "acquisition_study.development_negative_control_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    negative_control_id: Literal["shuffle_probe_class_reliability_summaries"] = _CONTROL_ID
    reliability_shuffle_hash: Sha256
    source_manifest_hash: Sha256
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    pixi_lock_sha256: Sha256
    result_file: Literal["development_negative_control.jsonl"] = (
        "development_negative_control.jsonl"
    )
    result_file_sha256: Sha256
    matrix_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    candidate_source_reproduction_verified: Literal[True] = True
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    environment_count: Literal[7] = 7
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
        if self.reliability_shuffle_hash != canonical_sha256(FIXED_RELIABILITY_SHUFFLE):
            raise ValueError("Negative-control manifest uses another reliability shuffle")
        if self.row_count != self.environment_count * self.episodes_per_environment:
            raise ValueError("Negative-control manifest row count is inconsistent")
        if self.case_prediction_count != self.row_count * self.candidates_per_episode * 2:
            raise ValueError("Negative-control manifest case count is inconsistent")
        if self.selected_probe_count != self.row_count * self.fixed_probe_budget * 2:
            raise ValueError("Negative-control manifest probe count is inconsistent")
        if self.matrix_content_hash != self.replay_content_hash:
            raise ValueError("Negative-control replay hash differs")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Negative-control manifest hash does not match its content")
        return self


def run_verified_development_negative_control(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentNegativeControlMatrix, Sha256]:
    """Run the development negative control twice and require exact agreement."""

    first = run_development_negative_control(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    replay = run_development_negative_control(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    if first.content_hash != replay.content_hash:
        raise DevelopmentNegativeControlError("Development negative-control replay differs")
    return first, replay.content_hash


def run_development_negative_control(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> DevelopmentNegativeControlMatrix:
    """Evaluate the candidate and its shuffled-reliability control on paired episodes."""

    source_manifest, source_matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    calibration = estimate_probe_reliability(specification)
    if calibration.calibration_hash != source_manifest.calibration_hash:
        raise DevelopmentNegativeControlError("Negative control uses another calibration result")
    if analysis.negative_control != _CONTROL_ID:
        raise DevelopmentNegativeControlError("Analysis specifies another negative control")

    candidate_policy = ReliabilityAwareEVSIPolicy(
        lower_quantile=analysis.policies.conservative_reliability_quantile,
        kappa=analysis.policies.bounded_log_likelihood_kappa,
        draw_count=analysis.policies.reliability_draw_count,
        seed=analysis.policies.reliability_draw_seed,
    )
    shuffled_policy = ShuffledReliabilityEVSIPolicy(
        lower_quantile=analysis.policies.conservative_reliability_quantile,
        kappa=analysis.policies.bounded_log_likelihood_kappa,
        draw_count=analysis.policies.reliability_draw_count,
        seed=analysis.policies.reliability_draw_seed,
    )
    rows: list[NegativeControlEpisodeMetric] = []
    for source in source_matrix.comparisons:
        policy_episode, privileged_episode = generate_acquisition_episode(
            specification,
            calibration,
            environment_id=source.environment_id,
            episode_index=source.episode_index,
            probe_budget=analysis.primary.exact_selected_probes_per_episode,
        )
        _verify_regenerated_episode(policy_episode.request.candidates, privileged_episode, source)
        candidate = evaluate_policy_episode(
            candidate_policy,
            policy_episode,
            privileged_episode,
            kappa=analysis.policies.bounded_log_likelihood_kappa,
            classification_threshold=analysis.primary.classification_threshold,
            probability_floor=analysis.secondary.probability_floor,
        )
        source_candidate = next(
            result for result in source.results if result.policy_id is _CANDIDATE
        )
        if candidate != source_candidate:
            raise DevelopmentNegativeControlError(
                "Negative-control candidate does not reproduce its source result"
            )
        shuffled = evaluate_policy_episode(
            shuffled_policy,
            policy_episode,
            privileged_episode,
            kappa=analysis.policies.bounded_log_likelihood_kappa,
            classification_threshold=analysis.primary.classification_threshold,
            probability_floor=analysis.secondary.probability_floor,
        )
        rows.append(
            NegativeControlEpisodeMetric(
                environment_id=source.environment_id,
                episode_index=source.episode_index,
                episode_id=source.episode_id,
                candidate=_compact(candidate, analysis),
                shuffled_control=_compact(shuffled, analysis),
            )
        )

    return DevelopmentNegativeControlMatrix(
        reliability_shuffle=FIXED_RELIABILITY_SHUFFLE,
        source_manifest_hash=source_manifest.manifest_hash,
        environment_specification_hash=specification.specification_hash,
        analysis_plan_hash=analysis.plan_hash,
        calibration_hash=calibration.calibration_hash,
        episodes_per_environment=source_matrix.episodes_per_environment,
        candidates_per_episode=specification.episodes.candidates_per_episode,
        fixed_probe_budget=analysis.primary.exact_selected_probes_per_episode,
        fixed_budget_fraction=analysis.primary.primary_budget_fraction,
        row_count=len(rows),
        rows=tuple(rows),
    )


def publish_development_negative_control(
    matrix: DevelopmentNegativeControlMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
    pixi_lock_path: Path,
) -> DevelopmentNegativeControlManifest:
    """Write compact negative-control rows and their immutable manifest."""

    if matrix.content_hash != replay_content_hash:
        raise DevelopmentNegativeControlError("Negative-control matrix and replay hashes differ")
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in matrix.rows)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise DevelopmentNegativeControlError(
            f"Could not read Pixi lock: {pixi_lock_path}"
        ) from error
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_negative_control_manifest.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "code_revision": code_revision,
        "split": matrix.split,
        "result_scope": matrix.result_scope,
        "access_scope": matrix.access_scope,
        "negative_control_id": matrix.negative_control_id,
        "reliability_shuffle_hash": canonical_sha256(matrix.reliability_shuffle),
        "source_manifest_hash": matrix.source_manifest_hash,
        "environment_specification_hash": matrix.environment_specification_hash,
        "analysis_plan_hash": matrix.analysis_plan_hash,
        "calibration_hash": matrix.calibration_hash,
        "pixi_lock_sha256": pixi_lock_hash,
        "result_file": "development_negative_control.jsonl",
        "result_file_sha256": file_sha256(rows_bytes),
        "matrix_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "deterministic_replay_verified": True,
        "candidate_source_reproduction_verified": (matrix.candidate_source_reproduction_verified),
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "fixed_probe_budget": matrix.fixed_probe_budget,
        "environment_count": matrix.environment_count,
        "row_count": matrix.row_count,
        "case_prediction_count": matrix.row_count * matrix.candidates_per_episode * 2,
        "selected_probe_count": matrix.row_count * matrix.fixed_probe_budget * 2,
        "external_model_call_count": matrix.external_model_call_count,
        "sandbox_call_count": matrix.sandbox_call_count,
        "human_record_count": matrix.human_record_count,
        "canonical_claim_allowed": matrix.canonical_claim_allowed,
    }
    manifest = DevelopmentNegativeControlManifest.model_validate(
        {**content, "manifest_hash": canonical_sha256(content)}
    )
    write_immutable_bytes(output_root / manifest.result_file, rows_bytes)
    write_immutable_json(output_root / "development_negative_control_manifest.json", manifest)
    return manifest


def load_verified_development_negative_control(
    manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentNegativeControlManifest, DevelopmentNegativeControlMatrix]:
    """Load negative-control rows only after checking their complete manifest."""

    try:
        manifest = DevelopmentNegativeControlManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        result_path = manifest_path.parent / manifest.result_file
        result_bytes = result_path.read_bytes()
        if file_sha256(result_bytes) != manifest.result_file_sha256:
            raise DevelopmentNegativeControlError("Development negative-control file hash differs")
        lines = result_bytes.decode("utf-8").splitlines()
        if len(lines) != manifest.row_count or any(not line for line in lines):
            raise DevelopmentNegativeControlError("Development negative-control row count differs")
        rows = tuple(
            NegativeControlEpisodeMetric.model_validate(json.loads(line)) for line in lines
        )
        matrix = DevelopmentNegativeControlMatrix(
            reliability_shuffle=FIXED_RELIABILITY_SHUFFLE,
            source_manifest_hash=manifest.source_manifest_hash,
            environment_specification_hash=manifest.environment_specification_hash,
            analysis_plan_hash=manifest.analysis_plan_hash,
            calibration_hash=manifest.calibration_hash,
            episodes_per_environment=manifest.episodes_per_environment,
            candidates_per_episode=manifest.candidates_per_episode,
            fixed_probe_budget=manifest.fixed_probe_budget,
            fixed_budget_fraction=analysis.primary.primary_budget_fraction,
            environment_count=manifest.environment_count,
            row_count=manifest.row_count,
            rows=rows,
        )
    except DevelopmentNegativeControlError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise DevelopmentNegativeControlError(
            f"Could not verify development negative control: {manifest_path}"
        ) from error
    if manifest.environment_specification_hash != specification.specification_hash:
        raise DevelopmentNegativeControlError(
            "Negative control uses another environment specification"
        )
    if manifest.analysis_plan_hash != analysis.plan_hash:
        raise DevelopmentNegativeControlError("Negative control uses another frozen analysis plan")
    if matrix.content_hash != manifest.matrix_content_hash:
        raise DevelopmentNegativeControlError("Development negative-control content hash differs")
    return manifest, matrix


def _compact(
    result: PolicyEpisodeResult,
    analysis: AcquisitionAnalysisSpecification,
) -> BudgetCurveEpisodeMetric:
    return compact_budget_policy_result(
        BudgetPolicyResult(
            budget_fraction=analysis.primary.primary_budget_fraction,
            selected_probe_count=result.burden.selected_probe_count,
            result=result,
        ),
        analysis.secondary.ece_bin_count,
    )


def _verify_regenerated_episode(
    candidates: tuple[AcquisitionCandidate, ...],
    privileged_episode: PrivilegedEpisodeRecord,
    source: EpisodePolicyComparison,
) -> None:
    if (
        canonical_sha256(candidates) != source.shared_candidate_hash
        or model_content_hash(privileged_episode) != source.privileged_episode_hash
    ):
        raise DevelopmentNegativeControlError(
            "Regenerated negative-control episode differs from its source"
        )
