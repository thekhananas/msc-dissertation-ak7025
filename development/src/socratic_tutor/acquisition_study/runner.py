"""Deterministic development runner for paired acquisition-policy episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.calibration import (
    ReliabilityCalibrationRun,
    estimate_probe_reliability,
)
from socratic_tutor.acquisition_study.contracts import (
    AcquisitionPolicy,
    AcquisitionRequest,
    PolicyId,
)
from socratic_tutor.acquisition_study.evaluation import (
    PolicyEpisodeResult,
    build_endpoint_policies,
    build_oracle_reference_policy,
    build_primary_non_oracle_policies,
    evaluate_policy_episode,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.simulation import (
    PolicyEpisodeRecord,
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

_POLICY_ORDER = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
    PolicyId.NEVER_PROBE,
    PolicyId.ALWAYS_PROBE_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)


@dataclass(frozen=True, slots=True)
class EpisodePolicyRun:
    """Safe inputs, simulator truth, and policy results for one episode."""

    policy_episode: PolicyEpisodeRecord
    privileged_episode: PrivilegedEpisodeRecord
    comparison: EpisodePolicyComparison


class EpisodePolicyComparison(ContractModel):
    """All policies scored against one underlying simulator episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.episode_policy_comparison.v1"] = (
        "acquisition_study.episode_policy_comparison.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    environment_id: EvaluationEnvironmentId
    episode_index: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    shared_candidate_hash: Sha256
    privileged_episode_hash: Sha256
    candidate_count: int = Field(ge=1)
    matched_budget: int = Field(ge=1)
    results: tuple[PolicyEpisodeResult, ...] = Field(min_length=7, max_length=7)
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.episode_id != f"episode-{self.episode_index:06d}":
            raise ValueError("Episode identifier does not match its index")
        if tuple(result.policy_id for result in self.results) != _POLICY_ORDER:
            raise ValueError("Policy comparison order differs from the frozen design")
        if any(
            result.episode_id != self.episode_id or result.environment_id is not self.environment_id
            for result in self.results
        ):
            raise ValueError("Policy results do not share one environment episode")
        if any(result.burden.candidate_count != self.candidate_count for result in self.results):
            raise ValueError("Policy results do not share one candidate set")

        expected_budgets = {
            PolicyId.RELIABILITY_AWARE_BOUNDED: self.matched_budget,
            PolicyId.SEEDED_RANDOM_BOUNDED: self.matched_budget,
            PolicyId.UNCERTAINTY_ONLY_BOUNDED: self.matched_budget,
            PolicyId.PLUG_IN_EVSI_BOUNDED: self.matched_budget,
            PolicyId.NEVER_PROBE: 0,
            PolicyId.ALWAYS_PROBE_BOUNDED: self.candidate_count,
            PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED: self.matched_budget,
        }
        if any(
            result.burden.selected_probe_count != expected_budgets[result.policy_id]
            for result in self.results
        ):
            raise ValueError("A policy result uses the wrong probe budget")
        if any(
            result.burden.external_workload.model_request_count != 0
            or result.burden.external_workload.sandbox_execution_count != 0
            for result in self.results
        ):
            raise ValueError("Glass-box policy results cannot contain external work")

        reference_cases = tuple(
            (
                case.case_id,
                case.prior_success_probability,
                case.latent_success,
            )
            for case in self.results[0].case_results
        )
        observed_by_case: dict[str, object] = {}
        for result in self.results:
            case_projection = tuple(
                (
                    case.case_id,
                    case.prior_success_probability,
                    case.latent_success,
                )
                for case in result.case_results
            )
            if case_projection != reference_cases:
                raise ValueError("Policies were not scored on identical cases")
            for case in result.case_results:
                if case.observed_probe_outcome is None:
                    continue
                previous = observed_by_case.setdefault(case.case_id, case.observed_probe_outcome)
                if previous != case.observed_probe_outcome:
                    raise ValueError("Policies received different outcomes for the same probe")
        return self


class DevelopmentPolicyMatrix(ContractModel):
    """Non-canonical development comparisons across every frozen environment."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_policy_matrix.v1"] = (
        "acquisition_study.development_policy_matrix.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    episodes_per_environment: int = Field(ge=1)
    environment_count: int = Field(ge=1)
    comparison_count: int = Field(ge=1)
    policy_count: int = Field(ge=1)
    comparisons: tuple[EpisodePolicyComparison, ...]
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.environment_count != len(EvaluationEnvironmentId):
            raise ValueError("Development matrix does not contain every environment")
        if self.policy_count != len(_POLICY_ORDER):
            raise ValueError("Development matrix does not contain every policy")
        expected_count = self.environment_count * self.episodes_per_environment
        if self.comparison_count != expected_count or len(self.comparisons) != expected_count:
            raise ValueError("Development comparison count is inconsistent")
        expected_keys = tuple(
            (environment_id, episode_index)
            for environment_id in EvaluationEnvironmentId
            for episode_index in range(self.episodes_per_environment)
        )
        actual_keys = tuple(
            (comparison.environment_id, comparison.episode_index) for comparison in self.comparisons
        )
        if actual_keys != expected_keys:
            raise ValueError("Development matrix order or coverage differs from the frozen design")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


class DevelopmentPolicyMatrixManifest(ContractModel):
    """Self-verifying record of one published development rehearsal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_policy_matrix_manifest.v1"] = (
        "acquisition_study.development_policy_matrix_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    pixi_lock_sha256: Sha256
    comparison_file: Literal["development_comparisons.jsonl"] = "development_comparisons.jsonl"
    comparison_file_sha256: Sha256
    matrix_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    environment_count: int = Field(ge=1)
    episodes_per_environment: int = Field(ge=1)
    comparison_count: int = Field(ge=1)
    policy_count: int = Field(ge=1)
    policy_result_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    missing_probe_result_count: int = Field(ge=0)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.matrix_content_hash != self.replay_content_hash:
            raise ValueError("Development replay hash differs from the first run")
        if self.environment_count != len(EvaluationEnvironmentId):
            raise ValueError("Development manifest does not contain every environment")
        if self.policy_count != len(_POLICY_ORDER):
            raise ValueError("Development manifest does not contain every policy")
        if self.comparison_count != self.environment_count * self.episodes_per_environment:
            raise ValueError("Development manifest comparison count is inconsistent")
        if self.policy_result_count != self.comparison_count * self.policy_count:
            raise ValueError("Development manifest policy-result count is inconsistent")
        expected_hash = model_content_hash(self, exclude={"manifest_hash"})
        if self.manifest_hash != expected_hash:
            raise ValueError("Development policy manifest hash does not match its content")
        return self


def run_verified_development_policy_matrix(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    episodes_per_environment: int | None = None,
) -> tuple[DevelopmentPolicyMatrix, Sha256]:
    """Run the development matrix twice and require exact agreement."""

    first = run_development_policy_matrix(
        specification,
        analysis,
        episodes_per_environment=episodes_per_environment,
    )
    replay = run_development_policy_matrix(
        specification,
        analysis,
        episodes_per_environment=episodes_per_environment,
    )
    if first.content_hash != replay.content_hash:
        raise RuntimeError("Development policy replay produced different content")
    return first, replay.content_hash


def publish_development_policy_matrix(
    matrix: DevelopmentPolicyMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
    pixi_lock_path: Path,
) -> DevelopmentPolicyMatrixManifest:
    """Write restricted development comparisons and an immutable manifest."""

    if matrix.content_hash != replay_content_hash:
        raise ValueError("Development matrix and replay hashes differ")
    comparison_bytes = b"".join(
        canonical_json_bytes(comparison) + b"\n" for comparison in matrix.comparisons
    )
    policy_results = tuple(
        result for comparison in matrix.comparisons for result in comparison.results
    )
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_policy_matrix_manifest.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "code_revision": code_revision,
        "split": matrix.split,
        "result_scope": matrix.result_scope,
        "access_scope": matrix.access_scope,
        "environment_specification_hash": matrix.environment_specification_hash,
        "analysis_plan_hash": matrix.analysis_plan_hash,
        "calibration_hash": matrix.calibration_hash,
        "pixi_lock_sha256": file_sha256(pixi_lock_path.read_bytes()),
        "comparison_file": "development_comparisons.jsonl",
        "comparison_file_sha256": file_sha256(comparison_bytes),
        "matrix_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "deterministic_replay_verified": True,
        "environment_count": matrix.environment_count,
        "episodes_per_environment": matrix.episodes_per_environment,
        "comparison_count": matrix.comparison_count,
        "policy_count": matrix.policy_count,
        "policy_result_count": len(policy_results),
        "case_prediction_count": sum(len(result.case_results) for result in policy_results),
        "selected_probe_count": sum(
            result.burden.selected_probe_count for result in policy_results
        ),
        "missing_probe_result_count": sum(
            result.burden.missing_probe_result_count for result in policy_results
        ),
        "external_model_call_count": matrix.external_model_call_count,
        "sandbox_call_count": matrix.sandbox_call_count,
        "human_record_count": matrix.human_record_count,
        "human_learning_claim_supported": matrix.human_learning_claim_supported,
        "tutoring_efficacy_claim_supported": matrix.tutoring_efficacy_claim_supported,
    }
    manifest = DevelopmentPolicyMatrixManifest.model_validate(
        {**content, "manifest_hash": canonical_sha256(content)}
    )
    write_immutable_bytes(output_root / manifest.comparison_file, comparison_bytes)
    write_immutable_json(output_root / "development_policy_matrix_manifest.json", manifest)
    return manifest


def run_development_policy_matrix(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    *,
    episodes_per_environment: int | None = None,
) -> DevelopmentPolicyMatrix:
    """Run every policy on paired development episodes without external calls."""

    if specification.specification_hash != analysis.environment_specification_hash:
        raise ValueError("Analysis and environment specifications do not match")
    configured_count = specification.episodes.development_episodes_per_environment
    episode_count = (
        configured_count if episodes_per_environment is None else episodes_per_environment
    )
    if episode_count < 1 or episode_count > configured_count:
        raise ValueError(f"Development episode count must lie in [1, {configured_count}]")

    calibration = estimate_probe_reliability(specification)
    comparisons = tuple(
        run_episode_policy_comparison(
            specification,
            analysis,
            calibration,
            environment_id=environment_id,
            episode_index=episode_index,
        )
        for environment_id in EvaluationEnvironmentId
        for episode_index in range(episode_count)
    )
    return DevelopmentPolicyMatrix(
        environment_specification_hash=specification.specification_hash,
        analysis_plan_hash=analysis.plan_hash,
        calibration_hash=calibration.calibration_hash,
        episodes_per_environment=episode_count,
        environment_count=len(EvaluationEnvironmentId),
        comparison_count=len(comparisons),
        policy_count=len(_POLICY_ORDER),
        comparisons=comparisons,
    )


def run_episode_policy_comparison(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    calibration: ReliabilityCalibrationRun,
    *,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
) -> EpisodePolicyComparison:
    """Evaluate every frozen policy against one shared hidden episode."""

    return run_episode_policy_bundle(
        specification,
        analysis,
        calibration,
        environment_id=environment_id,
        episode_index=episode_index,
    ).comparison


def run_episode_policy_bundle(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    calibration: ReliabilityCalibrationRun,
    *,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
) -> EpisodePolicyRun:
    """Return all projections produced by one shared simulator episode."""

    matched_budget = analysis.primary.exact_selected_probes_per_episode
    policy_episode, privileged_episode = generate_acquisition_episode(
        specification,
        calibration,
        environment_id=environment_id,
        episode_index=episode_index,
        probe_budget=matched_budget,
    )
    primary_results = tuple(
        _evaluate(policy, policy_episode, privileged_episode, analysis)
        for policy in build_primary_non_oracle_policies(analysis)
    )
    never_policy, always_policy = build_endpoint_policies(analysis)
    never_result = _evaluate(
        never_policy,
        _with_budget(policy_episode, 0),
        privileged_episode,
        analysis,
    )
    always_result = _evaluate(
        always_policy,
        _with_budget(policy_episode, len(policy_episode.request.candidates)),
        privileged_episode,
        analysis,
    )
    oracle = build_oracle_reference_policy(
        specification,
        analysis,
        environment_id=environment_id,
    )
    oracle_result = _evaluate(oracle, policy_episode, privileged_episode, analysis)
    results = (*primary_results, never_result, always_result, oracle_result)
    comparison = EpisodePolicyComparison(
        environment_id=environment_id,
        episode_index=episode_index,
        episode_id=policy_episode.episode_id,
        shared_candidate_hash=canonical_sha256(policy_episode.request.candidates),
        privileged_episode_hash=model_content_hash(privileged_episode),
        candidate_count=len(policy_episode.request.candidates),
        matched_budget=matched_budget,
        results=results,
    )
    return EpisodePolicyRun(
        policy_episode=policy_episode,
        privileged_episode=privileged_episode,
        comparison=comparison,
    )


def _with_budget(record: PolicyEpisodeRecord, budget: int) -> PolicyEpisodeRecord:
    return PolicyEpisodeRecord(
        episode_id=record.episode_id,
        environment_specification_hash=record.environment_specification_hash,
        calibration_hash=record.calibration_hash,
        request=AcquisitionRequest(
            candidates=record.request.candidates,
            remaining_probe_budget=budget,
        ),
    )


def _evaluate(
    policy: AcquisitionPolicy,
    policy_episode: PolicyEpisodeRecord,
    privileged_episode: PrivilegedEpisodeRecord,
    analysis: AcquisitionAnalysisSpecification,
) -> PolicyEpisodeResult:
    return evaluate_policy_episode(
        policy,
        policy_episode,
        privileged_episode,
        kappa=analysis.policies.bounded_log_likelihood_kappa,
        classification_threshold=analysis.primary.classification_threshold,
        probability_floor=analysis.secondary.probability_floor,
    )
