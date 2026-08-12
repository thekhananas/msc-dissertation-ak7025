"""Deterministic development runner for paired acquisition-policy episodes."""

from __future__ import annotations

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
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
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
        _run_episode_comparison(
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


def _run_episode_comparison(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    calibration: ReliabilityCalibrationRun,
    *,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
) -> EpisodePolicyComparison:
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
    return EpisodePolicyComparison(
        environment_id=environment_id,
        episode_index=episode_index,
        episode_id=policy_episode.episode_id,
        shared_candidate_hash=canonical_sha256(policy_episode.request.candidates),
        privileged_episode_hash=model_content_hash(privileged_episode),
        candidate_count=len(policy_episode.request.candidates),
        matched_budget=matched_budget,
        results=results,
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
