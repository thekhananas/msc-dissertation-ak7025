"""Paired development evaluation across the frozen probe-budget curve."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.calibration import ReliabilityCalibrationRun
from socratic_tutor.acquisition_study.contracts import (
    AcquisitionPolicy,
    AcquisitionRequest,
    PolicyId,
)
from socratic_tutor.acquisition_study.evaluation import (
    PolicyEpisodeResult,
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
    generate_acquisition_episode,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_CURVE_POLICY_ORDER = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)


class BudgetPolicyResult(ContractModel):
    """One policy result at one point on the frozen budget curve."""

    budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    selected_probe_count: int = Field(ge=0)
    result: PolicyEpisodeResult

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.selected_probe_count != self.result.burden.selected_probe_count:
            raise ValueError("Budget result selected-probe count does not reconcile")
        expected = self.budget_fraction * self.result.burden.candidate_count
        if not math.isclose(expected, self.selected_probe_count, abs_tol=1e-12):
            raise ValueError("Budget fraction does not produce an exact probe count")
        return self


class EpisodeBudgetCurve(ContractModel):
    """All ranking policies evaluated at every budget on one hidden episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.episode_budget_curve.v1"] = (
        "acquisition_study.episode_budget_curve.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    environment_id: EvaluationEnvironmentId
    episode_index: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    shared_candidate_hash: Sha256
    privileged_episode_hash: Sha256
    candidate_count: int = Field(ge=1)
    budget_fractions: tuple[float, ...] = Field(min_length=5, max_length=5)
    policy_ids: tuple[PolicyId, ...] = Field(min_length=5, max_length=5)
    results: tuple[BudgetPolicyResult, ...] = Field(min_length=25, max_length=25)
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_curve(self) -> Self:
        if self.episode_id != f"episode-{self.episode_index:06d}":
            raise ValueError("Budget curve episode identifier does not match its index")
        if self.budget_fractions != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("Budget curve differs from the frozen fractions")
        if self.policy_ids != _CURVE_POLICY_ORDER:
            raise ValueError("Budget curve policy order differs from the frozen design")
        expected_keys = tuple(
            (fraction, policy_id)
            for fraction in self.budget_fractions
            for policy_id in self.policy_ids
        )
        actual_keys = tuple((row.budget_fraction, row.result.policy_id) for row in self.results)
        if actual_keys != expected_keys:
            raise ValueError("Budget curve result order or coverage differs")
        if any(
            row.result.episode_id != self.episode_id
            or row.result.environment_id is not self.environment_id
            or row.result.burden.candidate_count != self.candidate_count
            for row in self.results
        ):
            raise ValueError("Budget curve results do not share one episode")
        if any(
            row.result.burden.external_workload.model_request_count != 0
            or row.result.burden.external_workload.sandbox_execution_count != 0
            for row in self.results
        ):
            raise ValueError("Glass-box budget results cannot contain external work")
        reference_cases = tuple(
            (
                case.case_id,
                case.prior_success_probability,
                case.latent_success,
            )
            for case in self.results[0].result.case_results
        )
        if any(
            tuple(
                (
                    case.case_id,
                    case.prior_success_probability,
                    case.latent_success,
                )
                for case in row.result.case_results
            )
            != reference_cases
            for row in self.results
        ):
            raise ValueError("Budget curve policies were not scored on identical cases")
        observed_by_case: dict[str, object] = {}
        for row in self.results:
            for case in row.result.case_results:
                if case.observed_probe_outcome is None:
                    continue
                previous = observed_by_case.setdefault(case.case_id, case.observed_probe_outcome)
                if previous != case.observed_probe_outcome:
                    raise ValueError("Budget curve exposed different outcomes for the same probe")
        for fraction in (0.0, 1.0):
            endpoint_rows = tuple(
                row.result.case_results for row in self.results if row.budget_fraction == fraction
            )
            if len(set(endpoint_rows)) != 1:
                raise ValueError("Policies disagree at a shared budget endpoint")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


def run_episode_budget_curve(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    calibration: ReliabilityCalibrationRun,
    *,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
) -> EpisodeBudgetCurve:
    """Evaluate the fixed policy set across all budgets on one generated episode."""

    if specification.specification_hash != analysis.environment_specification_hash:
        raise ValueError("Analysis and environment specifications do not match")
    if calibration.environment_specification_hash != specification.specification_hash:
        raise ValueError("Calibration belongs to another environment specification")

    policy_episode, privileged_episode = generate_acquisition_episode(
        specification,
        calibration,
        environment_id=environment_id,
        episode_index=episode_index,
        probe_budget=analysis.primary.exact_selected_probes_per_episode,
    )
    policies: tuple[AcquisitionPolicy, ...] = (
        *build_primary_non_oracle_policies(analysis),
        build_oracle_reference_policy(
            specification,
            analysis,
            environment_id=environment_id,
        ),
    )
    if tuple(policy.policy_id for policy in policies) != _CURVE_POLICY_ORDER:
        raise ValueError("Constructed budget-curve policies differ from the frozen design")

    candidate_count = len(policy_episode.request.candidates)
    results: list[BudgetPolicyResult] = []
    for fraction in analysis.secondary.budget_fractions:
        selected_count = int(fraction * candidate_count)
        if not math.isclose(fraction * candidate_count, selected_count, abs_tol=1e-12):
            raise ValueError("A frozen budget fraction does not select a whole number of probes")
        request = AcquisitionRequest(
            candidates=policy_episode.request.candidates,
            remaining_probe_budget=selected_count,
        )
        budget_episode = PolicyEpisodeRecord(
            episode_id=policy_episode.episode_id,
            environment_specification_hash=policy_episode.environment_specification_hash,
            calibration_hash=policy_episode.calibration_hash,
            request=request,
        )
        for policy in policies:
            result = evaluate_policy_episode(
                policy,
                budget_episode,
                privileged_episode,
                kappa=analysis.policies.bounded_log_likelihood_kappa,
                classification_threshold=analysis.primary.classification_threshold,
                probability_floor=analysis.secondary.probability_floor,
            )
            results.append(
                BudgetPolicyResult(
                    budget_fraction=fraction,
                    selected_probe_count=selected_count,
                    result=result,
                )
            )

    return EpisodeBudgetCurve(
        environment_id=environment_id,
        episode_index=episode_index,
        episode_id=policy_episode.episode_id,
        shared_candidate_hash=canonical_sha256(policy_episode.request.candidates),
        privileged_episode_hash=model_content_hash(privileged_episode),
        candidate_count=candidate_count,
        budget_fractions=analysis.secondary.budget_fractions,
        policy_ids=_CURVE_POLICY_ORDER,
        results=tuple(results),
    )
