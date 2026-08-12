"""Apply acquisition policies to matched episodes and score their predictions."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.burden import (
    AcquisitionBurdenRecord,
    glass_box_burden_record,
)
from socratic_tutor.acquisition_study.contracts import (
    AcquisitionDecision,
    AcquisitionPolicy,
    PolicyId,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.policies import (
    PlugInEVSIPolicy,
    ReliabilityAwareEVSIPolicy,
    SeededRandomPolicy,
    UncertaintyOnlyPolicy,
    bounded_posterior,
)
from socratic_tutor.acquisition_study.simulation import (
    PolicyEpisodeRecord,
    PrivilegedEpisodeRecord,
    SimulatedProbeOutcome,
)
from socratic_tutor.contracts import ContractModel


class AcquisitionEvaluationError(ValueError):
    """Safe and privileged records cannot support a fair policy comparison."""


class CasePredictionResult(ContractModel):
    """Restricted prediction result for one simulated case."""

    case_id: str = Field(min_length=1)
    selected_for_probe: bool
    observed_probe_outcome: SimulatedProbeOutcome | None
    prior_success_probability: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    final_success_probability: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    latent_success: bool
    predicted_success: bool
    classification_error: int = Field(ge=0, le=1)
    squared_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    negative_log_likelihood: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_probe_visibility(self) -> Self:
        if self.selected_for_probe != (self.observed_probe_outcome is not None):
            raise ValueError("Only selected probes may expose an outcome")
        return self


class PolicyEpisodeResult(ContractModel):
    """Restricted per-episode result used for paired policy analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.policy_episode_result.v1"] = (
        "acquisition_study.policy_episode_result.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    episode_id: str = Field(min_length=1)
    environment_id: EvaluationEnvironmentId
    policy_id: PolicyId
    classification_threshold: float = Field(
        default=0.5,
        ge=0.5,
        le=0.5,
        allow_inf_nan=False,
    )
    threshold_rule: Literal["predict_success_at_or_above_threshold"] = (
        "predict_success_at_or_above_threshold"
    )
    probability_floor: float = Field(gt=0.0, lt=0.5, allow_inf_nan=False)
    decision: AcquisitionDecision
    case_results: tuple[CasePredictionResult, ...] = Field(min_length=1)
    final_classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    brier_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    negative_log_likelihood: float = Field(ge=0.0, allow_inf_nan=False)
    burden: AcquisitionBurdenRecord
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.decision.policy_id != self.policy_id or self.burden.policy_id != self.policy_id:
            raise ValueError("Policy identifiers do not reconcile")
        if self.burden.episode_id != self.episode_id:
            raise ValueError("Burden belongs to another episode")
        if self.burden.candidate_count != len(self.case_results):
            raise ValueError("Burden candidate count does not match the scored cases")
        case_ids = tuple(case.case_id for case in self.case_results)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Scored case IDs must be unique")
        selected = tuple(case.case_id for case in self.case_results if case.selected_for_probe)
        if set(selected) != set(self.decision.selected_case_ids):
            raise ValueError("Case results do not match the policy decision")
        if self.burden.selected_probe_count != len(selected):
            raise ValueError("Burden selected count does not match the scored cases")
        count = len(self.case_results)
        expected_error = math.fsum(case.classification_error for case in self.case_results) / count
        expected_brier = math.fsum(case.squared_error for case in self.case_results) / count
        expected_nll = math.fsum(case.negative_log_likelihood for case in self.case_results) / count
        if not math.isclose(self.final_classification_error, expected_error, abs_tol=1e-12):
            raise ValueError("Episode classification error does not reconcile")
        if not math.isclose(self.brier_score, expected_brier, abs_tol=1e-12):
            raise ValueError("Episode Brier score does not reconcile")
        if not math.isclose(self.negative_log_likelihood, expected_nll, abs_tol=1e-12):
            raise ValueError("Episode negative log-likelihood does not reconcile")
        return self


def build_primary_non_oracle_policies(
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[AcquisitionPolicy, ...]:
    """Construct the frozen candidate and three matched-budget comparators."""

    configured = analysis.policies
    policies: tuple[AcquisitionPolicy, ...] = (
        ReliabilityAwareEVSIPolicy(
            lower_quantile=configured.conservative_reliability_quantile,
            kappa=configured.bounded_log_likelihood_kappa,
            draw_count=configured.reliability_draw_count,
            seed=configured.reliability_draw_seed,
        ),
        SeededRandomPolicy(seed=configured.random_policy_seed),
        UncertaintyOnlyPolicy(),
        PlugInEVSIPolicy(kappa=configured.bounded_log_likelihood_kappa),
    )
    expected_ids = (configured.candidate, *configured.matched_budget_comparators)
    if tuple(policy.policy_id for policy in policies) != expected_ids:
        raise AcquisitionEvaluationError("Constructed policies differ from the frozen comparison")
    return policies


def evaluate_policy_episode(
    policy: AcquisitionPolicy,
    policy_episode: PolicyEpisodeRecord,
    privileged_episode: PrivilegedEpisodeRecord,
    *,
    kappa: float,
    classification_threshold: float,
    probability_floor: float,
) -> PolicyEpisodeResult:
    """Select from safe inputs, then reveal and score only the chosen probes."""

    if not math.isclose(classification_threshold, 0.5, abs_tol=1e-12):
        raise AcquisitionEvaluationError("The frozen classification threshold is 0.5")
    if not 0.0 < probability_floor < 0.5 or not math.isfinite(probability_floor):
        raise AcquisitionEvaluationError("Probability floor must be finite and lie in (0, 0.5)")
    if policy_episode.episode_id != privileged_episode.episode_id:
        raise AcquisitionEvaluationError("Safe and privileged records describe different episodes")
    if (
        policy_episode.environment_specification_hash
        != privileged_episode.environment_specification_hash
    ):
        raise AcquisitionEvaluationError("Episode records use different environment specifications")

    decision = policy.select(policy_episode.request).checked_against(policy_episode.request)
    if decision.policy_id != policy.policy_id:
        raise AcquisitionEvaluationError("Policy returned a decision under another identifier")
    selected_case_ids = set(decision.selected_case_ids)
    truth_by_case = {case.case_id: case for case in privileged_episode.cases}
    candidate_ids = {candidate.case_id for candidate in policy_episode.request.candidates}
    if candidate_ids != set(truth_by_case):
        raise AcquisitionEvaluationError("Safe and privileged case sets differ")

    for candidate in policy_episode.request.candidates:
        truth = truth_by_case[candidate.case_id]
        if (
            candidate.probe_class != truth.probe_class
            or candidate.prior_success_probability != truth.prior_success_probability
        ):
            raise AcquisitionEvaluationError("Safe and privileged case projections disagree")

    case_results: list[CasePredictionResult] = []
    missing_count = 0
    for candidate in policy_episode.request.candidates:
        truth = truth_by_case[candidate.case_id]
        selected = candidate.case_id in selected_case_ids
        observed_outcome = truth.observed_probe_outcome if selected else None
        final_probability = candidate.prior_success_probability
        if selected and observed_outcome is SimulatedProbeOutcome.MISSING:
            missing_count += 1
        elif selected and observed_outcome is not None:
            posterior = candidate.calibration_beta_posterior
            final_probability = bounded_posterior(
                candidate.prior_success_probability,
                evidence_passed=observed_outcome is SimulatedProbeOutcome.PASS,
                sensitivity=posterior.sensitivity.mean,
                specificity=posterior.specificity.mean,
                kappa=kappa,
            )

        predicted_success = final_probability >= classification_threshold
        target = float(truth.latent_success)
        likelihood_probability = min(
            1.0 - probability_floor,
            max(probability_floor, final_probability),
        )
        case_results.append(
            CasePredictionResult(
                case_id=candidate.case_id,
                selected_for_probe=selected,
                observed_probe_outcome=observed_outcome,
                prior_success_probability=candidate.prior_success_probability,
                final_success_probability=final_probability,
                latent_success=truth.latent_success,
                predicted_success=predicted_success,
                classification_error=int(predicted_success != truth.latent_success),
                squared_error=(final_probability - target) ** 2,
                negative_log_likelihood=-math.log(
                    likelihood_probability if truth.latent_success else 1.0 - likelihood_probability
                ),
            )
        )

    case_count = len(case_results)
    burden = glass_box_burden_record(
        episode_id=policy_episode.episode_id,
        policy_id=policy.policy_id,
        candidate_count=case_count,
        selected_probe_count=len(selected_case_ids),
        missing_probe_result_count=missing_count,
        failed_probe_result_count=0,
    )
    return PolicyEpisodeResult(
        episode_id=policy_episode.episode_id,
        environment_id=privileged_episode.environment_id,
        policy_id=policy.policy_id,
        probability_floor=probability_floor,
        decision=decision,
        case_results=tuple(case_results),
        final_classification_error=(
            math.fsum(case.classification_error for case in case_results) / case_count
        ),
        brier_score=math.fsum(case.squared_error for case in case_results) / case_count,
        negative_log_likelihood=(
            math.fsum(case.negative_log_likelihood for case in case_results) / case_count
        ),
        burden=burden,
    )
