"""Checks for fair per-episode policy execution and scoring."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from socratic_tutor.acquisition_study import (
    AcquisitionEvaluationError,
    AcquisitionPolicy,
    AlwaysProbePolicy,
    EvaluationEnvironmentId,
    PolicyEpisodeRecord,
    PolicyEpisodeResult,
    PrivilegedEpisodeRecord,
    ReliabilityAwareUnboundedPolicy,
    SimulatedProbeOutcome,
    UncertaintyOnlyPolicy,
    bounded_posterior,
    build_primary_non_oracle_policies,
    estimate_probe_reliability,
    evaluate_policy_episode,
    generate_acquisition_episode,
    load_acquisition_study_plan,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"
SPECIFICATION, ANALYSIS_PLAN = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)
CALIBRATION = estimate_probe_reliability(SPECIFICATION)


def _episode(
    environment_id: EvaluationEnvironmentId,
    *,
    budget: int = 20,
) -> tuple[PolicyEpisodeRecord, PrivilegedEpisodeRecord]:
    return generate_acquisition_episode(
        SPECIFICATION,
        CALIBRATION,
        environment_id=environment_id,
        episode_index=11,
        probe_budget=budget,
    )


def _evaluate(
    policy: AcquisitionPolicy,
    policy_episode: PolicyEpisodeRecord,
    privileged_episode: PrivilegedEpisodeRecord,
) -> PolicyEpisodeResult:
    return evaluate_policy_episode(
        policy,
        policy_episode,
        privileged_episode,
        kappa=ANALYSIS_PLAN.policies.bounded_log_likelihood_kappa,
        classification_threshold=ANALYSIS_PLAN.primary.classification_threshold,
        probability_floor=ANALYSIS_PLAN.secondary.probability_floor,
    )


def test_primary_policies_receive_the_same_episode_and_budget() -> None:
    policy_episode, privileged_episode = _episode(EvaluationEnvironmentId.LOWER_RELIABILITY)
    policies = build_primary_non_oracle_policies(ANALYSIS_PLAN)

    results = tuple(_evaluate(policy, policy_episode, privileged_episode) for policy in policies)

    expected_policy_ids = (
        ANALYSIS_PLAN.policies.candidate,
        *ANALYSIS_PLAN.policies.matched_budget_comparators,
    )
    assert tuple(result.policy_id for result in results) == expected_policy_ids
    assert all(result.episode_id == policy_episode.episode_id for result in results)
    assert all(result.burden.selected_probe_count == 20 for result in results)
    assert all(result.burden.external_workload.model_request_count == 0 for result in results)
    latent_outcomes = tuple(case.latent_success for case in results[0].case_results)
    assert all(
        tuple(case.latent_success for case in result.case_results) == latent_outcomes
        for result in results
    )


def test_selected_results_use_the_common_update_and_missing_results_do_not_update() -> None:
    policy_episode, privileged_episode = _episode(
        EvaluationEnvironmentId.DIFFICULTY_MISSINGNESS,
        budget=40,
    )
    result = _evaluate(AlwaysProbePolicy(), policy_episode, privileged_episode)
    candidates = {candidate.case_id: candidate for candidate in policy_episode.request.candidates}

    assert result.burden.missing_probe_result_count > 0
    for case in result.case_results:
        candidate = candidates[case.case_id]
        if case.observed_probe_outcome is SimulatedProbeOutcome.MISSING:
            assert case.final_success_probability == case.prior_success_probability
            continue
        posterior = candidate.calibration_beta_posterior
        expected = bounded_posterior(
            candidate.prior_success_probability,
            evidence_passed=case.observed_probe_outcome is SimulatedProbeOutcome.PASS,
            sensitivity=posterior.sensitivity.mean,
            specificity=posterior.specificity.mean,
            kappa=ANALYSIS_PLAN.policies.bounded_log_likelihood_kappa,
        )
        assert math.isclose(case.final_success_probability, expected, abs_tol=1e-12)


def test_unselected_probe_outcome_cannot_change_a_policy_result() -> None:
    policy_episode, privileged_episode = _episode(EvaluationEnvironmentId.MATCHED)
    policy = UncertaintyOnlyPolicy()
    original = _evaluate(policy, policy_episode, privileged_episode)
    selected = set(original.decision.selected_case_ids)
    unselected_index = next(
        index for index, case in enumerate(privileged_episode.cases) if case.case_id not in selected
    )
    original_truth = privileged_episode.cases[unselected_index]
    replacement_outcome = (
        SimulatedProbeOutcome.FAIL
        if original_truth.observed_probe_outcome is SimulatedProbeOutcome.PASS
        else SimulatedProbeOutcome.PASS
    )
    changed_cases = list(privileged_episode.cases)
    changed_cases[unselected_index] = original_truth.model_copy(
        update={"observed_probe_outcome": replacement_outcome}
    )
    changed_truth = privileged_episode.model_copy(update={"cases": tuple(changed_cases)})

    assert _evaluate(policy, policy_episode, changed_truth) == original


def test_uncapped_ablation_requires_and_applies_the_uncapped_update() -> None:
    policy_episode, privileged_episode = _episode(EvaluationEnvironmentId.MATCHED, budget=40)
    policy = ReliabilityAwareUnboundedPolicy(
        lower_quantile=ANALYSIS_PLAN.policies.conservative_reliability_quantile,
        draw_count=ANALYSIS_PLAN.policies.reliability_draw_count,
        seed=ANALYSIS_PLAN.policies.reliability_draw_seed,
    )

    result = evaluate_policy_episode(
        policy,
        policy_episode,
        privileged_episode,
        kappa=None,
        classification_threshold=ANALYSIS_PLAN.primary.classification_threshold,
        probability_floor=ANALYSIS_PLAN.secondary.probability_floor,
    )

    assert result.policy_id is policy.policy_id
    assert result.burden.selected_probe_count == 40
    with pytest.raises(AcquisitionEvaluationError, match="update rule"):
        evaluate_policy_episode(
            policy,
            policy_episode,
            privileged_episode,
            kappa=ANALYSIS_PLAN.policies.bounded_log_likelihood_kappa,
            classification_threshold=ANALYSIS_PLAN.primary.classification_threshold,
            probability_floor=ANALYSIS_PLAN.secondary.probability_floor,
        )
