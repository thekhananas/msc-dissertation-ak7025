"""Checks for the unattainable true-reliability policy reference."""

from __future__ import annotations

import math
from pathlib import Path

from socratic_tutor.acquisition_study import (
    AcquisitionCandidate,
    BetaPosterior,
    CalibrationBetaPosterior,
    EvaluationEnvironmentId,
    PolicyId,
    ProbeClassId,
    build_endpoint_policies,
    build_oracle_reference_policy,
    estimate_probe_reliability,
    generate_acquisition_episode,
    load_acquisition_study_plan,
    oracle_expected_risk_reduction,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"
SPECIFICATION, ANALYSIS_PLAN = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)
CALIBRATION = estimate_probe_reliability(SPECIFICATION)


def test_oracle_value_matches_the_hand_worked_probe_and_missingness_discount() -> None:
    candidate = AcquisitionCandidate(
        case_id="hand-worked",
        prior_success_probability=0.4,
        probe_class=ProbeClassId.MODERATE_RELIABILITY_DENSE,
        calibration_beta_posterior=CalibrationBetaPosterior(
            sensitivity=BetaPosterior(alpha=8.0, beta=2.0),
            specificity=BetaPosterior(alpha=7.0, beta=3.0),
        ),
    )

    full_value = oracle_expected_risk_reduction(
        candidate,
        true_sensitivity=0.8,
        true_specificity=0.7,
        missing_probability=0.0,
        kappa=2.0,
    )
    discounted_value = oracle_expected_risk_reduction(
        candidate,
        true_sensitivity=0.8,
        true_specificity=0.7,
        missing_probability=0.25,
        kappa=2.0,
    )

    assert math.isclose(full_value, 0.14, abs_tol=1e-12)
    assert math.isclose(discounted_value, 0.105, abs_tol=1e-12)


def test_oracle_and_endpoints_match_the_frozen_policy_set() -> None:
    policy_episode, _ = generate_acquisition_episode(
        SPECIFICATION,
        CALIBRATION,
        environment_id=EvaluationEnvironmentId.INVERTED_EVIDENCE,
        episode_index=17,
        probe_budget=ANALYSIS_PLAN.primary.exact_selected_probes_per_episode,
    )
    oracle = build_oracle_reference_policy(
        SPECIFICATION,
        ANALYSIS_PLAN,
        environment_id=EvaluationEnvironmentId.INVERTED_EVIDENCE,
    )
    decision = oracle.select(policy_episode.request)
    endpoints = build_endpoint_policies(ANALYSIS_PLAN)

    assert decision.policy_id is PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED
    assert len(decision.selected_case_ids) == policy_episode.request.remaining_probe_budget
    assert tuple(policy.policy_id for policy in endpoints) == ANALYSIS_PLAN.policies.endpoints
