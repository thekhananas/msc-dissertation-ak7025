"""Behavioural checks for the fixed-budget acquisition baselines."""

from __future__ import annotations

import math

import pytest

from socratic_tutor.acquisition_study import (
    FIXED_RELIABILITY_SHUFFLE,
    AcquisitionCandidate,
    AcquisitionContractError,
    AcquisitionRequest,
    AlwaysProbePolicy,
    BetaPosterior,
    CalibrationBetaPosterior,
    NeverProbePolicy,
    PlugInEVSIPolicy,
    PlugInEVSIUnboundedPolicy,
    PolicyId,
    ProbeClassId,
    ReliabilityAwareEVSIPolicy,
    ReliabilityAwareUnboundedPolicy,
    SeededRandomPolicy,
    ShuffledReliabilityEVSIPolicy,
    UncertaintyOnlyPolicy,
    bounded_posterior,
    shuffle_probe_class_reliability_summaries,
    unbounded_posterior,
)


def _candidate(
    case_id: str,
    prior: float,
    *,
    sensitivity_alpha: float = 90.0,
    sensitivity_beta: float = 10.0,
    specificity_alpha: float = 90.0,
    specificity_beta: float = 10.0,
    probe_class: ProbeClassId = ProbeClassId.HIGH_RELIABILITY_DENSE,
) -> AcquisitionCandidate:
    return AcquisitionCandidate(
        case_id=case_id,
        prior_success_probability=prior,
        probe_class=probe_class,
        calibration_beta_posterior=CalibrationBetaPosterior(
            sensitivity=BetaPosterior(alpha=sensitivity_alpha, beta=sensitivity_beta),
            specificity=BetaPosterior(alpha=specificity_alpha, beta=specificity_beta),
        ),
    )


def _request(*, budget: int = 2) -> AcquisitionRequest:
    return AcquisitionRequest(
        candidates=(
            _candidate("case-a", 0.10),
            _candidate("case-b", 0.49),
            _candidate("case-c", 0.51),
            _candidate("case-d", 0.90),
        ),
        remaining_probe_budget=budget,
    )


def test_endpoints_require_their_declared_budgets() -> None:
    candidates = _request().candidates
    never_request = AcquisitionRequest(candidates=candidates, remaining_probe_budget=0)
    always_request = AcquisitionRequest(
        candidates=candidates, remaining_probe_budget=len(candidates)
    )

    assert NeverProbePolicy().select(never_request).selected_case_ids == ()
    assert AlwaysProbePolicy().select(always_request).selected_case_ids == (
        "case-a",
        "case-b",
        "case-c",
        "case-d",
    )

    with pytest.raises(AcquisitionContractError, match="zero budget"):
        NeverProbePolicy().select(_request())
    with pytest.raises(AcquisitionContractError, match="every candidate"):
        AlwaysProbePolicy().select(_request())


def test_seeded_random_selection_is_repeatable_and_input_order_independent() -> None:
    request = _request()
    reversed_request = AcquisitionRequest(
        candidates=tuple(reversed(request.candidates)),
        remaining_probe_budget=request.remaining_probe_budget,
    )
    policy = SeededRandomPolicy(seed=9052707)

    first = policy.select(request)
    second = policy.select(reversed_request)

    assert first == second
    assert len(first.selected_case_ids) == request.remaining_probe_budget


def test_uncertainty_policy_selects_cases_nearest_one_half_with_stable_ties() -> None:
    decision = UncertaintyOnlyPolicy().select(_request())

    assert decision.selected_case_ids == ("case-b", "case-c")


def test_plugin_evsi_prefers_more_informative_evidence_at_the_same_prior() -> None:
    request = AcquisitionRequest(
        candidates=(
            _candidate(
                "weak-evidence",
                0.5,
                sensitivity_alpha=55.0,
                sensitivity_beta=45.0,
                specificity_alpha=55.0,
                specificity_beta=45.0,
            ),
            _candidate("strong-evidence", 0.5),
        ),
        remaining_probe_budget=1,
    )

    decision = PlugInEVSIPolicy(kappa=2.0).select(request)

    assert decision.selected_case_ids == ("strong-evidence",)


def test_reliability_aware_evsi_penalises_sparse_calibration_evidence() -> None:
    request = AcquisitionRequest(
        candidates=(
            _candidate(
                "a-sparse",
                0.5,
                sensitivity_alpha=9.0,
                sensitivity_beta=1.0,
                specificity_alpha=9.0,
                specificity_beta=1.0,
                probe_class=ProbeClassId.HIGH_RELIABILITY_SPARSE,
            ),
            _candidate("z-dense", 0.5),
        ),
        remaining_probe_budget=1,
    )
    reversed_request = AcquisitionRequest(
        candidates=tuple(reversed(request.candidates)),
        remaining_probe_budget=1,
    )

    assert PlugInEVSIPolicy(kappa=2.0).select(request).selected_case_ids == ("a-sparse",)

    policy = ReliabilityAwareEVSIPolicy(
        lower_quantile=0.10,
        kappa=2.0,
        draw_count=8_192,
        seed=9_052_808,
    )
    assert policy.select(request).selected_case_ids == ("z-dense",)
    assert policy.select(reversed_request) == policy.select(request)


def test_shuffled_reliability_control_reassigns_only_calibration_summaries() -> None:
    candidates = tuple(
        _candidate(
            f"case-{index}",
            0.35 + index * 0.1,
            sensitivity_alpha=70.0 + index,
            sensitivity_beta=30.0 - index,
            specificity_alpha=65.0 + index,
            specificity_beta=35.0 - index,
            probe_class=probe_class,
        )
        for index, probe_class in enumerate(ProbeClassId)
    )
    shuffled = shuffle_probe_class_reliability_summaries(candidates)
    original_by_class = {
        candidate.probe_class: candidate.calibration_beta_posterior for candidate in candidates
    }

    for original, changed in zip(candidates, shuffled, strict=True):
        assert (
            changed.case_id,
            changed.prior_success_probability,
            changed.probe_class,
        ) == (
            original.case_id,
            original.prior_success_probability,
            original.probe_class,
        )
    for target_class, source_class in FIXED_RELIABILITY_SHUFFLE:
        changed = next(item for item in shuffled if item.probe_class is target_class)
        assert changed.calibration_beta_posterior == original_by_class[source_class]

    request = AcquisitionRequest(candidates=candidates, remaining_probe_budget=2)
    reversed_request = AcquisitionRequest(
        candidates=tuple(reversed(candidates)),
        remaining_probe_budget=2,
    )
    policy = ShuffledReliabilityEVSIPolicy(
        lower_quantile=0.10,
        kappa=2.0,
        draw_count=8_192,
        seed=9_052_808,
    )
    decision = policy.select(request)

    assert decision.policy_id is PolicyId.SHUFFLED_RELIABILITY_BOUNDED
    assert len(decision.selected_case_ids) == 2
    assert policy.select(reversed_request) == decision


def test_uncapped_ablation_removes_the_declared_log_odds_limit() -> None:
    prior = 0.5
    bounded = bounded_posterior(
        prior,
        evidence_passed=True,
        sensitivity=0.99,
        specificity=0.99,
        kappa=2.0,
    )
    unbounded = unbounded_posterior(
        prior,
        evidence_passed=True,
        sensitivity=0.99,
        specificity=0.99,
    )

    assert math.isclose(math.log(bounded / (1.0 - bounded)), 2.0, abs_tol=1e-12)
    assert math.log(unbounded / (1.0 - unbounded)) > 2.0


def test_uncapped_ablation_policies_obey_the_same_fixed_budget() -> None:
    request = _request()
    policies = (
        ReliabilityAwareUnboundedPolicy(
            lower_quantile=0.10,
            draw_count=8_192,
            seed=9_052_808,
        ),
        PlugInEVSIUnboundedPolicy(),
    )

    decisions = tuple(policy.select(request) for policy in policies)

    assert all(
        len(decision.selected_case_ids) == request.remaining_probe_budget for decision in decisions
    )
