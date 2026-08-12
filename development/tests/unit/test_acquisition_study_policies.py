"""Behavioural checks for the fixed-budget acquisition baselines."""

from __future__ import annotations

import pytest

from socratic_tutor.acquisition_study import (
    AcquisitionCandidate,
    AcquisitionContractError,
    AcquisitionRequest,
    AlwaysProbePolicy,
    BetaPosterior,
    CalibrationBetaPosterior,
    NeverProbePolicy,
    PlugInEVSIPolicy,
    ProbeClassId,
    ReliabilityAwareEVSIPolicy,
    SeededRandomPolicy,
    UncertaintyOnlyPolicy,
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
