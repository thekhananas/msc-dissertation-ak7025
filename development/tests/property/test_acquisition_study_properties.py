from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from socratic_tutor.acquisition_study import (
    AcquisitionCandidate,
    AcquisitionRequest,
    BetaPosterior,
    CalibrationBetaPosterior,
    PlugInEVSIPolicy,
    ProbeClassId,
    ReliabilityAwareEVSIPolicy,
    bounded_posterior,
    classification_risk,
    plug_in_evsi,
    reliability_aware_evsi_scores,
)

OPEN_PROBABILITY = st.floats(
    min_value=1e-6,
    max_value=1.0 - 1e-6,
    allow_nan=False,
    allow_infinity=False,
)


def _candidate(
    case_id: str,
    *,
    prior: float,
    sensitivity_alpha: float,
    sensitivity_beta: float,
    specificity_alpha: float,
    specificity_beta: float,
    probe_class: ProbeClassId,
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


@given(
    prior=OPEN_PROBABILITY,
    sensitivity=OPEN_PROBABILITY,
    specificity=OPEN_PROBABILITY,
    evidence_passed=st.booleans(),
    kappa=st.floats(
        min_value=0.01,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_one_probe_cannot_exceed_the_log_odds_bound(
    prior: float,
    sensitivity: float,
    specificity: float,
    evidence_passed: bool,
    kappa: float,
) -> None:
    posterior = bounded_posterior(
        prior,
        evidence_passed=evidence_passed,
        sensitivity=sensitivity,
        specificity=specificity,
        kappa=kappa,
    )
    prior_log_odds = math.log(prior / (1.0 - prior))
    posterior_log_odds = math.log(posterior / (1.0 - posterior))

    assert 0.0 < posterior < 1.0
    assert abs(posterior_log_odds - prior_log_odds) <= kappa + 1e-9


@given(
    probability=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_classification_risk_cannot_be_negative(probability: float) -> None:
    assert 0.0 <= classification_risk(probability) <= 0.5


def test_plugin_evsi_matches_the_hand_worked_binary_enumeration() -> None:
    candidate = _candidate(
        "hand-worked",
        prior=0.4,
        sensitivity_alpha=8.0,
        sensitivity_beta=2.0,
        specificity_alpha=7.0,
        specificity_beta=3.0,
        probe_class=ProbeClassId.HIGH_RELIABILITY_DENSE,
    )

    assert math.isclose(plug_in_evsi(candidate, kappa=2.0), 0.14, abs_tol=1e-12)


def test_concentrated_reliability_recovers_plugin_ordering_without_active_clipping() -> None:
    candidates = (
        _candidate(
            "strong",
            prior=0.5,
            sensitivity_alpha=900_000.0,
            sensitivity_beta=100_000.0,
            specificity_alpha=900_000.0,
            specificity_beta=100_000.0,
            probe_class=ProbeClassId.HIGH_RELIABILITY_DENSE,
        ),
        _candidate(
            "moderate",
            prior=0.5,
            sensitivity_alpha=700_000.0,
            sensitivity_beta=300_000.0,
            specificity_alpha=700_000.0,
            specificity_beta=300_000.0,
            probe_class=ProbeClassId.MODERATE_RELIABILITY_DENSE,
        ),
    )
    request = AcquisitionRequest(candidates=candidates, remaining_probe_budget=1)
    scores = reliability_aware_evsi_scores(
        candidates,
        lower_quantile=0.10,
        kappa=100.0,
        draw_count=8_192,
        seed=9_052_808,
    )

    assert (
        ReliabilityAwareEVSIPolicy(0.10, 100.0, 8_192, 9_052_808).select(request).selected_case_ids
        == PlugInEVSIPolicy(kappa=100.0).select(request).selected_case_ids
    )
    assert math.isclose(scores["strong"], plug_in_evsi(candidates[0], kappa=100.0), abs_tol=0.002)
    assert math.isclose(scores["moderate"], plug_in_evsi(candidates[1], kappa=100.0), abs_tol=0.002)
