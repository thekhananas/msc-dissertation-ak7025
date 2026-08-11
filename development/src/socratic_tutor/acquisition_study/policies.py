"""Transparent baseline policies for fixed-budget probe selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from socratic_tutor.acquisition_study.contracts import (
    AcquisitionCandidate,
    AcquisitionContractError,
    AcquisitionDecision,
    AcquisitionRequest,
    PolicyId,
)


def classification_risk(success_probability: float) -> float:
    """Return the error risk of classifying at probability one half."""

    if not 0.0 <= success_probability <= 1.0 or not math.isfinite(success_probability):
        raise ValueError("Success probability must be finite and lie in [0, 1]")
    return min(success_probability, 1.0 - success_probability)


def bounded_posterior(
    prior: float,
    *,
    evidence_passed: bool,
    sensitivity: float,
    specificity: float,
    kappa: float,
) -> float:
    """Apply one Bayesian evidence update with a bounded log-likelihood change."""

    values = (prior, sensitivity, specificity)
    if any(not 0.0 < value < 1.0 or not math.isfinite(value) for value in values):
        raise ValueError("Prior, sensitivity, and specificity must be finite and lie in (0, 1)")
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise ValueError("Kappa must be finite and positive")

    if evidence_passed:
        true_likelihood = sensitivity
        false_likelihood = 1.0 - specificity
    else:
        true_likelihood = 1.0 - sensitivity
        false_likelihood = specificity

    raw_delta = math.log(true_likelihood / false_likelihood)
    bounded_delta = min(kappa, max(-kappa, raw_delta))
    updated_log_odds = math.log(prior / (1.0 - prior)) + bounded_delta
    if updated_log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-updated_log_odds))
    odds = math.exp(updated_log_odds)
    return odds / (1.0 + odds)


def plug_in_evsi(candidate: AcquisitionCandidate, *, kappa: float) -> float:
    """Estimate risk reduction using mean calibrated sensitivity and specificity."""

    prior = candidate.prior_success_probability
    sensitivity = candidate.calibration_beta_posterior.sensitivity.mean
    specificity = candidate.calibration_beta_posterior.specificity.mean
    pass_probability = prior * sensitivity + (1.0 - prior) * (1.0 - specificity)
    posterior_if_pass = bounded_posterior(
        prior,
        evidence_passed=True,
        sensitivity=sensitivity,
        specificity=specificity,
        kappa=kappa,
    )
    posterior_if_fail = bounded_posterior(
        prior,
        evidence_passed=False,
        sensitivity=sensitivity,
        specificity=specificity,
        kappa=kappa,
    )
    expected_posterior_risk = pass_probability * classification_risk(posterior_if_pass) + (
        1.0 - pass_probability
    ) * classification_risk(posterior_if_fail)
    return classification_risk(prior) - expected_posterior_risk


def _ranked_decision(
    policy_id: PolicyId,
    request: AcquisitionRequest,
    ranked_candidates: list[AcquisitionCandidate],
) -> AcquisitionDecision:
    selected = tuple(
        candidate.case_id for candidate in ranked_candidates[: request.remaining_probe_budget]
    )
    return AcquisitionDecision(
        policy_id=policy_id,
        selected_case_ids=selected,
    ).checked_against(request)


@dataclass(frozen=True)
class NeverProbePolicy:
    """Endpoint representing prediction without additional evidence."""

    policy_id: PolicyId = field(default=PolicyId.NEVER_PROBE, init=False)

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        if request.remaining_probe_budget != 0:
            raise AcquisitionContractError("The never-probe endpoint requires a zero budget")
        return _ranked_decision(self.policy_id, request, [])


@dataclass(frozen=True)
class AlwaysProbePolicy:
    """Endpoint representing evidence collection for every candidate."""

    policy_id: PolicyId = field(default=PolicyId.ALWAYS_PROBE_BOUNDED, init=False)

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        if request.remaining_probe_budget != len(request.candidates):
            raise AcquisitionContractError(
                "The always-probe endpoint requires a budget covering every candidate"
            )
        ranked = sorted(request.candidates, key=lambda candidate: candidate.case_id)
        return _ranked_decision(self.policy_id, request, ranked)


@dataclass(frozen=True)
class SeededRandomPolicy:
    """Stable pseudo-random selection that does not depend on input order."""

    seed: int
    policy_id: PolicyId = field(default=PolicyId.SEEDED_RANDOM_BOUNDED, init=False)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("Random policy seed must be non-negative")

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        def random_key(candidate: AcquisitionCandidate) -> tuple[bytes, str]:
            payload = f"{self.seed}:{candidate.case_id}".encode()
            return hashlib.sha256(payload).digest(), candidate.case_id

        ranked = sorted(request.candidates, key=random_key)
        return _ranked_decision(self.policy_id, request, ranked)


@dataclass(frozen=True)
class UncertaintyOnlyPolicy:
    """Select cases nearest the classification boundary."""

    policy_id: PolicyId = field(default=PolicyId.UNCERTAINTY_ONLY_BOUNDED, init=False)

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        ranked = sorted(
            request.candidates,
            key=lambda candidate: (
                -classification_risk(candidate.prior_success_probability),
                candidate.case_id,
            ),
        )
        return _ranked_decision(self.policy_id, request, ranked)


@dataclass(frozen=True)
class PlugInEVSIPolicy:
    """Rank probes by expected risk reduction at mean calibrated reliability."""

    kappa: float
    policy_id: PolicyId = field(default=PolicyId.PLUG_IN_EVSI_BOUNDED, init=False)

    def __post_init__(self) -> None:
        if self.kappa <= 0.0 or not math.isfinite(self.kappa):
            raise ValueError("Kappa must be finite and positive")

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        ranked = sorted(
            request.candidates,
            key=lambda candidate: (-plug_in_evsi(candidate, kappa=self.kappa), candidate.case_id),
        )
        return _ranked_decision(self.policy_id, request, ranked)
