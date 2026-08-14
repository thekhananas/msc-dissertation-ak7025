"""Transparent policies for fixed-budget probe selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from socratic_tutor.acquisition_study.contracts import (
    AcquisitionCandidate,
    AcquisitionContractError,
    AcquisitionDecision,
    AcquisitionRequest,
    PolicyId,
    ProbeClassId,
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


def unbounded_posterior(
    prior: float,
    *,
    evidence_passed: bool,
    sensitivity: float,
    specificity: float,
) -> float:
    """Apply the ordinary Bayesian update used by the uncapped ablations."""

    values = (prior, sensitivity, specificity)
    if any(not 0.0 < value < 1.0 or not math.isfinite(value) for value in values):
        raise ValueError("Prior, sensitivity, and specificity must be finite and lie in (0, 1)")
    if evidence_passed:
        true_likelihood = sensitivity
        false_likelihood = 1.0 - specificity
    else:
        true_likelihood = 1.0 - sensitivity
        false_likelihood = specificity
    updated_log_odds = math.log(prior / (1.0 - prior)) + math.log(
        true_likelihood / false_likelihood
    )
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


def unbounded_plug_in_evsi(candidate: AcquisitionCandidate) -> float:
    """Estimate risk reduction at mean reliability without capping the update."""

    prior = candidate.prior_success_probability
    sensitivity = candidate.calibration_beta_posterior.sensitivity.mean
    specificity = candidate.calibration_beta_posterior.specificity.mean
    pass_probability = prior * sensitivity + (1.0 - prior) * (1.0 - specificity)
    posterior_if_pass = unbounded_posterior(
        prior,
        evidence_passed=True,
        sensitivity=sensitivity,
        specificity=specificity,
    )
    posterior_if_fail = unbounded_posterior(
        prior,
        evidence_passed=False,
        sensitivity=sensitivity,
        specificity=specificity,
    )
    expected_posterior_risk = pass_probability * classification_risk(posterior_if_pass) + (
        1.0 - pass_probability
    ) * classification_risk(posterior_if_fail)
    return classification_risk(prior) - expected_posterior_risk


def _stream_entropy(seed: int, probe_class: ProbeClassId, parameter: str) -> list[int]:
    payload = f"{probe_class.value}:{parameter}".encode()
    digest = hashlib.sha256(payload).digest()
    return [
        seed,
        *(int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 16, 4)),
    ]


@lru_cache(maxsize=256)
def _reliability_draws(
    probe_class: ProbeClassId,
    sensitivity_alpha: float,
    sensitivity_beta: float,
    specificity_alpha: float,
    specificity_beta: float,
    draw_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sensitivity_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(_stream_entropy(seed, probe_class, "sensitivity")))
    )
    specificity_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(_stream_entropy(seed, probe_class, "specificity")))
    )
    sensitivity = sensitivity_rng.beta(sensitivity_alpha, sensitivity_beta, size=draw_count)
    specificity = specificity_rng.beta(specificity_alpha, specificity_beta, size=draw_count)
    sensitivity.flags.writeable = False
    specificity.flags.writeable = False
    return sensitivity, specificity


def _sampled_evsi(
    candidate: AcquisitionCandidate,
    *,
    kappa: float | None,
    draw_count: int,
    seed: int,
) -> np.ndarray:
    posterior = candidate.calibration_beta_posterior
    sensitivity, specificity = _reliability_draws(
        candidate.probe_class,
        posterior.sensitivity.alpha,
        posterior.sensitivity.beta,
        posterior.specificity.alpha,
        posterior.specificity.beta,
        draw_count,
        seed,
    )
    epsilon = np.finfo(np.float64).eps
    sensitivity = np.clip(sensitivity, epsilon, 1.0 - epsilon)
    specificity = np.clip(specificity, epsilon, 1.0 - epsilon)

    prior = candidate.prior_success_probability
    prior_log_odds = math.log(prior / (1.0 - prior))
    pass_probability = prior * sensitivity + (1.0 - prior) * (1.0 - specificity)
    pass_delta = np.log(sensitivity / (1.0 - specificity))
    fail_delta = np.log((1.0 - sensitivity) / specificity)
    if kappa is not None:
        pass_delta = np.clip(pass_delta, -kappa, kappa)
        fail_delta = np.clip(fail_delta, -kappa, kappa)

    posterior_if_pass = _stable_expit(prior_log_odds + pass_delta)
    posterior_if_fail = _stable_expit(prior_log_odds + fail_delta)
    risk_if_pass = np.minimum(posterior_if_pass, 1.0 - posterior_if_pass)
    risk_if_fail = np.minimum(posterior_if_fail, 1.0 - posterior_if_fail)
    expected_posterior_risk = (
        pass_probability * risk_if_pass + (1.0 - pass_probability) * risk_if_fail
    )
    return classification_risk(prior) - expected_posterior_risk


def _stable_expit(log_odds: np.ndarray) -> np.ndarray:
    result = np.empty_like(log_odds)
    non_negative = log_odds >= 0.0
    result[non_negative] = 1.0 / (1.0 + np.exp(-log_odds[non_negative]))
    odds = np.exp(log_odds[~non_negative])
    result[~non_negative] = odds / (1.0 + odds)
    return result


def reliability_aware_evsi_scores(
    candidates: tuple[AcquisitionCandidate, ...],
    *,
    lower_quantile: float,
    kappa: float,
    draw_count: int,
    seed: int,
) -> dict[str, float]:
    """Return a cautious risk-reduction estimate for each candidate probe."""

    if not 0.0 < lower_quantile < 0.5 or not math.isfinite(lower_quantile):
        raise ValueError("Lower quantile must be finite and lie in (0, 0.5)")
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise ValueError("Kappa must be finite and positive")
    if draw_count < 4_096:
        raise ValueError("Reliability draw count must be at least 4096")
    if seed < 0:
        raise ValueError("Reliability draw seed must be non-negative")

    return {
        candidate.case_id: float(
            np.quantile(
                _sampled_evsi(
                    candidate,
                    kappa=kappa,
                    draw_count=draw_count,
                    seed=seed,
                ),
                lower_quantile,
                method="linear",
            )
        )
        for candidate in sorted(candidates, key=lambda item: item.case_id)
    }


def unbounded_reliability_aware_evsi_scores(
    candidates: tuple[AcquisitionCandidate, ...],
    *,
    lower_quantile: float,
    draw_count: int,
    seed: int,
) -> dict[str, float]:
    """Return cautious risk-reduction scores without capping evidence updates."""

    if not 0.0 < lower_quantile < 0.5 or not math.isfinite(lower_quantile):
        raise ValueError("Lower quantile must be finite and lie in (0, 0.5)")
    if draw_count < 4_096:
        raise ValueError("Reliability draw count must be at least 4096")
    if seed < 0:
        raise ValueError("Reliability draw seed must be non-negative")
    return {
        candidate.case_id: float(
            np.quantile(
                _sampled_evsi(
                    candidate,
                    kappa=None,
                    draw_count=draw_count,
                    seed=seed,
                ),
                lower_quantile,
                method="linear",
            )
        )
        for candidate in sorted(candidates, key=lambda item: item.case_id)
    }


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


@dataclass(frozen=True)
class ReliabilityAwareEVSIPolicy:
    """Rank probes by a lower quantile of uncertain expected risk reduction."""

    lower_quantile: float
    kappa: float
    draw_count: int
    seed: int
    policy_id: PolicyId = field(default=PolicyId.RELIABILITY_AWARE_BOUNDED, init=False)

    def __post_init__(self) -> None:
        reliability_aware_evsi_scores(
            (),
            lower_quantile=self.lower_quantile,
            kappa=self.kappa,
            draw_count=self.draw_count,
            seed=self.seed,
        )

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        scores = reliability_aware_evsi_scores(
            request.candidates,
            lower_quantile=self.lower_quantile,
            kappa=self.kappa,
            draw_count=self.draw_count,
            seed=self.seed,
        )
        ranked = sorted(
            request.candidates,
            key=lambda candidate: (-scores[candidate.case_id], candidate.case_id),
        )
        return _ranked_decision(self.policy_id, request, ranked)


@dataclass(frozen=True)
class ReliabilityAwareUnboundedPolicy:
    """Ablation using cautious reliability but uncapped evidence updates."""

    lower_quantile: float
    draw_count: int
    seed: int
    policy_id: PolicyId = field(default=PolicyId.ABLATION_QUANTILE_UNBOUNDED, init=False)

    def __post_init__(self) -> None:
        unbounded_reliability_aware_evsi_scores(
            (),
            lower_quantile=self.lower_quantile,
            draw_count=self.draw_count,
            seed=self.seed,
        )

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        scores = unbounded_reliability_aware_evsi_scores(
            request.candidates,
            lower_quantile=self.lower_quantile,
            draw_count=self.draw_count,
            seed=self.seed,
        )
        ranked = sorted(
            request.candidates,
            key=lambda candidate: (-scores[candidate.case_id], candidate.case_id),
        )
        return _ranked_decision(self.policy_id, request, ranked)


@dataclass(frozen=True)
class PlugInEVSIUnboundedPolicy:
    """Ablation using mean reliability and uncapped evidence updates."""

    policy_id: PolicyId = field(default=PolicyId.ABLATION_MEAN_UNBOUNDED, init=False)

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        ranked = sorted(
            request.candidates,
            key=lambda candidate: (-unbounded_plug_in_evsi(candidate), candidate.case_id),
        )
        return _ranked_decision(self.policy_id, request, ranked)
