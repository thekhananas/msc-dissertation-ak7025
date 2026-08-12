"""Unattainable policy reference that knows the generating evidence model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from socratic_tutor.acquisition_study.contracts import (
    AcquisitionCandidate,
    AcquisitionDecision,
    AcquisitionRequest,
    PolicyId,
)
from socratic_tutor.acquisition_study.plan import CalibrationDesign, EvaluationEnvironment
from socratic_tutor.acquisition_study.policies import bounded_posterior, classification_risk


def oracle_expected_risk_reduction(
    candidate: AcquisitionCandidate,
    *,
    true_sensitivity: float,
    true_specificity: float,
    missing_probability: float,
    kappa: float,
) -> float:
    """Value a probe under its true channel while retaining the common update."""

    true_rates = (true_sensitivity, true_specificity)
    if any(not 0.0 <= value <= 1.0 or not math.isfinite(value) for value in true_rates):
        raise ValueError("True sensitivity and specificity must be finite and lie in [0, 1]")
    if not 0.0 <= missing_probability < 1.0 or not math.isfinite(missing_probability):
        raise ValueError("Missing probability must be finite and lie in [0, 1)")
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise ValueError("Kappa must be finite and positive")

    prior = candidate.prior_success_probability
    assumed = candidate.calibration_beta_posterior
    posterior_if_pass = bounded_posterior(
        prior,
        evidence_passed=True,
        sensitivity=assumed.sensitivity.mean,
        specificity=assumed.specificity.mean,
        kappa=kappa,
    )
    posterior_if_fail = bounded_posterior(
        prior,
        evidence_passed=False,
        sensitivity=assumed.sensitivity.mean,
        specificity=assumed.specificity.mean,
        kappa=kappa,
    )
    pass_probability = prior * true_sensitivity + (1.0 - prior) * (1.0 - true_specificity)
    observed_risk = pass_probability * classification_risk(posterior_if_pass) + (
        1.0 - pass_probability
    ) * classification_risk(posterior_if_fail)
    expected_risk = (
        missing_probability * classification_risk(prior)
        + (1.0 - missing_probability) * observed_risk
    )
    return classification_risk(prior) - expected_risk


@dataclass(frozen=True)
class TrueReliabilityOraclePolicy:
    """Rank probes with exact environment rates but no realised outcomes."""

    environment: EvaluationEnvironment
    calibration: CalibrationDesign
    kappa: float
    policy_id: PolicyId = field(default=PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED, init=False)

    def __post_init__(self) -> None:
        if self.kappa <= 0.0 or not math.isfinite(self.kappa):
            raise ValueError("Kappa must be finite and positive")

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision:
        calibration_by_class = {item.probe_class: item for item in self.calibration.probe_classes}
        combined_inversion_probability = _xor_probability(
            self.environment.reliability.outcome_inversion_probability,
            self.environment.family_fault.probability,
        )

        def score(candidate: AcquisitionCandidate) -> float:
            base = calibration_by_class[candidate.probe_class]
            scaled_sensitivity = _scale_from_chance(
                base.sensitivity,
                self.environment.reliability.sensitivity_scale_from_chance,
            )
            scaled_specificity = _scale_from_chance(
                base.specificity,
                self.environment.reliability.specificity_scale_from_chance,
            )
            return oracle_expected_risk_reduction(
                candidate,
                true_sensitivity=_after_random_inversion(
                    scaled_sensitivity,
                    combined_inversion_probability,
                ),
                true_specificity=_after_random_inversion(
                    scaled_specificity,
                    combined_inversion_probability,
                ),
                missing_probability=_missing_probability(
                    self.environment,
                    candidate.prior_success_probability,
                ),
                kappa=self.kappa,
            )

        ranked = sorted(
            request.candidates,
            key=lambda candidate: (-score(candidate), candidate.case_id),
        )
        return AcquisitionDecision(
            policy_id=self.policy_id,
            selected_case_ids=tuple(
                candidate.case_id for candidate in ranked[: request.remaining_probe_budget]
            ),
        ).checked_against(request)


def _scale_from_chance(base: float, scale: float) -> float:
    return 0.5 + scale * (base - 0.5)


def _xor_probability(first: float, second: float) -> float:
    return first * (1.0 - second) + (1.0 - first) * second


def _after_random_inversion(probability: float, inversion_probability: float) -> float:
    return inversion_probability + probability * (1.0 - 2.0 * inversion_probability)


def _missing_probability(environment: EvaluationEnvironment, prior: float) -> float:
    difficulty = 1.0 - 2.0 * abs(prior - 0.5)
    return min(
        environment.missingness.maximum_probability,
        environment.missingness.base_probability
        + environment.missingness.prior_difficulty_slope * difficulty,
    )
