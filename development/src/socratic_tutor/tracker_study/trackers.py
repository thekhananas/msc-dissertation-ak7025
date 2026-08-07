"""Transparent tracker baselines for the glass-box robustness study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, model_validator

from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.config import (
    EvidenceCategory,
    EvidenceChannel,
    LatentDynamics,
    TrackerId,
    TrackerStudyConfiguration,
)


class EvidenceObservation(ContractModel):
    """The identical observable record supplied to every tracker."""

    channel: EvidenceChannel
    category: EvidenceCategory
    confidence: float = Field(ge=0.5, le=1.0)

    @model_validator(mode="after")
    def validate_missing_confidence(self) -> EvidenceObservation:
        if self.category is EvidenceCategory.MISSING and self.confidence != 0.5:
            raise ValueError("Missing evidence must use neutral confidence 0.5")
        return self


class TrackerUpdate(ContractModel):
    """One auditable correction step before the state transition."""

    tracker_id: TrackerId
    prior_mastery_probability: float = Field(ge=0.0, le=1.0)
    posterior_mastery_probability: float = Field(ge=0.0, le=1.0)
    evidence_used: bool
    raw_log_likelihood_ratio: float | None
    applied_log_odds_delta: float | None
    confidence_used: bool
    channel_model_used: bool
    update_kind: Literal[
        "last_observation",
        "hard_bkt_bayes",
        "legacy_fractional_pseudo_likelihood",
        "channel_aware_bayes",
        "bounded_channel_aware_bayes",
        "missing_no_correction",
    ]


class MasteryTracker(Protocol):
    tracker_id: TrackerId

    def update(self, prior: float, observation: EvidenceObservation) -> TrackerUpdate: ...


@dataclass(frozen=True)
class ConfiguredMasteryTracker:
    """One tracker implementation selected by a stable identifier."""

    tracker_id: TrackerId
    configuration: TrackerStudyConfiguration
    trust_weight_override: float | None = None
    trust_weight_multiplier: float = 1.0
    clipping_kappa_override: float | None = None
    clipping_disabled: bool = False
    assumed_channel_reliability_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.trust_weight_override is not None and not (
            0.0 <= self.trust_weight_override <= 1.0
        ):
            raise ValueError("Trust-weight override must lie in [0, 1]")
        if not 0.0 <= self.trust_weight_multiplier <= 1.0:
            raise ValueError("Trust-weight multiplier must lie in [0, 1]")
        if self.clipping_kappa_override is not None and self.clipping_kappa_override <= 0.0:
            raise ValueError("Clipping override must be positive")
        if not 0.0 <= self.assumed_channel_reliability_scale <= 1.0:
            raise ValueError("Assumed channel reliability scale must lie in [0, 1]")

    def update(self, prior: float, observation: EvidenceObservation) -> TrackerUpdate:
        floor = self.configuration.trackers.probability_floor
        bounded_prior = _clamp_probability(prior, floor)
        if observation.category is EvidenceCategory.MISSING:
            return _missing_update(self.tracker_id, prior)
        if self.tracker_id is TrackerId.LAST_OBSERVATION:
            posterior = (
                1.0 - floor if observation.category is EvidenceCategory.SUPPORTS_MASTERY else floor
            )
            return TrackerUpdate(
                tracker_id=self.tracker_id,
                prior_mastery_probability=prior,
                posterior_mastery_probability=posterior,
                evidence_used=True,
                raw_log_likelihood_ratio=None,
                applied_log_odds_delta=None,
                confidence_used=False,
                channel_model_used=False,
                update_kind="last_observation",
            )
        if self.tracker_id is TrackerId.HARD_BKT:
            likelihoods = _hard_bkt_likelihoods(observation.category, self.configuration)
            return _likelihood_update(
                tracker_id=self.tracker_id,
                prior=prior,
                bounded_prior=bounded_prior,
                likelihoods=likelihoods,
                floor=floor,
                confidence_used=False,
                channel_model_used=False,
                update_kind="hard_bkt_bayes",
            )
        if self.tracker_id is TrackerId.LEGACY_FRACTIONAL:
            likelihoods = _legacy_fractional_likelihoods(observation, self.configuration)
            return _likelihood_update(
                tracker_id=self.tracker_id,
                prior=prior,
                bounded_prior=bounded_prior,
                likelihoods=likelihoods,
                floor=floor,
                confidence_used=True,
                channel_model_used=False,
                update_kind="legacy_fractional_pseudo_likelihood",
            )

        likelihoods = channel_likelihoods(
            observation,
            self.configuration,
            reliability_scale=self.assumed_channel_reliability_scale,
        )
        raw_delta = math.log(likelihoods[0] / likelihoods[1])
        if self.tracker_id is TrackerId.CHANNEL_AWARE:
            applied_delta = raw_delta
            kind = "channel_aware_bayes"
        elif self.tracker_id is TrackerId.BOUNDED_CHANNEL_AWARE:
            trust = self.trust_weight_override
            if trust is None:
                trust = self.configuration.trackers.channel_trust_weights[observation.channel]
            trust *= self.trust_weight_multiplier
            kappa = self.clipping_kappa_override
            if kappa is None:
                kappa = self.configuration.trackers.bounded_log_likelihood_kappa
            clipped = raw_delta if self.clipping_disabled else min(kappa, max(-kappa, raw_delta))
            applied_delta = trust * clipped
            kind = "bounded_channel_aware_bayes"
        else:
            raise ValueError(f"Unsupported tracker: {self.tracker_id}")
        return TrackerUpdate(
            tracker_id=self.tracker_id,
            prior_mastery_probability=prior,
            posterior_mastery_probability=posterior_from_log_odds_delta(
                bounded_prior, applied_delta, floor
            ),
            evidence_used=True,
            raw_log_likelihood_ratio=raw_delta,
            applied_log_odds_delta=applied_delta,
            confidence_used=False,
            channel_model_used=True,
            update_kind=kind,
        )


def build_trackers(
    configuration: TrackerStudyConfiguration,
) -> tuple[ConfiguredMasteryTracker, ...]:
    """Construct every frozen baseline in the declared comparison order."""

    return tuple(
        ConfiguredMasteryTracker(tracker_id=tracker_id, configuration=configuration)
        for tracker_id in configuration.trackers.tracker_ids
    )


def channel_likelihoods(
    observation: EvidenceObservation,
    configuration: TrackerStudyConfiguration,
    *,
    reliability_scale: float = 1.0,
) -> tuple[float, float]:
    """Return P(observation | mastered) and P(observation | not mastered)."""

    if not 0.0 <= reliability_scale <= 1.0:
        raise ValueError("Channel reliability scale must lie in [0, 1]")
    if observation.category is EvidenceCategory.MISSING:
        return (1.0, 1.0)
    model = next(
        item for item in configuration.observation_models if item.channel is observation.channel
    )
    field = observation.category.value
    mastered = getattr(model.given_mastered, field)
    not_mastered = getattr(model.given_not_mastered, field)
    if reliability_scale == 1.0:
        return mastered, not_mastered
    return (
        0.5 + reliability_scale * (mastered - 0.5),
        0.5 + reliability_scale * (not_mastered - 0.5),
    )


def posterior_from_log_odds_delta(prior: float, delta: float, floor: float) -> float:
    """Apply one likelihood-ratio correction with stable probability bounds."""

    bounded_prior = _clamp_probability(prior, floor)
    if delta == 0.0:
        return prior
    prior_log_odds = math.log(bounded_prior / (1.0 - bounded_prior))
    return _clamp_probability(_sigmoid(prior_log_odds + delta), floor)


def propagate_mastery_probability(
    posterior: float, dynamics: LatentDynamics, floor: float
) -> float:
    """Apply the slow BKT-style learning and forgetting transition once per turn."""

    bounded_posterior = _clamp_probability(posterior, floor)
    next_probability = (
        bounded_posterior * (1.0 - dynamics.forget_probability)
        + (1.0 - bounded_posterior) * dynamics.learn_probability
    )
    return _clamp_probability(next_probability, floor)


def _hard_bkt_likelihoods(
    category: EvidenceCategory, configuration: TrackerStudyConfiguration
) -> tuple[float, float]:
    slip = configuration.trackers.hard_bkt_slip
    guess = configuration.trackers.hard_bkt_guess
    if category is EvidenceCategory.SUPPORTS_MASTERY:
        return (1.0 - slip, guess)
    return (slip, 1.0 - guess)


def _legacy_fractional_likelihoods(
    observation: EvidenceObservation, configuration: TrackerStudyConfiguration
) -> tuple[float, float]:
    confidence = observation.confidence
    rho = (
        confidence
        if observation.category is EvidenceCategory.SUPPORTS_MASTERY
        else 1.0 - confidence
    )
    epsilon = configuration.trackers.legacy_noise_epsilon
    likelihood_if_mastered = _fractional_bernoulli_likelihood(rho, 1.0 - epsilon)
    likelihood_if_not_mastered = _fractional_bernoulli_likelihood(rho, epsilon)
    return likelihood_if_mastered, likelihood_if_not_mastered


def _fractional_bernoulli_likelihood(target: float, probability: float) -> float:
    return math.exp(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability))


def _likelihood_update(
    *,
    tracker_id: TrackerId,
    prior: float,
    bounded_prior: float,
    likelihoods: tuple[float, float],
    floor: float,
    confidence_used: bool,
    channel_model_used: bool,
    update_kind: Literal[
        "hard_bkt_bayes",
        "legacy_fractional_pseudo_likelihood",
    ],
) -> TrackerUpdate:
    raw_delta = math.log(likelihoods[0] / likelihoods[1])
    return TrackerUpdate(
        tracker_id=tracker_id,
        prior_mastery_probability=prior,
        posterior_mastery_probability=posterior_from_log_odds_delta(
            bounded_prior, raw_delta, floor
        ),
        evidence_used=True,
        raw_log_likelihood_ratio=raw_delta,
        applied_log_odds_delta=raw_delta,
        confidence_used=confidence_used,
        channel_model_used=channel_model_used,
        update_kind=update_kind,
    )


def _missing_update(tracker_id: TrackerId, prior: float) -> TrackerUpdate:
    _validate_probability(prior)
    return TrackerUpdate(
        tracker_id=tracker_id,
        prior_mastery_probability=prior,
        posterior_mastery_probability=prior,
        evidence_used=False,
        raw_log_likelihood_ratio=0.0,
        applied_log_odds_delta=0.0,
        confidence_used=False,
        channel_model_used=False,
        update_kind="missing_no_correction",
    )


def _clamp_probability(value: float, floor: float) -> float:
    _validate_probability(value)
    return min(1.0 - floor, max(floor, value))


def _validate_probability(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("Mastery probability must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError("Mastery probability must lie in [0, 1]")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
