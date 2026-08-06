"""Frozen configuration contracts for the Bayesian tracker study."""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel


class TrackerStudyConfigurationError(ValueError):
    """A tracker-study input is missing, malformed, or internally inconsistent."""


class EvidenceChannel(StrEnum):
    PUBLIC_ANSWER = "public_answer"
    EXECUTABLE_PROBE = "executable_probe"
    SELF_EXPLANATION = "self_explanation"


class EvidenceCategory(StrEnum):
    SUPPORTS_MASTERY = "supports_mastery"
    SUPPORTS_NON_MASTERY = "supports_non_mastery"
    MISSING = "missing"


class StressCondition(StrEnum):
    CLEAN = "clean"
    UNINFORMATIVE = "uninformative"
    INVERTED = "inverted"
    MISSING = "missing"
    RELIABILITY_MISSPECIFIED = "reliability_misspecified"
    CONTRADICTORY = "contradictory"


class TrackerId(StrEnum):
    LAST_OBSERVATION = "last_observation"
    HARD_BKT = "hard_bkt"
    LEGACY_FRACTIONAL = "legacy_fractional_bernoulli"
    CHANNEL_AWARE = "channel_aware_virtual_evidence"
    BOUNDED_CHANNEL_AWARE = "bounded_channel_aware"


class LatentDynamics(ContractModel):
    initial_mastery_probability: float = Field(gt=0.0, lt=1.0)
    learn_probability: float = Field(ge=0.0, lt=1.0)
    forget_probability: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_transition(self) -> LatentDynamics:
        if self.learn_probability + self.forget_probability >= 1.0:
            raise ValueError("Learning and forgetting probabilities must sum to less than one")
        return self


class CategoricalDistribution(ContractModel):
    supports_mastery: float = Field(gt=0.0, lt=1.0)
    supports_non_mastery: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> CategoricalDistribution:
        if not math.isclose(self.supports_mastery + self.supports_non_mastery, 1.0, abs_tol=1e-12):
            raise ValueError("Observation probabilities must sum to one")
        return self


class ChannelObservationModel(ContractModel):
    channel: EvidenceChannel
    given_mastered: CategoricalDistribution
    given_not_mastered: CategoricalDistribution
    confidence_supports_mastery: float = Field(ge=0.5, le=1.0)
    confidence_supports_non_mastery: float = Field(ge=0.5, le=1.0)

    @model_validator(mode="after")
    def validate_information_direction(self) -> ChannelObservationModel:
        if self.given_mastered.supports_mastery <= self.given_not_mastered.supports_mastery:
            raise ValueError("Base channel must support mastery more often when mastered")
        return self


class StressSpecification(ContractModel):
    condition: StressCondition
    definition: str = Field(min_length=20)
    generator_reliability_scale: float = Field(ge=0.0, le=1.0)
    inversion_probability: float = Field(ge=0.0, le=1.0)
    missing_probability: float = Field(ge=0.0, lt=1.0)
    contradictory_pair_probability: float = Field(ge=0.0, le=1.0)
    tracker_uses_base_observation_model: Literal[True] = True


class SeedHierarchy(ContractModel):
    root: int = Field(ge=0)
    development: int = Field(ge=0)
    training: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_unique_seeds(self) -> SeedHierarchy:
        values = (self.root, self.development, self.training, self.validation, self.test)
        if len(set(values)) != len(values):
            raise ValueError("Seed hierarchy values must be distinct")
        return self


class ExperimentSize(ContractModel):
    development_episodes_per_condition: int = Field(ge=10)
    test_episodes_per_condition: int = Field(ge=100)
    turns_per_episode: int = Field(ge=5)
    ece_bin_count: int = Field(ge=5)
    canonical_split: Literal["test"] = "test"
    canonical_test_run_limit: Literal[1] = 1


class TrackerParameters(ContractModel):
    tracker_ids: tuple[TrackerId, ...] = Field(min_length=5, max_length=5)
    probability_floor: float = Field(gt=0.0, lt=0.01)
    hard_bkt_slip: float = Field(gt=0.0, lt=0.5)
    hard_bkt_guess: float = Field(gt=0.0, lt=0.5)
    legacy_noise_epsilon: float = Field(gt=0.0, lt=0.5)
    bounded_log_likelihood_kappa: float = Field(gt=0.0)
    channel_trust_weights: dict[EvidenceChannel, float]

    @model_validator(mode="after")
    def validate_trackers(self) -> TrackerParameters:
        if tuple(self.tracker_ids) != tuple(TrackerId):
            raise ValueError("Tracker comparison must contain each frozen baseline once")
        if set(self.channel_trust_weights) != set(EvidenceChannel):
            raise ValueError("Every evidence channel needs one trust weight")
        if any(not 0.0 <= weight <= 1.0 for weight in self.channel_trust_weights.values()):
            raise ValueError("Channel trust weights must lie in [0, 1]")
        return self


class SensitivityGrid(ContractModel):
    trust_weight_multipliers: tuple[float, ...] = Field(min_length=2)
    clipping_kappas: tuple[float, ...] = Field(min_length=2)
    reliability_scales: tuple[float, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_grid(self) -> SensitivityGrid:
        if any(value < 0.0 or value > 1.0 for value in self.trust_weight_multipliers):
            raise ValueError("Trust multipliers must lie in [0, 1]")
        if any(value <= 0.0 for value in self.clipping_kappas):
            raise ValueError("Clipping values must be positive")
        if any(value < 0.0 or value > 1.0 for value in self.reliability_scales):
            raise ValueError("Reliability scales must lie in [0, 1]")
        return self


class TrackerStudyConfiguration(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.configuration.v1"] = "tracker_study.configuration.v1"
    study_id: Literal["bayesian-evidence-robustness-v1"] = "bayesian-evidence-robustness-v1"
    claim_scope: Literal["glass_box_simulator_robustness_not_human_cognition"] = (
        "glass_box_simulator_robustness_not_human_cognition"
    )
    configuration_status: Literal["prespecified_before_results"] = "prespecified_before_results"
    confidence_assignment: Literal["fixed_channel_mapping_not_fitted"] = (
        "fixed_channel_mapping_not_fitted"
    )
    held_out_benchmark_outcomes_used: Literal[False] = False
    external_model_data_used: Literal[False] = False
    human_data_used: Literal[False] = False
    latent_dynamics: LatentDynamics
    observation_probability_floor: float = Field(gt=0.0, lt=0.5)
    channels_per_turn: tuple[EvidenceChannel, ...] = Field(min_length=3, max_length=3)
    contradictory_channel_pair: tuple[EvidenceChannel, EvidenceChannel]
    observation_models: tuple[ChannelObservationModel, ...] = Field(min_length=3, max_length=3)
    stress_conditions: tuple[StressSpecification, ...] = Field(min_length=6, max_length=6)
    seeds: SeedHierarchy
    experiment: ExperimentSize
    trackers: TrackerParameters
    sensitivity: SensitivityGrid
    configuration_hash: Sha256

    @model_validator(mode="after")
    def validate_configuration(self) -> TrackerStudyConfiguration:
        if tuple(model.channel for model in self.observation_models) != tuple(EvidenceChannel):
            raise ValueError("Observation models must contain each channel once and in order")
        if self.channels_per_turn != tuple(EvidenceChannel):
            raise ValueError("Every turn must emit the three frozen evidence channels")
        if self.contradictory_channel_pair != (
            EvidenceChannel.PUBLIC_ANSWER,
            EvidenceChannel.EXECUTABLE_PROBE,
        ):
            raise ValueError("Contradictory stress must use the frozen channel pair")
        probabilities = (
            probability
            for model in self.observation_models
            for distribution in (model.given_mastered, model.given_not_mastered)
            for probability in (
                distribution.supports_mastery,
                distribution.supports_non_mastery,
            )
        )
        if any(
            value < self.observation_probability_floor
            or value > 1.0 - self.observation_probability_floor
            for value in probabilities
        ):
            raise ValueError("Observation probabilities violate the frozen floor")
        if tuple(item.condition for item in self.stress_conditions) != tuple(StressCondition):
            raise ValueError("Stress matrix must contain each condition once and in order")
        expected_stress = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.5, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 1.0),
        )
        actual_stress = tuple(
            (
                item.generator_reliability_scale,
                item.inversion_probability,
                item.missing_probability,
                item.contradictory_pair_probability,
            )
            for item in self.stress_conditions
        )
        if actual_stress != expected_stress:
            raise ValueError("Stress parameters differ from the frozen comparison matrix")
        if self.configuration_hash != model_content_hash(self, exclude={"configuration_hash"}):
            raise ValueError("Tracker-study configuration hash does not match its content")
        return self


class HandWorkedTrace(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.hand_worked_trace.v1"] = "tracker_study.hand_worked_trace.v1"
    configuration_hash: Sha256
    prior_mastery_probability: float = Field(gt=0.0, lt=1.0)
    channel: EvidenceChannel
    category: EvidenceCategory
    likelihood_if_mastered: float = Field(gt=0.0, lt=1.0)
    likelihood_if_not_mastered: float = Field(gt=0.0, lt=1.0)
    raw_log_likelihood_ratio: float
    ordinary_posterior: float = Field(gt=0.0, lt=1.0)
    trust_weight: float = Field(ge=0.0, le=1.0)
    clipping_kappa: float = Field(gt=0.0)
    bounded_log_odds_delta: float
    bounded_posterior: float = Field(gt=0.0, lt=1.0)
    ordinary_next_turn_probability: float = Field(gt=0.0, lt=1.0)
    bounded_next_turn_probability: float = Field(gt=0.0, lt=1.0)
    assumptions: tuple[str, ...] = Field(min_length=3)
    calculation_notes: tuple[str, ...] = Field(min_length=4)
    trace_hash: Sha256

    @model_validator(mode="after")
    def validate_trace(self) -> HandWorkedTrace:
        raw = math.log(self.likelihood_if_mastered / self.likelihood_if_not_mastered)
        prior_log_odds = math.log(
            self.prior_mastery_probability / (1.0 - self.prior_mastery_probability)
        )
        ordinary = _sigmoid(prior_log_odds + raw)
        bounded_delta = self.trust_weight * min(self.clipping_kappa, max(-self.clipping_kappa, raw))
        bounded = _sigmoid(prior_log_odds + bounded_delta)
        expected = (raw, ordinary, bounded_delta, bounded)
        actual = (
            self.raw_log_likelihood_ratio,
            self.ordinary_posterior,
            self.bounded_log_odds_delta,
            self.bounded_posterior,
        )
        if any(
            not math.isclose(left, right, abs_tol=1e-12)
            for left, right in zip(actual, expected, strict=True)
        ):
            raise ValueError("Hand-worked posterior calculations do not reconcile")
        if self.trace_hash != model_content_hash(self, exclude={"trace_hash"}):
            raise ValueError("Hand-worked trace hash does not match its content")
        return self


def load_tracker_study_configuration(path: Path) -> TrackerStudyConfiguration:
    return _load_yaml(path, TrackerStudyConfiguration)


def load_hand_worked_trace(path: Path, configuration: TrackerStudyConfiguration) -> HandWorkedTrace:
    trace = _load_yaml(path, HandWorkedTrace)
    if trace.configuration_hash != configuration.configuration_hash:
        raise TrackerStudyConfigurationError("Trace belongs to another study configuration")
    dynamics = configuration.latent_dynamics
    expected_ordinary_next = _transition(trace.ordinary_posterior, dynamics)
    expected_bounded_next = _transition(trace.bounded_posterior, dynamics)
    if not math.isclose(
        trace.ordinary_next_turn_probability, expected_ordinary_next, abs_tol=1e-12
    ) or not math.isclose(
        trace.bounded_next_turn_probability, expected_bounded_next, abs_tol=1e-12
    ):
        raise TrackerStudyConfigurationError("Trace transition calculations do not reconcile")
    return trace


def _transition(posterior: float, dynamics: LatentDynamics) -> float:
    return (
        posterior * (1.0 - dynamics.forget_probability)
        + (1.0 - posterior) * dynamics.learn_probability
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _load_yaml[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return model.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise TrackerStudyConfigurationError(
            f"Could not verify tracker-study input: {path}"
        ) from error
