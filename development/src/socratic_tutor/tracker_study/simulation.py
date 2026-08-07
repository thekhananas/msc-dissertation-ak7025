"""Deterministic glass-box simulator for the tracker robustness study."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_json_bytes, canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.config import (
    EvidenceCategory,
    EvidenceChannel,
    StressCondition,
    StressSpecification,
    TrackerId,
    TrackerStudyConfiguration,
)
from socratic_tutor.tracker_study.trackers import (
    EvidenceObservation,
    build_trackers,
    propagate_mastery_probability,
)


class StudySplit(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"


class TrackerTurnEstimate(ContractModel):
    tracker_id: TrackerId
    prior_mastery_probability: float = Field(ge=0.0, le=1.0)
    posterior_mastery_probability: float = Field(ge=0.0, le=1.0)
    next_prior_mastery_probability: float = Field(ge=0.0, le=1.0)
    evidence_used_count: int = Field(ge=0, le=3)


class SimulatedTurn(ContractModel):
    turn_index: int = Field(ge=0)
    latent_mastery: bool
    observations: tuple[EvidenceObservation, ...]
    tracker_estimates: tuple[TrackerTurnEstimate, ...]

    @model_validator(mode="after")
    def validate_comparison_order(self) -> SimulatedTurn:
        if tuple(item.channel for item in self.observations) != tuple(EvidenceChannel):
            raise ValueError("Observations must contain every channel once and in frozen order")
        if tuple(item.tracker_id for item in self.tracker_estimates) != tuple(TrackerId):
            raise ValueError("Turn must contain every tracker once and in frozen order")
        return self


class SimulatedEpisode(ContractModel):
    split: StudySplit
    condition: StressCondition
    episode_index: int = Field(ge=0)
    latent_seed: int = Field(ge=0)
    observation_seed: int = Field(ge=0)
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    turns: tuple[SimulatedTurn, ...]


class StressMatrix(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.stress_matrix.v1"] = "tracker_study.stress_matrix.v1"
    study_id: str
    configuration_hash: Sha256
    split: StudySplit
    episodes_per_condition: int = Field(ge=1)
    turns_per_episode: int = Field(ge=1)
    episodes: tuple[SimulatedEpisode, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> StressMatrix:
        expected_count = len(StressCondition) * self.episodes_per_condition
        if len(self.episodes) != expected_count:
            raise ValueError("Stress matrix has the wrong episode count")
        keys = tuple((item.condition, item.episode_index) for item in self.episodes)
        expected_keys = tuple(
            (condition, episode_index)
            for condition in StressCondition
            for episode_index in range(self.episodes_per_condition)
        )
        if keys != expected_keys:
            raise ValueError("Stress matrix order or coverage differs from the frozen design")
        if any(item.split is not self.split for item in self.episodes):
            raise ValueError("Stress matrix mixes experiment splits")
        if any(len(item.turns) != self.turns_per_episode for item in self.episodes):
            raise ValueError("Episode has the wrong number of turns")
        _validate_matched_latent_paths(self.episodes, self.episodes_per_condition)
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


class StressMatrixManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.stress_matrix_manifest.v1"] = (
        "tracker_study.stress_matrix_manifest.v1"
    )
    study_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    configuration_hash: Sha256
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    claim_scope: Literal["glass_box_simulator_robustness_not_human_cognition"] = (
        "glass_box_simulator_robustness_not_human_cognition"
    )
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    trajectory_file: Literal["trajectories.jsonl"] = "trajectories.jsonl"
    trajectory_file_sha256: Sha256
    trajectory_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    matched_latent_paths_verified: Literal[True] = True
    random_generator: Literal["numpy.PCG64"] = "numpy.PCG64"
    condition_count: int = Field(ge=1)
    episodes_per_condition: int = Field(ge=1)
    episode_count: int = Field(ge=1)
    turns_per_episode: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    observation_count: int = Field(ge=1)
    missing_observation_count: int = Field(ge=0)
    tracker_estimate_count: int = Field(ge=1)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> StressMatrixManifest:
        if self.trajectory_content_hash != self.replay_content_hash:
            raise ValueError("Deterministic replay hash differs from the first simulation")
        if self.condition_count != len(StressCondition):
            raise ValueError("Manifest does not contain every frozen stress condition")
        if self.episode_count != self.condition_count * self.episodes_per_condition:
            raise ValueError("Manifest episode count is inconsistent")
        if self.turn_count != self.episode_count * self.turns_per_episode:
            raise ValueError("Manifest turn count is inconsistent")
        if self.observation_count != self.turn_count * len(EvidenceChannel):
            raise ValueError("Manifest observation count is inconsistent")
        if self.tracker_estimate_count != self.turn_count * len(TrackerId):
            raise ValueError("Manifest tracker-estimate count is inconsistent")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash != expected_hash:
            raise ValueError("Stress-matrix manifest hash does not match its content")
        return self


class CanonicalStressMatrixManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.canonical_stress_matrix_manifest.v1"] = (
        "tracker_study.canonical_stress_matrix_manifest.v1"
    )
    study_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    configuration_hash: Sha256
    canonical_execution_plan_hash: Sha256
    split: Literal[StudySplit.TEST] = StudySplit.TEST
    result_scope: Literal["canonical_glass_box_simulator_test"] = (
        "canonical_glass_box_simulator_test"
    )
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    trajectory_file: Literal["canonical_trajectories.jsonl"] = "canonical_trajectories.jsonl"
    trajectory_file_sha256: Sha256
    trajectory_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    matched_latent_paths_verified: Literal[True] = True
    random_generator: Literal["numpy.PCG64"] = "numpy.PCG64"
    canonical_test_run_limit: Literal[1] = 1
    condition_count: int = Field(ge=1)
    episodes_per_condition: int = Field(ge=1)
    episode_count: int = Field(ge=1)
    turns_per_episode: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    observation_count: int = Field(ge=1)
    missing_observation_count: int = Field(ge=0)
    tracker_estimate_count: int = Field(ge=1)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> CanonicalStressMatrixManifest:
        if self.trajectory_content_hash != self.replay_content_hash:
            raise ValueError("Canonical replay hash differs from the first simulation")
        if self.condition_count != len(StressCondition):
            raise ValueError("Canonical manifest does not contain every stress condition")
        if self.episode_count != self.condition_count * self.episodes_per_condition:
            raise ValueError("Canonical manifest episode count is inconsistent")
        if self.turn_count != self.episode_count * self.turns_per_episode:
            raise ValueError("Canonical manifest turn count is inconsistent")
        if self.observation_count != self.turn_count * len(EvidenceChannel):
            raise ValueError("Canonical manifest observation count is inconsistent")
        if self.tracker_estimate_count != self.turn_count * len(TrackerId):
            raise ValueError("Canonical manifest tracker-estimate count is inconsistent")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash != expected_hash:
            raise ValueError("Canonical stress-matrix manifest hash does not match its content")
        return self


def simulate_verified_development_matrix(
    configuration: TrackerStudyConfiguration,
    *,
    episodes_per_condition: int | None = None,
) -> tuple[StressMatrix, Sha256]:
    """Run the development matrix twice and require exact content agreement."""

    first = simulate_stress_matrix(
        configuration,
        split=StudySplit.DEVELOPMENT,
        episodes_per_condition=episodes_per_condition,
    )
    replay = simulate_stress_matrix(
        configuration,
        split=StudySplit.DEVELOPMENT,
        episodes_per_condition=episodes_per_condition,
    )
    if first.content_hash != replay.content_hash:
        raise RuntimeError("Deterministic stress-matrix replay produced different content")
    return first, replay.content_hash


def simulate_verified_canonical_matrix(
    configuration: TrackerStudyConfiguration,
) -> tuple[StressMatrix, Sha256]:
    """Run the one fixed test matrix twice and require exact content agreement."""

    first = simulate_stress_matrix(configuration, split=StudySplit.TEST)
    replay = simulate_stress_matrix(configuration, split=StudySplit.TEST)
    if first.content_hash != replay.content_hash:
        raise RuntimeError("Deterministic canonical replay produced different content")
    if first.episodes_per_condition != configuration.experiment.test_episodes_per_condition:
        raise RuntimeError("Canonical simulation did not use the frozen test episode count")
    return first, replay.content_hash


def publish_development_matrix(
    matrix: StressMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
) -> StressMatrixManifest:
    """Write restricted trajectories and their self-verifying immutable manifest."""

    if matrix.split is not StudySplit.DEVELOPMENT:
        raise ValueError("This publisher accepts development trajectories only")
    trajectory_bytes = b"".join(
        canonical_json_bytes(episode) + b"\n" for episode in matrix.episodes
    )
    trajectory_path = output_root / "trajectories.jsonl"
    turn_count = sum(len(episode.turns) for episode in matrix.episodes)
    missing_count = sum(
        observation.category is EvidenceCategory.MISSING
        for episode in matrix.episodes
        for turn in episode.turns
        for observation in turn.observations
    )
    content = {
        "study_id": matrix.study_id,
        "run_id": run_id,
        "code_revision": code_revision,
        "configuration_hash": matrix.configuration_hash,
        "trajectory_file_sha256": file_sha256(trajectory_bytes),
        "trajectory_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "condition_count": len(StressCondition),
        "episodes_per_condition": matrix.episodes_per_condition,
        "episode_count": len(matrix.episodes),
        "turns_per_episode": matrix.turns_per_episode,
        "turn_count": turn_count,
        "observation_count": turn_count * len(EvidenceChannel),
        "missing_observation_count": missing_count,
        "tracker_estimate_count": turn_count * len(TrackerId),
    }
    manifest_content = {
        "schema_version": 1,
        "schema_id": "tracker_study.stress_matrix_manifest.v1",
        "split": StudySplit.DEVELOPMENT,
        "claim_scope": "glass_box_simulator_robustness_not_human_cognition",
        "access_scope": "simulator_restricted",
        "trajectory_file": "trajectories.jsonl",
        "deterministic_replay_verified": True,
        "matched_latent_paths_verified": True,
        "random_generator": "numpy.PCG64",
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        **content,
    }
    manifest = StressMatrixManifest.model_validate(
        {
            **manifest_content,
            "manifest_hash": canonical_sha256(manifest_content),
        }
    )
    write_immutable_bytes(trajectory_path, trajectory_bytes)
    write_immutable_json(output_root / "stress_matrix_manifest.json", manifest)
    return manifest


def publish_canonical_matrix(
    matrix: StressMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
    configuration: TrackerStudyConfiguration,
    canonical_execution_plan_hash: Sha256,
) -> CanonicalStressMatrixManifest:
    """Publish the single fixed test matrix without weakening its claim boundary."""

    if matrix.split is not StudySplit.TEST:
        raise ValueError("The canonical publisher accepts test trajectories only")
    if matrix.episodes_per_condition != configuration.experiment.test_episodes_per_condition:
        raise ValueError("Canonical trajectories must use the frozen test episode count")
    if matrix.configuration_hash != configuration.configuration_hash:
        raise ValueError("Canonical trajectories belong to another configuration")
    trajectory_bytes = b"".join(
        canonical_json_bytes(episode) + b"\n" for episode in matrix.episodes
    )
    turn_count = sum(len(episode.turns) for episode in matrix.episodes)
    missing_count = sum(
        observation.category is EvidenceCategory.MISSING
        for episode in matrix.episodes
        for turn in episode.turns
        for observation in turn.observations
    )
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.canonical_stress_matrix_manifest.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "code_revision": code_revision,
        "configuration_hash": matrix.configuration_hash,
        "canonical_execution_plan_hash": canonical_execution_plan_hash,
        "split": StudySplit.TEST,
        "result_scope": "canonical_glass_box_simulator_test",
        "access_scope": "simulator_restricted",
        "trajectory_file": "canonical_trajectories.jsonl",
        "trajectory_file_sha256": file_sha256(trajectory_bytes),
        "trajectory_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "deterministic_replay_verified": True,
        "matched_latent_paths_verified": True,
        "random_generator": "numpy.PCG64",
        "canonical_test_run_limit": configuration.experiment.canonical_test_run_limit,
        "condition_count": len(StressCondition),
        "episodes_per_condition": matrix.episodes_per_condition,
        "episode_count": len(matrix.episodes),
        "turns_per_episode": matrix.turns_per_episode,
        "turn_count": turn_count,
        "observation_count": turn_count * len(EvidenceChannel),
        "missing_observation_count": missing_count,
        "tracker_estimate_count": turn_count * len(TrackerId),
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    manifest = CanonicalStressMatrixManifest.model_validate(
        {**content, "manifest_hash": canonical_sha256(content)}
    )
    write_immutable_bytes(output_root / "canonical_trajectories.jsonl", trajectory_bytes)
    write_immutable_json(output_root / "canonical_stress_matrix_manifest.json", manifest)
    return manifest


def simulate_stress_matrix(
    configuration: TrackerStudyConfiguration,
    *,
    split: StudySplit,
    episodes_per_condition: int | None = None,
) -> StressMatrix:
    """Run every frozen stress condition over paired latent trajectories."""

    configured_count = _configured_episode_count(configuration, split)
    episode_count = configured_count if episodes_per_condition is None else episodes_per_condition
    if episode_count < 1 or episode_count > configured_count:
        raise ValueError(
            f"Episode count must lie in [1, {configured_count}] for the {split.value} split"
        )
    episodes = tuple(
        simulate_episode(
            configuration,
            split=split,
            condition=condition,
            episode_index=episode_index,
        )
        for condition in StressCondition
        for episode_index in range(episode_count)
    )
    return StressMatrix(
        study_id=configuration.study_id,
        configuration_hash=configuration.configuration_hash,
        split=split,
        episodes_per_condition=episode_count,
        turns_per_episode=configuration.experiment.turns_per_episode,
        episodes=episodes,
    )


def simulate_episode(
    configuration: TrackerStudyConfiguration,
    *,
    split: StudySplit,
    condition: StressCondition,
    episode_index: int,
) -> SimulatedEpisode:
    """Simulate one episode without exposing latent truth to tracker update methods."""

    if episode_index < 0:
        raise ValueError("Episode index cannot be negative")
    split_seed = _split_seed(configuration, split)
    latent_seed = _derive_seed(configuration.seeds.root, split_seed, "latent", episode_index)
    observation_seed = _derive_seed(
        configuration.seeds.root,
        split_seed,
        "observations",
        episode_index,
    )
    latent_rng = np.random.Generator(np.random.PCG64(latent_seed))
    observation_rng = np.random.Generator(np.random.PCG64(observation_seed))
    turn_count = configuration.experiment.turns_per_episode
    channel_count = len(EvidenceChannel)
    support_draws = observation_rng.random((turn_count, channel_count))
    inversion_draws = observation_rng.random((turn_count, channel_count))
    missing_draws = observation_rng.random((turn_count, channel_count))
    contradiction_draws = observation_rng.random(turn_count)

    latent_mastery = bool(
        latent_rng.random() < configuration.latent_dynamics.initial_mastery_probability
    )
    trackers = build_trackers(configuration)
    tracker_priors = {
        tracker.tracker_id: configuration.latent_dynamics.initial_mastery_probability
        for tracker in trackers
    }
    stress = _stress_specification(configuration, condition)
    turns: list[SimulatedTurn] = []

    for turn_index in range(turn_count):
        observations = _sample_turn_observations(
            configuration,
            stress=stress,
            latent_mastery=latent_mastery,
            support_draws=support_draws[turn_index],
            inversion_draws=inversion_draws[turn_index],
            missing_draws=missing_draws[turn_index],
            contradiction_draw=float(contradiction_draws[turn_index]),
        )
        estimates: list[TrackerTurnEstimate] = []
        for tracker in trackers:
            prior = tracker_priors[tracker.tracker_id]
            belief = prior
            evidence_used_count = 0
            for observation in observations:
                update = tracker.update(belief, observation)
                evidence_used_count += int(update.evidence_used)
                belief = update.posterior_mastery_probability
            next_prior = propagate_mastery_probability(
                belief,
                configuration.latent_dynamics,
                configuration.trackers.probability_floor,
            )
            tracker_priors[tracker.tracker_id] = next_prior
            estimates.append(
                TrackerTurnEstimate(
                    tracker_id=tracker.tracker_id,
                    prior_mastery_probability=prior,
                    posterior_mastery_probability=belief,
                    next_prior_mastery_probability=next_prior,
                    evidence_used_count=evidence_used_count,
                )
            )
        turns.append(
            SimulatedTurn(
                turn_index=turn_index,
                latent_mastery=latent_mastery,
                observations=observations,
                tracker_estimates=tuple(estimates),
            )
        )
        latent_mastery = _sample_next_latent_state(
            latent_mastery,
            configuration,
            float(latent_rng.random()),
        )

    return SimulatedEpisode(
        split=split,
        condition=condition,
        episode_index=episode_index,
        latent_seed=latent_seed,
        observation_seed=observation_seed,
        access_scope="simulator_restricted",
        turns=tuple(turns),
    )


def _sample_turn_observations(
    configuration: TrackerStudyConfiguration,
    *,
    stress: StressSpecification,
    latent_mastery: bool,
    support_draws: np.ndarray,
    inversion_draws: np.ndarray,
    missing_draws: np.ndarray,
    contradiction_draw: float,
) -> tuple[EvidenceObservation, ...]:
    categories: dict[EvidenceChannel, EvidenceCategory] = {}
    models = {item.channel: item for item in configuration.observation_models}

    for channel_index, channel in enumerate(EvidenceChannel):
        model = models[channel]
        base_probability = (
            model.given_mastered.supports_mastery
            if latent_mastery
            else model.given_not_mastered.supports_mastery
        )
        probability = 0.5 + stress.generator_reliability_scale * (base_probability - 0.5)
        category = (
            EvidenceCategory.SUPPORTS_MASTERY
            if float(support_draws[channel_index]) < probability
            else EvidenceCategory.SUPPORTS_NON_MASTERY
        )
        if float(inversion_draws[channel_index]) < stress.inversion_probability:
            category = _opposite(category)
        categories[channel] = category

    public_channel, executable_channel = configuration.contradictory_channel_pair
    if contradiction_draw < stress.contradictory_pair_probability:
        categories[public_channel] = _opposite(categories[executable_channel])

    observations: list[EvidenceObservation] = []
    for channel_index, channel in enumerate(EvidenceChannel):
        model = models[channel]
        category = categories[channel]
        if float(missing_draws[channel_index]) < stress.missing_probability:
            category = EvidenceCategory.MISSING
            confidence = 0.5
        elif category is EvidenceCategory.SUPPORTS_MASTERY:
            confidence = model.confidence_supports_mastery
        else:
            confidence = model.confidence_supports_non_mastery
        observations.append(
            EvidenceObservation(
                channel=channel,
                category=category,
                confidence=confidence,
            )
        )
    return tuple(observations)


def _sample_next_latent_state(
    current: bool,
    configuration: TrackerStudyConfiguration,
    draw: float,
) -> bool:
    dynamics = configuration.latent_dynamics
    if current:
        return not draw < dynamics.forget_probability
    return draw < dynamics.learn_probability


def _opposite(category: EvidenceCategory) -> EvidenceCategory:
    if category is EvidenceCategory.SUPPORTS_MASTERY:
        return EvidenceCategory.SUPPORTS_NON_MASTERY
    if category is EvidenceCategory.SUPPORTS_NON_MASTERY:
        return EvidenceCategory.SUPPORTS_MASTERY
    raise ValueError("Missing evidence has no opposite category")


def _stress_specification(
    configuration: TrackerStudyConfiguration,
    condition: StressCondition,
) -> StressSpecification:
    return next(item for item in configuration.stress_conditions if item.condition is condition)


def _split_seed(configuration: TrackerStudyConfiguration, split: StudySplit) -> int:
    if split is StudySplit.DEVELOPMENT:
        return configuration.seeds.development
    return configuration.seeds.test


def _configured_episode_count(
    configuration: TrackerStudyConfiguration,
    split: StudySplit,
) -> int:
    if split is StudySplit.DEVELOPMENT:
        return configuration.experiment.development_episodes_per_condition
    return configuration.experiment.test_episodes_per_condition


def _derive_seed(root: int, split_seed: int, component: str, episode_index: int) -> int:
    digest = canonical_sha256(
        {
            "root": root,
            "split_seed": split_seed,
            "component": component,
            "episode_index": episode_index,
        }
    )
    return int(digest[:16], 16)


def _validate_matched_latent_paths(
    episodes: tuple[SimulatedEpisode, ...],
    episodes_per_condition: int,
) -> None:
    reference = episodes[:episodes_per_condition]
    reference_paths = {
        episode.episode_index: tuple(turn.latent_mastery for turn in episode.turns)
        for episode in reference
    }
    reference_seeds = {episode.episode_index: episode.latent_seed for episode in reference}
    for episode in episodes:
        path = tuple(turn.latent_mastery for turn in episode.turns)
        if path != reference_paths[episode.episode_index]:
            raise ValueError("Stress conditions do not share matched latent trajectories")
        if episode.latent_seed != reference_seeds[episode.episode_index]:
            raise ValueError("Stress conditions do not share matched latent seeds")
