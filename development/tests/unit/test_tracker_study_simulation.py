from __future__ import annotations

from pathlib import Path

from socratic_tutor.tracker_study import (
    StudySplit,
    load_tracker_study_configuration,
    publish_development_matrix,
    simulate_episode,
    simulate_verified_development_matrix,
)
from socratic_tutor.tracker_study.config import (
    EvidenceCategory,
    EvidenceChannel,
    StressCondition,
    TrackerId,
)

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")


def test_stress_matrix_is_exactly_reproducible_and_uses_matched_latent_paths() -> None:
    first, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=2,
    )

    assert first.content_hash == replay_hash
    for episode_index in range(2):
        matched = tuple(
            episode for episode in first.episodes if episode.episode_index == episode_index
        )
        latent_paths = {tuple(turn.latent_mastery for turn in episode.turns) for episode in matched}
        latent_seeds = {episode.latent_seed for episode in matched}
        observation_seeds = {episode.observation_seed for episode in matched}
        assert len(latent_paths) == 1
        assert len(latent_seeds) == 1
        assert len(observation_seeds) == 1


def test_development_matrix_artifacts_are_self_verifying_and_retry_safe(
    tmp_path: Path,
) -> None:
    matrix, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=1,
    )
    first = publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=tmp_path,
        run_id="tracker-development-test",
        code_revision="b8cada9",
    )
    retry = publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=tmp_path,
        run_id="tracker-development-test",
        code_revision="b8cada9",
    )

    assert first == retry
    assert first.deterministic_replay_verified
    assert first.matched_latent_paths_verified
    assert first.episode_count == len(StressCondition)
    assert first.turn_count == len(StressCondition) * first.turns_per_episode
    assert first.observation_count == first.turn_count * len(EvidenceChannel)
    assert first.tracker_estimate_count == first.turn_count * len(TrackerId)
    assert first.trajectory_content_hash == replay_hash
    assert (tmp_path / "trajectories.jsonl").is_file()
    assert (tmp_path / "stress_matrix_manifest.json").is_file()


def test_inverted_condition_flips_every_clean_observation() -> None:
    clean = simulate_episode(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        condition=StressCondition.CLEAN,
        episode_index=0,
    )
    inverted = simulate_episode(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        condition=StressCondition.INVERTED,
        episode_index=0,
    )

    assert tuple(turn.latent_mastery for turn in clean.turns) == tuple(
        turn.latent_mastery for turn in inverted.turns
    )
    for clean_turn, inverted_turn in zip(clean.turns, inverted.turns, strict=True):
        for clean_observation, inverted_observation in zip(
            clean_turn.observations,
            inverted_turn.observations,
            strict=True,
        ):
            assert clean_observation.channel is inverted_observation.channel
            assert clean_observation.category is not inverted_observation.category


def test_contradictory_condition_forces_public_opposite_to_executable() -> None:
    episode = simulate_episode(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        condition=StressCondition.CONTRADICTORY,
        episode_index=0,
    )

    for turn in episode.turns:
        observations = {item.channel: item for item in turn.observations}
        public = observations[EvidenceChannel.PUBLIC_ANSWER]
        executable = observations[EvidenceChannel.EXECUTABLE_PROBE]
        assert public.category is not executable.category
        assert public.category is not EvidenceCategory.MISSING
        assert executable.category is not EvidenceCategory.MISSING


def test_missing_condition_emits_neutral_records_and_counts_used_evidence() -> None:
    episode = simulate_episode(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        condition=StressCondition.MISSING,
        episode_index=0,
    )
    missing_count = 0

    for turn in episode.turns:
        for observation in turn.observations:
            if observation.category is not EvidenceCategory.MISSING:
                continue
            missing_count += 1
            assert observation.confidence == 0.5
        expected_used = sum(
            observation.category is not EvidenceCategory.MISSING
            for observation in turn.observations
        )
        assert all(
            estimate.evidence_used_count == expected_used for estimate in turn.tracker_estimates
        )

    assert 0 < missing_count < len(episode.turns) * len(EvidenceChannel)


def test_every_tracker_receives_the_same_ordered_observations_without_latent_truth() -> None:
    episode = simulate_episode(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        condition=StressCondition.CLEAN,
        episode_index=1,
    )

    for turn in episode.turns:
        assert tuple(item.channel for item in turn.observations) == tuple(EvidenceChannel)
        assert tuple(item.tracker_id for item in turn.tracker_estimates) == tuple(TrackerId)
        for estimate in turn.tracker_estimates:
            assert "latent_mastery" not in estimate.model_dump()
            assert estimate.evidence_used_count == len(EvidenceChannel)


def test_tracker_state_propagates_once_between_turns() -> None:
    episode = simulate_episode(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        condition=StressCondition.CLEAN,
        episode_index=2,
    )

    for current, following in zip(episode.turns[:-1], episode.turns[1:], strict=True):
        current_by_tracker = {item.tracker_id: item for item in current.tracker_estimates}
        following_by_tracker = {item.tracker_id: item for item in following.tracker_estimates}
        for tracker_id in TrackerId:
            assert (
                current_by_tracker[tracker_id].next_prior_mastery_probability
                == following_by_tracker[tracker_id].prior_mastery_probability
            )
