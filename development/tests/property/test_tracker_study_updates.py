from __future__ import annotations

import math
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from socratic_tutor.tracker_study import (
    ConfiguredMasteryTracker,
    EvidenceObservation,
    build_trackers,
    load_hand_worked_trace,
    load_tracker_study_configuration,
    propagate_mastery_probability,
)
from socratic_tutor.tracker_study.config import (
    EvidenceCategory,
    EvidenceChannel,
    TrackerId,
)
from socratic_tutor.tracker_study.trackers import posterior_from_log_odds_delta

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
TRACE = load_hand_worked_trace(
    ROOT / "data" / "tracker-study" / "v1" / "hand-worked-trace.yaml",
    CONFIGURATION,
)
MATCHED_INPUT_EXPECTATIONS = {
    TrackerId.LAST_OBSERVATION: 0.999999,
    TrackerId.HARD_BKT: 0.7391304347826086,
    TrackerId.LEGACY_FRACTIONAL: 0.7605461050711252,
    TrackerId.CHANNEL_AWARE: 0.8823529411764706,
    TrackerId.BOUNDED_CHANNEL_AWARE: 0.8013152260326584,
}


@given(
    prior=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    channel=st.sampled_from(tuple(EvidenceChannel)),
    category=st.sampled_from(
        (
            EvidenceCategory.SUPPORTS_MASTERY,
            EvidenceCategory.SUPPORTS_NON_MASTERY,
        )
    ),
    confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
)
def test_every_tracker_returns_a_bounded_probability(
    prior: float,
    channel: EvidenceChannel,
    category: EvidenceCategory,
    confidence: float,
) -> None:
    observation = EvidenceObservation(channel=channel, category=category, confidence=confidence)

    for tracker in build_trackers(CONFIGURATION):
        update = tracker.update(prior, observation)
        assert 0.0 <= update.posterior_mastery_probability <= 1.0
        assert math.isfinite(update.posterior_mastery_probability)


@given(prior=st.floats(min_value=0.001, max_value=0.999, allow_nan=False))
def test_uninformative_likelihood_causes_no_correction(prior: float) -> None:
    floor = CONFIGURATION.trackers.probability_floor
    posterior = posterior_from_log_odds_delta(prior, 0.0, floor)
    assert math.isclose(posterior, prior, abs_tol=1e-12)


@given(prior=st.floats(min_value=0.001, max_value=0.999, allow_nan=False))
def test_informative_channel_evidence_moves_belief_in_expected_direction(
    prior: float,
) -> None:
    tracker = ConfiguredMasteryTracker(TrackerId.CHANNEL_AWARE, CONFIGURATION)
    supports = EvidenceObservation(
        channel=EvidenceChannel.EXECUTABLE_PROBE,
        category=EvidenceCategory.SUPPORTS_MASTERY,
        confidence=0.95,
    )
    opposes = supports.model_copy(update={"category": EvidenceCategory.SUPPORTS_NON_MASTERY})

    assert tracker.update(prior, supports).posterior_mastery_probability > prior
    assert tracker.update(prior, opposes).posterior_mastery_probability < prior


@given(prior=st.floats(min_value=0.001, max_value=0.999, allow_nan=False))
def test_bounded_tracker_limits_each_log_odds_change(prior: float) -> None:
    tracker = ConfiguredMasteryTracker(TrackerId.BOUNDED_CHANNEL_AWARE, CONFIGURATION)
    observation = EvidenceObservation(
        channel=EvidenceChannel.EXECUTABLE_PROBE,
        category=EvidenceCategory.SUPPORTS_MASTERY,
        confidence=0.95,
    )

    update = tracker.update(prior, observation)
    assert update.applied_log_odds_delta is not None
    expected_bound = (
        CONFIGURATION.trackers.channel_trust_weights[observation.channel]
        * CONFIGURATION.trackers.bounded_log_likelihood_kappa
    )
    assert abs(update.applied_log_odds_delta) <= expected_bound + 1e-12


def test_unit_trust_without_clipping_recovers_channel_aware_bayes() -> None:
    observation = EvidenceObservation(
        channel=EvidenceChannel.PUBLIC_ANSWER,
        category=EvidenceCategory.SUPPORTS_NON_MASTERY,
        confidence=0.70,
    )
    ordinary = ConfiguredMasteryTracker(TrackerId.CHANNEL_AWARE, CONFIGURATION)
    recovered = ConfiguredMasteryTracker(
        TrackerId.BOUNDED_CHANNEL_AWARE,
        CONFIGURATION,
        trust_weight_override=1.0,
        clipping_disabled=True,
    )

    ordinary_update = ordinary.update(0.63, observation)
    recovered_update = recovered.update(0.63, observation)
    assert math.isclose(
        ordinary_update.posterior_mastery_probability,
        recovered_update.posterior_mastery_probability,
        abs_tol=1e-12,
    )


def test_all_trackers_receive_the_same_record_and_missing_evidence_is_ignored() -> None:
    trackers = build_trackers(CONFIGURATION)
    assert tuple(tracker.tracker_id for tracker in trackers) == tuple(TrackerId)
    observation = EvidenceObservation(
        channel=EvidenceChannel.SELF_EXPLANATION,
        category=EvidenceCategory.MISSING,
        confidence=0.5,
    )

    updates = tuple(tracker.update(0.4, observation) for tracker in trackers)
    assert all(not update.evidence_used for update in updates)
    assert all(
        math.isclose(update.posterior_mastery_probability, 0.4, abs_tol=1e-12) for update in updates
    )


def test_all_five_baselines_match_the_fixed_input_fixture() -> None:
    observation = EvidenceObservation(
        channel=EvidenceChannel.EXECUTABLE_PROBE,
        category=EvidenceCategory.SUPPORTS_MASTERY,
        confidence=0.95,
    )

    updates = {
        tracker.tracker_id: tracker.update(0.4, observation)
        for tracker in build_trackers(CONFIGURATION)
    }

    assert set(updates) == set(TrackerId)
    for tracker_id, expected in MATCHED_INPUT_EXPECTATIONS.items():
        assert math.isclose(
            updates[tracker_id].posterior_mastery_probability,
            expected,
            abs_tol=1e-12,
        )


def test_equation_code_matches_the_frozen_hand_worked_trace() -> None:
    observation = EvidenceObservation(
        channel=TRACE.channel,
        category=TRACE.category,
        confidence=0.95,
    )
    ordinary = ConfiguredMasteryTracker(TrackerId.CHANNEL_AWARE, CONFIGURATION).update(
        TRACE.prior_mastery_probability, observation
    )
    bounded = ConfiguredMasteryTracker(TrackerId.BOUNDED_CHANNEL_AWARE, CONFIGURATION).update(
        TRACE.prior_mastery_probability, observation
    )
    floor = CONFIGURATION.trackers.probability_floor

    assert math.isclose(
        ordinary.posterior_mastery_probability, TRACE.ordinary_posterior, abs_tol=1e-12
    )
    assert math.isclose(
        bounded.posterior_mastery_probability, TRACE.bounded_posterior, abs_tol=1e-12
    )
    assert math.isclose(
        propagate_mastery_probability(
            ordinary.posterior_mastery_probability,
            CONFIGURATION.latent_dynamics,
            floor,
        ),
        TRACE.ordinary_next_turn_probability,
        abs_tol=1e-12,
    )
    assert math.isclose(
        propagate_mastery_probability(
            bounded.posterior_mastery_probability,
            CONFIGURATION.latent_dynamics,
            floor,
        ),
        TRACE.bounded_next_turn_probability,
        abs_tol=1e-12,
    )
