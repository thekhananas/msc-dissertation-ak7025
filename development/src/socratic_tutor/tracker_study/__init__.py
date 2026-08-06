"""Glass-box robustness study for evidence-aware mastery trackers."""

from socratic_tutor.tracker_study.config import (
    HandWorkedTrace,
    TrackerStudyConfiguration,
    load_hand_worked_trace,
    load_tracker_study_configuration,
)
from socratic_tutor.tracker_study.simulation import (
    SimulatedEpisode,
    SimulatedTurn,
    StressMatrix,
    StressMatrixManifest,
    StudySplit,
    TrackerTurnEstimate,
    publish_development_matrix,
    simulate_episode,
    simulate_stress_matrix,
    simulate_verified_development_matrix,
)
from socratic_tutor.tracker_study.trackers import (
    ConfiguredMasteryTracker,
    EvidenceObservation,
    MasteryTracker,
    TrackerUpdate,
    build_trackers,
    propagate_mastery_probability,
)

__all__ = [
    "ConfiguredMasteryTracker",
    "EvidenceObservation",
    "HandWorkedTrace",
    "MasteryTracker",
    "SimulatedEpisode",
    "SimulatedTurn",
    "StressMatrix",
    "StressMatrixManifest",
    "StudySplit",
    "TrackerStudyConfiguration",
    "TrackerTurnEstimate",
    "TrackerUpdate",
    "build_trackers",
    "load_hand_worked_trace",
    "load_tracker_study_configuration",
    "propagate_mastery_probability",
    "publish_development_matrix",
    "simulate_episode",
    "simulate_stress_matrix",
    "simulate_verified_development_matrix",
]
