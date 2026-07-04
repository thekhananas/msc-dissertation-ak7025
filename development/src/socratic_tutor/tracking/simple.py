"""Transparent baseline tracker for the first tutoring slice."""

from socratic_tutor.contracts import Evidence, EvidenceCategory, TrackerState

_EVIDENCE_DELTAS: dict[EvidenceCategory, float] = {
    EvidenceCategory.CORRECT: 0.20,
    EvidenceCategory.MISCONCEPTION: -0.20,
    EvidenceCategory.CONFLICTING: -0.10,
    EvidenceCategory.UNCERTAIN: 0.0,
    EvidenceCategory.EMPTY: 0.0,
}


def initial_tracker_state(concept: str) -> TrackerState:
    """Create the documented neutral prior used by the demo."""

    return TrackerState(concept=concept, mastery_probability=0.5, observations=0)


def update_tracker(state: TrackerState, evidence: Evidence) -> TrackerState:
    """Apply an additive evidence update and clamp the estimate to [0, 1]."""

    updated = min(1.0, max(0.0, state.mastery_probability + _EVIDENCE_DELTAS[evidence.category]))
    return TrackerState(
        concept=state.concept,
        mastery_probability=updated,
        observations=state.observations + 1,
        last_evidence=evidence.category,
    )
