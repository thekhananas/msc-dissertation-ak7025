"""Auditable tutoring heuristic used before learned policies are introduced."""

from socratic_tutor.contracts import Evidence, EvidenceCategory, PolicyDecision, TutorAction

_ACTIONS: dict[EvidenceCategory, tuple[TutorAction, str]] = {
    EvidenceCategory.CORRECT: (
        TutorAction.TRANSFER,
        "Correct evidence supports moving to a nearby transfer question.",
    ),
    EvidenceCategory.MISCONCEPTION: (
        TutorAction.HINT,
        "A recognised misconception should receive a conceptual hint, not the answer.",
    ),
    EvidenceCategory.CONFLICTING: (
        TutorAction.CLARIFY,
        "Conflicting evidence should be resolved before increasing difficulty.",
    ),
    EvidenceCategory.UNCERTAIN: (
        TutorAction.PROBE,
        "Insufficient evidence calls for a focused diagnostic question.",
    ),
    EvidenceCategory.EMPTY: (
        TutorAction.ENCOURAGE,
        "An empty response calls for a smaller, low-pressure first step.",
    ),
}


def choose_action(evidence: Evidence) -> PolicyDecision:
    """Select one deterministic tutoring move from the evidence category."""

    action, rationale = _ACTIONS[evidence.category]
    return PolicyDecision(action=action, rationale=rationale)
