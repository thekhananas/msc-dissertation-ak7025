"""Auditable tutoring heuristic used before learned policies are introduced."""

from socratic_tutor.contracts import Evidence, EvidenceCategory, PolicyDecision, TutorAction

_ACTIONS: dict[EvidenceCategory, tuple[TutorAction, str]] = {
    EvidenceCategory.CORRECT: (
        TutorAction.TRANSFER,
        "Correct evidence supports moving to a nearby transfer question.",
    ),
    EvidenceCategory.INCORRECT: (
        TutorAction.HINT,
        "Incorrect evidence should receive a conceptual hint without assuming a misconception.",
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
    EvidenceCategory.INCOMPLETE: (
        TutorAction.CLARIFY,
        "The response contains part of the answer, so the missing explanation should be clarified.",
    ),
    EvidenceCategory.EMPTY: (
        TutorAction.ENCOURAGE,
        "An empty response calls for a smaller, low-pressure first step.",
    ),
}


def choose_action(evidence: Evidence, *, observation_number: int = 1) -> PolicyDecision:
    """Select a tutoring move and recover from repeated unresolved evidence."""

    if evidence.category is EvidenceCategory.UNCERTAIN and observation_number > 3:
        return PolicyDecision(
            action=TutorAction.CLARIFY,
            rationale="Repeated uncertainty calls for a clear restatement of the key distinction.",
        )

    action, rationale = _ACTIONS[evidence.category]
    return PolicyDecision(action=action, rationale=rationale)
