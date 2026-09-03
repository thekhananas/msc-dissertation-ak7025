"""Reviewed task-specific Socratic templates for offline tutoring turns."""

from socratic_tutor.contracts import EvidenceCategory, PolicyDecision, TaskDefinition


def generate_prompt(
    task: TaskDefinition,
    decision: PolicyDecision,
    observation_number: int,
    evidence_category: EvidenceCategory | None = None,
    question_prompt: str | None = None,
) -> str:
    """Render a reviewed prompt without repeating consecutive tutor moves."""

    if observation_number < 1:
        raise ValueError("Observation number must be at least one")
    if evidence_category is EvidenceCategory.INCOMPLETE:
        variants = task.tutor_prompts.incomplete
    elif question_prompt is not None:
        follow_up_rule = next(
            (
                rule
                for rule in task.follow_up_evidence_rules
                if any(
                    marker.casefold() in question_prompt.casefold()
                    for marker in rule.question_prompt_markers
                )
            ),
            None,
        )
        if follow_up_rule is not None and evidence_category is not EvidenceCategory.CORRECT:
            return follow_up_rule.follow_up_prompt
        variants = task.tutor_prompts.for_action(decision.action)
    else:
        variants = task.tutor_prompts.for_action(decision.action)
    return variants[(observation_number - 1) % len(variants)]
