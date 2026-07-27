"""Reviewed task-specific Socratic templates for offline tutoring turns."""

from socratic_tutor.contracts import PolicyDecision, TaskDefinition


def generate_prompt(
    task: TaskDefinition,
    decision: PolicyDecision,
    observation_number: int,
) -> str:
    """Render a reviewed prompt without repeating consecutive tutor moves."""

    if observation_number < 1:
        raise ValueError("Observation number must be at least one")
    variants = task.tutor_prompts.for_action(decision.action)
    return variants[(observation_number - 1) % len(variants)]
