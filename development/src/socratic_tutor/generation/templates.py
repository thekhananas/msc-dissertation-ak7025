"""Reviewed Socratic templates for offline tutoring turns."""

from socratic_tutor.contracts import PolicyDecision, TutorAction

_PROMPTS: dict[TutorAction, tuple[str, ...]] = {
    TutorAction.TRANSFER: (
        (
            "Good reasoning. If two names refer to one mutable object, what would happen if the "
            "change were made through the other name? Explain before trying it."
        ),
        (
            "Now compare `alias = numbers` with `alias = numbers.copy()`. What difference would "
            "you expect after appending through `alias`, and why?"
        ),
        (
            "Suppose a function receives `numbers` and mutates it. What would the caller observe? "
            "Connect your answer to the relationship between a name and an object."
        ),
        (
            "State a general rule for assignment and mutation of Python lists, then give one case "
            "where making a copy would be necessary."
        ),
    ),
    TutorAction.HINT: (
        (
            "Look closely at the assignment line. Does it construct a new list, or give another "
            "name to an existing object? Use that distinction to revise your prediction."
        ),
        (
            "Imagine checking `id(alias) == id(numbers)` immediately after the assignment. What "
            "would you expect, and what would that imply for the append?"
        ),
        (
            "Contrast the original assignment with `alias = numbers.copy()`. Which version creates "
            "two list objects? Use that comparison to trace the program again."
        ),
        (
            "Draw one object and the variable names that point to it after each line. At which "
            "line, if any, is a second list created?"
        ),
    ),
    TutorAction.CLARIFY: (
        (
            "Your response points in two directions. State whether a new list is created, then use "
            "that choice to make one consistent prediction."
        ),
        (
            "Separate your answer into two claims: how many list objects exist, and what gets "
            "mutated. Which pair of claims is consistent?"
        ),
        (
            "Choose one model: shared object or independent copy. Trace all four lines using only "
            "that model, then report the resulting output."
        ),
    ),
    TutorAction.PROBE: (
        (
            "Focus on `alias = numbers`. After that line, how many list objects do you think "
            "exist, and what makes you think so?"
        ),
        (
            "Which line in the program could create a new list? Identify it, or explain why none "
            "of the lines does so."
        ),
        (
            "Before predicting the output, describe what `append` changes: a variable name or the "
            "list object reached through that name?"
        ),
    ),
    TutorAction.ENCOURAGE: (
        (
            "Start with just the assignment line: does `alias = numbers` create a new list? A "
            "short answer is enough."
        ),
        "Try naming only the list objects that exist after the first two lines.",
        "Take one step: what does `append(3)` do to the list reached through `alias`?",
    ),
}


def generate_prompt(decision: PolicyDecision, observation_number: int) -> str:
    """Render a reviewed prompt without repeating consecutive tutor moves."""

    if observation_number < 1:
        raise ValueError("Observation number must be at least one")
    variants = _PROMPTS[decision.action]
    return variants[(observation_number - 1) % len(variants)]
