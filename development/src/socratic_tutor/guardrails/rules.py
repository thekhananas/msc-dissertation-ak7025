"""Small deterministic leakage guardrail for reviewed tutor prompts."""

from socratic_tutor.contracts import GuardrailResult

_PROHIBITED_MARKERS = ("[1, 2, 3]", "[1,2,3]", "the answer is", "it prints")
_SAFE_FALLBACK = (
    "Trace the program one line at a time. Which objects exist after the assignment, and which "
    "names refer to them?"
)


def check_prompt(prompt: str) -> GuardrailResult:
    """Replace a prompt if it directly reveals the authored task answer."""

    normalized = prompt.casefold()
    violations = tuple(marker for marker in _PROHIBITED_MARKERS if marker in normalized)
    if violations:
        return GuardrailResult(
            safe=False,
            output_prompt=_SAFE_FALLBACK,
            violations=violations,
        )
    return GuardrailResult(safe=True, output_prompt=prompt)
