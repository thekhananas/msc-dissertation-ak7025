"""Public contracts shared by the deterministic tutor and demo application."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base model that rejects undeclared data and accidental mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceCategory(StrEnum):
    """Small, explicit evidence vocabulary used by the first vertical slice."""

    CORRECT = "correct"
    MISCONCEPTION = "misconception"
    CONFLICTING = "conflicting"
    UNCERTAIN = "uncertain"
    EMPTY = "empty"


class TutorAction(StrEnum):
    """Tutor moves available to the deterministic heuristic."""

    TRANSFER = "transfer"
    HINT = "hint"
    CLARIFY = "clarify"
    PROBE = "probe"
    ENCOURAGE = "encourage"


class EvidenceRules(ContractModel):
    """Reviewed string signals for an authored, non-LLM evidence classifier."""

    correct_output_markers: tuple[str, ...]
    correct_explanation_markers: tuple[str, ...]
    misconception_output_markers: tuple[str, ...]
    misconception_explanation_markers: tuple[str, ...]


class TutorPromptTemplates(ContractModel):
    """Task-specific Socratic prompts for each transparent policy action."""

    transfer: tuple[str, ...] = Field(min_length=1)
    hint: tuple[str, ...] = Field(min_length=1)
    clarify: tuple[str, ...] = Field(min_length=1)
    probe: tuple[str, ...] = Field(min_length=1)
    encourage: tuple[str, ...] = Field(min_length=1)

    def for_action(self, action: TutorAction) -> tuple[str, ...]:
        """Return the reviewed variants associated with one policy action."""

        return {
            TutorAction.TRANSFER: self.transfer,
            TutorAction.HINT: self.hint,
            TutorAction.CLARIFY: self.clarify,
            TutorAction.PROBE: self.probe,
            TutorAction.ENCOURAGE: self.encourage,
        }[action]


class TaskDefinition(ContractModel):
    """Full authored task, including rules that remain server-side."""

    schema_version: int = Field(ge=1)
    task_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    starter_code: str = Field(min_length=1)
    initial_prompt: str = Field(min_length=1)
    evidence_rules: EvidenceRules
    tutor_prompts: TutorPromptTemplates

    def public_view(self) -> "TaskView":
        """Return the task fields that are safe to show to a student."""

        return TaskView(
            task_id=self.task_id,
            version=self.version,
            title=self.title,
            concept=self.concept,
            instructions=self.instructions,
            starter_code=self.starter_code,
        )


class TaskView(ContractModel):
    """Student-facing task fields with evaluator rules removed."""

    task_id: str
    version: str
    title: str
    concept: str
    instructions: str
    starter_code: str


class StudentSubmission(ContractModel):
    """One natural-language response to the authored task."""

    response_text: str


class Evidence(ContractModel):
    """Deterministic interpretation of a student response."""

    category: EvidenceCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    observed_signals: tuple[str, ...] = ()


class TrackerState(ContractModel):
    """Simple, bounded estimate; it is not simulator truth or a human diagnosis."""

    concept: str
    mastery_probability: float = Field(ge=0.0, le=1.0)
    observations: int = Field(ge=0)
    last_evidence: EvidenceCategory | None = None


class PolicyDecision(ContractModel):
    """Inspectable output from the heuristic tutoring policy."""

    action: TutorAction
    rationale: str


class GuardrailResult(ContractModel):
    """Result of checking a generated tutor prompt for direct-answer leakage."""

    safe: bool
    output_prompt: str
    violations: tuple[str, ...] = ()


class TurnResult(ContractModel):
    """Complete, deterministic result of one tutoring turn."""

    evidence: Evidence
    tracker_before: TrackerState
    tracker_after: TrackerState
    decision: PolicyDecision
    next_prompt: str
    guardrail: GuardrailResult
