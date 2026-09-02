"""Public contracts for the optional live evaluation illustration."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.contracts.models import ContractModel, EvidenceCategory


class LiveEvaluationPhase(StrEnum):
    """Visible stages of one non-canonical live run."""

    PREDICTIONS_COMMITTED = "predictions_committed"
    OUTCOME_REVEALED = "outcome_revealed"
    FAILED = "failed"


class LiveFailureStage(StrEnum):
    """Bounded failure locations safe to expose to the browser."""

    PUBLIC_GENERATION = "public_generation"
    EVIDENCE_GENERATION = "evidence_generation"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    EVIDENCE_EXECUTION = "evidence_execution"
    CRITERION_GENERATION = "criterion_generation"
    CRITERION_EXTRACTION = "criterion_extraction"
    CRITERION_EXECUTION = "criterion_execution"


class LiveEvaluationStatus(ContractModel):
    """Availability without credentials or provider internals."""

    enabled: bool
    available: bool
    case_id: str = Field(min_length=1)
    evaluation_model: str = Field(min_length=1)
    reason: str | None = None
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_reason_when_unavailable(self) -> LiveEvaluationStatus:
        if self.available and not self.enabled:
            raise ValueError("A disabled live illustration cannot be available")
        if self.available == (self.reason is not None):
            raise ValueError("Only an unavailable live illustration requires a reason")
        return self


class StartLiveEvaluationRequest(ContractModel):
    """Idempotent input for one allowlisted development case."""

    idempotency_key: str = Field(min_length=1, max_length=200)
    case_id: str = Field(min_length=1, max_length=200)


class LiveDialogueAssessmentView(ContractModel):
    """Transparent development-only reading of the visible answer."""

    category: EvidenceCategory
    method: Literal["transparent_development_rule"] = "transparent_development_rule"
    rationale: str = Field(min_length=1)


class LiveExecutionView(ContractModel):
    """Safe observable result from one remote Python execution."""

    passed_checks: int = Field(ge=0)
    failed_checks: int = Field(ge=0)
    passed_all_checks: bool
    sandbox_id: str | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> LiveExecutionView:
        if self.passed_checks + self.failed_checks == 0:
            raise ValueError("A live execution requires at least one check")
        if self.passed_all_checks is not (self.failed_checks == 0):
            raise ValueError("Execution result does not match its check counts")
        return self


class LivePredictionView(ContractModel):
    """One prediction fixed before the later task is requested."""

    condition: Literal["dialogue_only", "probe_informed"]
    label: str = Field(min_length=1)
    tracker_score: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    predicts_success: bool
    committed_at_utc: datetime
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_prediction(self) -> LivePredictionView:
        if self.predicts_success is not (self.tracker_score >= self.policy_threshold):
            raise ValueError("Live prediction does not match its threshold")
        return self


class LiveOutcomeView(ContractModel):
    """Later-task response and execution exposed only after commitment."""

    response: str = Field(min_length=1)
    execution: LiveExecutionView
    revealed_at_utc: datetime


class LiveFailureView(ContractModel):
    """A safe failure summary without raw transport or execution details."""

    stage: LiveFailureStage
    message: str = Field(min_length=1)


class LiveEvaluationSnapshot(ContractModel):
    """One in-memory, non-canonical live demonstration state."""

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str = Field(min_length=1)
    source: Literal["live_demo"] = "live_demo"
    canonical: Literal[False] = False
    phase: LiveEvaluationPhase
    evaluation_model: str = Field(min_length=1)
    public_response: str | None = None
    public_assessment: LiveDialogueAssessmentView | None = None
    evidence_response: str | None = None
    evidence_execution: LiveExecutionView | None = None
    predictions: tuple[LivePredictionView, ...] = ()
    commitment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    committed_at_utc: datetime | None = None
    outcome: LiveOutcomeView | None = None
    predictions_sealed_before_outcome_reveal: bool = False
    model_calls_made: int = Field(ge=0, le=3)
    sandbox_calls_made: int = Field(ge=0, le=2)
    provider_latency_ms: int = Field(ge=0)
    failure: LiveFailureView | None = None
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_phase(self) -> LiveEvaluationSnapshot:
        committed = self.phase in {
            LiveEvaluationPhase.PREDICTIONS_COMMITTED,
            LiveEvaluationPhase.OUTCOME_REVEALED,
        }
        if committed and (
            self.public_response is None
            or self.public_assessment is None
            or self.evidence_response is None
            or self.evidence_execution is None
            or len(self.predictions) != 2
            or self.commitment_hash is None
            or self.committed_at_utc is None
            or not self.predictions_sealed_before_outcome_reveal
            or self.failure is not None
        ):
            raise ValueError("Committed live state is incomplete")
        if self.phase is LiveEvaluationPhase.PREDICTIONS_COMMITTED and self.outcome is not None:
            raise ValueError("Committed live state cannot include the later outcome")
        if self.phase is LiveEvaluationPhase.OUTCOME_REVEALED and self.outcome is None:
            raise ValueError("Revealed live state requires the later outcome")
        if self.phase is LiveEvaluationPhase.FAILED and self.failure is None:
            raise ValueError("Failed live state requires a safe failure")
        return self
