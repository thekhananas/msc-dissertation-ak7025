"""Contracts for demo sessions, persisted turns, and HTTP requests."""

from datetime import datetime

from pydantic import Field

from socratic_tutor.contracts.models import (
    ContractModel,
    Evidence,
    GuardrailResult,
    PolicyDecision,
    TaskView,
    TrackerState,
)


class CreateSessionRequest(ContractModel):
    """Idempotent request to start the fixed demonstration task."""

    idempotency_key: str = Field(min_length=8, max_length=128)


class SubmitTurnRequest(ContractModel):
    """Idempotent student response submitted to one session."""

    response_text: str = Field(max_length=10_000)
    idempotency_key: str = Field(min_length=8, max_length=128)


class TurnRecord(ContractModel):
    """Student-safe record of one completed tutoring turn."""

    turn_number: int = Field(ge=1)
    idempotency_key: str
    student_response: str
    evidence: Evidence
    tracker_before: TrackerState
    tracker_after: TrackerState
    decision: PolicyDecision
    tutor_prompt: str
    guardrail: GuardrailResult
    completed_at: datetime


class SessionSnapshot(ContractModel):
    """Complete public state required to render or recover a demo session."""

    session_id: str
    task: TaskView
    initial_prompt: str
    tracker: TrackerState
    turns: tuple[TurnRecord, ...] = ()
    created_at: datetime
    updated_at: datetime
