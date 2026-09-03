"""Application service for idempotent, recoverable tutoring sessions."""

from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from socratic_tutor.contracts import (
    CreateSessionRequest,
    SessionSnapshot,
    StudentSubmission,
    SubmitTurnRequest,
    TurnRecord,
)
from socratic_tutor.graph import run_graph_turn
from socratic_tutor.tasks import load_task
from socratic_tutor.tracking import initial_tracker_state
from socratic_tutor.trajectories import (
    EventType,
    JsonlEventStore,
    SessionCreatedPayload,
    TurnCompletedPayload,
)


class SessionNotFoundError(KeyError):
    """Raised when a client addresses an unknown session."""


class IdempotencyConflictError(RuntimeError):
    """Raised when one key is reused for a different request."""


class SessionService:
    """Coordinate graph turns and append-only local persistence."""

    def __init__(self, event_store: JsonlEventStore) -> None:
        self._event_store = event_store
        self._lock = RLock()
        self._sessions: dict[str, SessionSnapshot] = {}
        self._create_keys: dict[str, str] = {}
        self._recover()

    def create(self, request: CreateSessionRequest) -> SessionSnapshot:
        """Create one task session, or replay an exactly matching request."""

        with self._lock:
            prior_session_id = self._create_keys.get(request.idempotency_key)
            if prior_session_id is not None:
                existing = self._sessions[prior_session_id]
                if existing.task.task_id != request.task_id:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used for a different task"
                    )
                return existing

            task = load_task(request.task_id)
            now = datetime.now(UTC)
            snapshot = SessionSnapshot(
                session_id=str(uuid4()),
                task=task.public_view(),
                initial_prompt=task.initial_prompt,
                tracker=initial_tracker_state(task.concept),
                created_at=now,
                updated_at=now,
            )
            self._event_store.append(
                event_type=EventType.SESSION_CREATED,
                session_id=snapshot.session_id,
                idempotency_key=request.idempotency_key,
                payload=SessionCreatedPayload(snapshot=snapshot),
            )
            self._sessions[snapshot.session_id] = snapshot
            self._create_keys[request.idempotency_key] = snapshot.session_id
            return snapshot

    def get(self, session_id: str) -> SessionSnapshot:
        """Return one immutable session snapshot."""

        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise SessionNotFoundError(session_id) from error

    def submit(self, session_id: str, request: SubmitTurnRequest) -> SessionSnapshot:
        """Run and persist one graph turn, with request-level idempotency."""

        with self._lock:
            snapshot = self.get(session_id)
            for existing in snapshot.turns:
                if existing.idempotency_key != request.idempotency_key:
                    continue
                if existing.student_response != request.response_text:
                    raise IdempotencyConflictError(
                        "Idempotency key was already used with a different response"
                    )
                return snapshot

            task = load_task(snapshot.task.task_id)
            if task.version != snapshot.task.version:
                raise RuntimeError("Stored session task version is no longer available")

            result = run_graph_turn(
                task,
                StudentSubmission(response_text=request.response_text),
                snapshot.tracker,
                question_prompt=(
                    snapshot.initial_prompt
                    if not snapshot.turns
                    else snapshot.turns[-1].tutor_prompt
                ),
            )
            completed_at = datetime.now(UTC)
            turn = TurnRecord(
                turn_number=len(snapshot.turns) + 1,
                idempotency_key=request.idempotency_key,
                student_response=request.response_text,
                evidence=result.evidence,
                tracker_before=result.tracker_before,
                tracker_after=result.tracker_after,
                decision=result.decision,
                tutor_prompt=result.next_prompt,
                guardrail=result.guardrail,
                completed_at=completed_at,
            )
            self._event_store.append(
                event_type=EventType.TURN_COMPLETED,
                session_id=session_id,
                idempotency_key=request.idempotency_key,
                payload=TurnCompletedPayload(turn=turn),
            )
            updated = snapshot.model_copy(
                update={
                    "tracker": turn.tracker_after,
                    "turns": (*snapshot.turns, turn),
                    "updated_at": completed_at,
                }
            )
            self._sessions[session_id] = updated
            return updated

    def _recover(self) -> None:
        for event in self._event_store.read_all():
            if event.event_type is EventType.SESSION_CREATED:
                payload = SessionCreatedPayload.model_validate(event.payload)
                self._sessions[event.session_id] = payload.snapshot
                self._create_keys[event.idempotency_key] = event.session_id
                continue

            payload = TurnCompletedPayload.model_validate(event.payload)
            try:
                snapshot = self._sessions[event.session_id]
            except KeyError as error:
                raise RuntimeError("Turn event appeared before its session event") from error
            turn = payload.turn
            self._sessions[event.session_id] = snapshot.model_copy(
                update={
                    "tracker": turn.tracker_after,
                    "turns": (*snapshot.turns, turn),
                    "updated_at": turn.completed_at,
                }
            )
