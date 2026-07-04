"""Append-only, hash-chained JSONL events for local demo recovery."""

import hashlib
import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import Field

from socratic_tutor.contracts import ContractModel, SessionSnapshot, TurnRecord


class EventType(StrEnum):
    """Events sufficient to reconstruct the local demo."""

    SESSION_CREATED = "session.created"
    TURN_COMPLETED = "turn.completed"


class SessionCreatedPayload(ContractModel):
    """Initial session state persisted without private task rules."""

    snapshot: SessionSnapshot


class TurnCompletedPayload(ContractModel):
    """One immutable turn record."""

    turn: TurnRecord


class StoredEvent(ContractModel):
    """One verifiable JSONL event."""

    schema_version: int = 1
    sequence: int = Field(ge=1)
    event_type: EventType
    session_id: str
    idempotency_key: str
    occurred_at: datetime
    previous_hash: str | None
    payload: dict[str, Any]
    event_hash: str


class EventLogCorruptionError(RuntimeError):
    """Raised when a complete event line fails validation or hash checks."""


def _event_hash(event_data: dict[str, Any]) -> str:
    canonical = json.dumps(event_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hashable_data(event: StoredEvent) -> dict[str, Any]:
    return event.model_dump(mode="json", exclude={"event_hash"})


class JsonlEventStore:
    """Thread-safe local event store with startup verification."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._events = self._load()

    def read_all(self) -> tuple[StoredEvent, ...]:
        """Return the verified events in append order."""

        with self._lock:
            return tuple(self._events)

    def append(
        self,
        *,
        event_type: EventType,
        session_id: str,
        idempotency_key: str,
        payload: ContractModel,
    ) -> StoredEvent:
        """Persist and fsync one event before exposing it to the caller."""

        with self._lock:
            event_data: dict[str, Any] = {
                "schema_version": 1,
                "sequence": len(self._events) + 1,
                "event_type": event_type,
                "session_id": session_id,
                "idempotency_key": idempotency_key,
                "occurred_at": datetime.now(UTC),
                "previous_hash": self._events[-1].event_hash if self._events else None,
                "payload": payload.model_dump(mode="json"),
                "event_hash": "",
            }
            draft = StoredEvent.model_validate(event_data)
            event_data["event_hash"] = _event_hash(_hashable_data(draft))
            event = StoredEvent.model_validate(event_data)

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._events.append(event)
            return event

    def _load(self) -> list[StoredEvent]:
        if not self.path.exists():
            return []

        raw = self.path.read_text(encoding="utf-8")
        lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines = lines[:-1]
            self.path.write_text("".join(lines), encoding="utf-8")
        events: list[StoredEvent] = []
        previous_hash: str | None = None
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = StoredEvent.model_validate_json(line)
            except ValueError as error:
                raise EventLogCorruptionError(f"Invalid event at line {index + 1}") from error

            expected_sequence = len(events) + 1
            expected_hash = _event_hash(_hashable_data(event))
            if event.sequence != expected_sequence:
                raise EventLogCorruptionError(f"Unexpected sequence at line {index + 1}")
            if event.previous_hash != previous_hash:
                raise EventLogCorruptionError(f"Broken hash chain at line {index + 1}")
            if event.event_hash != expected_hash:
                raise EventLogCorruptionError(f"Invalid event hash at line {index + 1}")
            events.append(event)
            previous_hash = event.event_hash
        return events
