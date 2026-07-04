"""Recovery and idempotency tests for local tutoring sessions."""

import json
from pathlib import Path

import pytest

from socratic_tutor.contracts import CreateSessionRequest, SubmitTurnRequest
from socratic_tutor.sessions import IdempotencyConflictError, SessionService
from socratic_tutor.trajectories import EventLogCorruptionError, JsonlEventStore


def test_session_recovers_without_duplicate_turns(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    service = SessionService(JsonlEventStore(log_path))
    created = service.create(CreateSessionRequest(idempotency_key="create-0001"))
    request = SubmitTurnRequest(
        response_text="[1, 2, 3], because alias references the same list",
        idempotency_key="turn-0001",
    )

    completed = service.submit(created.session_id, request)
    replayed = service.submit(created.session_id, request)

    assert completed == replayed
    assert len(JsonlEventStore(log_path).read_all()) == 2

    recovered = SessionService(JsonlEventStore(log_path))
    assert recovered.get(created.session_id) == completed
    assert recovered.create(CreateSessionRequest(idempotency_key="create-0001")) == completed


def test_reused_turn_key_with_different_input_is_rejected(tmp_path: Path) -> None:
    service = SessionService(JsonlEventStore(tmp_path / "events.jsonl"))
    session = service.create(CreateSessionRequest(idempotency_key="create-0002"))
    service.submit(
        session.session_id,
        SubmitTurnRequest(response_text="first", idempotency_key="turn-0002"),
    )

    with pytest.raises(IdempotencyConflictError, match="different response"):
        service.submit(
            session.session_id,
            SubmitTurnRequest(response_text="second", idempotency_key="turn-0002"),
        )


def test_trailing_partial_event_is_removed_during_recovery(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    service = SessionService(JsonlEventStore(log_path))
    service.create(CreateSessionRequest(idempotency_key="create-0003"))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write('{"incomplete":')

    recovered = JsonlEventStore(log_path)

    assert len(recovered.read_all()) == 1
    assert log_path.read_text(encoding="utf-8").endswith("\n")


def test_complete_event_with_invalid_hash_is_rejected(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    service = SessionService(JsonlEventStore(log_path))
    service.create(CreateSessionRequest(idempotency_key="create-0004"))
    event = json.loads(log_path.read_text(encoding="utf-8"))
    event["event_hash"] = "0" * 64
    log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(EventLogCorruptionError, match="Invalid event hash"):
        JsonlEventStore(log_path)
