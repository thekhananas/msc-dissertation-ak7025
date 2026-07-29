"""Recovery and idempotency tests for local tutoring sessions."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from socratic_tutor.contracts import CreateSessionRequest, SubmitTurnRequest
from socratic_tutor.sessions import IdempotencyConflictError, SessionService
from socratic_tutor.trajectories import (
    EventLogCorruptionError,
    JsonlEventStore,
    prepare_demo_event_log,
)


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


def test_reused_create_key_with_different_task_is_rejected(tmp_path: Path) -> None:
    service = SessionService(JsonlEventStore(tmp_path / "events.jsonl"))
    service.create(
        CreateSessionRequest(
            idempotency_key="create-task-0001",
            task_id="mutable-list-aliasing",
        )
    )

    with pytest.raises(IdempotencyConflictError, match="different task"):
        service.create(
            CreateSessionRequest(
                idempotency_key="create-task-0001",
                task_id="none-versus-falsy",
            )
        )


def test_session_uses_requested_task_and_recovers_it(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    service = SessionService(JsonlEventStore(log_path))

    created = service.create(
        CreateSessionRequest(
            idempotency_key="create-task-0002",
            task_id="none-versus-falsy",
        )
    )
    recovered = SessionService(JsonlEventStore(log_path)).get(created.session_id)

    assert created.task.task_id == "none-versus-falsy"
    assert recovered == created


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


def test_two_store_instances_append_one_valid_hash_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    first_service = SessionService(JsonlEventStore(log_path))
    second_service = SessionService(JsonlEventStore(log_path))

    first_service.create(CreateSessionRequest(idempotency_key="parallel-create-0001"))
    second_service.create(CreateSessionRequest(idempotency_key="parallel-create-0002"))

    events = JsonlEventStore(log_path).read_all()
    assert tuple(event.sequence for event in events) == (1, 2)
    assert events[1].previous_hash == events[0].event_hash


def test_corrupt_demo_log_is_preserved_before_fresh_start(tmp_path: Path) -> None:
    log_path = tmp_path / "demo-events.jsonl"
    corrupt_content = '{"complete":"but invalid"}\n'
    log_path.write_text(corrupt_content, encoding="utf-8")

    quarantine_path = prepare_demo_event_log(
        log_path,
        now=datetime(2026, 7, 25, 15, 45, tzinfo=UTC),
    )

    assert quarantine_path is not None
    assert quarantine_path == tmp_path / "demo-events.corrupt-20260725T154500000000Z.jsonl"
    assert quarantine_path.read_text(encoding="utf-8") == corrupt_content
    assert log_path.read_text(encoding="utf-8") == ""
    assert prepare_demo_event_log(log_path) is None
