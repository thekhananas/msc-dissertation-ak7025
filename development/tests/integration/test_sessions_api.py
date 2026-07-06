"""HTTP contract tests for the local tutoring session flow."""

import asyncio
from pathlib import Path
from typing import Any, cast

import httpx

from apps.api.main import create_app


async def _exercise_session_api(
    log_path: Path,
) -> tuple[int, dict[str, Any], dict[str, Any], tuple[int, int, int], list[dict[str, Any]]]:
    app = create_app(log_path)
    transport = httpx.ASGITransport(app=app)  # pyright: ignore[reportUnknownMemberType]
    async with httpx.AsyncClient(  # pyright: ignore[reportUnknownMemberType]
        transport=transport,
        base_url="http://testserver",
    ) as client:
        tasks_response = await client.get("/api/tasks")  # pyright: ignore[reportUnknownMemberType]
        tasks = cast(list[dict[str, Any]], tasks_response.json())
        created_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/sessions",
            json={"idempotency_key": "create-api-0001"},
        )
        created = cast(dict[str, Any], created_response.json())
        turn_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/sessions/{created['session_id']}/turns",
            json={
                "response_text": "[1, 2, 3], because alias references the same list",
                "idempotency_key": "turn-api-0001",
            },
        )
        duplicate_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/sessions/{created['session_id']}/turns",
            json={
                "response_text": "[1, 2, 3], because alias references the same list",
                "idempotency_key": "turn-api-0001",
            },
        )
        conflict_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/sessions/{created['session_id']}/turns",
            json={
                "response_text": "a different response",
                "idempotency_key": "turn-api-0001",
            },
        )
        missing_response = await client.get(  # pyright: ignore[reportUnknownMemberType]
            "/api/sessions/not-a-session"
        )
        invalid_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/sessions/{created['session_id']}/turns",
            json={"response_text": "missing idempotency key"},
        )
        return (
            created_response.status_code,
            cast(dict[str, Any], turn_response.json()),
            cast(dict[str, Any], duplicate_response.json()),
            (
                conflict_response.status_code,
                missing_response.status_code,
                invalid_response.status_code,
            ),
            tasks,
        )


def test_complete_session_api_flow_is_idempotent(tmp_path: Path) -> None:
    status_code, completed, duplicate, error_statuses, tasks = asyncio.run(
        _exercise_session_api(tmp_path / "events.jsonl")
    )

    assert status_code == 201
    assert completed == duplicate
    turns = cast(list[dict[str, Any]], completed["turns"])
    assert len(turns) == 1
    assert turns[0]["evidence"]["category"] == "correct"
    assert turns[0]["decision"]["action"] == "transfer"
    assert completed["tracker"]["mastery_probability"] == 0.7
    assert error_statuses == (409, 404, 422)
    assert [task["task_id"] for task in tasks] == [
        "mutable-list-aliasing",
        "none-versus-falsy",
    ]


async def _exercise_second_task(log_path: Path) -> tuple[dict[str, Any], int]:
    app = create_app(log_path)
    transport = httpx.ASGITransport(app=app)  # pyright: ignore[reportUnknownMemberType]
    async with httpx.AsyncClient(  # pyright: ignore[reportUnknownMemberType]
        transport=transport,
        base_url="http://testserver",
    ) as client:
        created_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/sessions",
            json={
                "idempotency_key": "create-api-none-0001",
                "task_id": "none-versus-falsy",
            },
        )
        created = cast(dict[str, Any], created_response.json())
        turn_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/sessions/{created['session_id']}/turns",
            json={
                "response_text": (
                    "It prints score=0 and missing because score is None only for the second call."
                ),
                "idempotency_key": "turn-api-none-0001",
            },
        )
        missing_task_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/sessions",
            json={
                "idempotency_key": "create-api-none-0002",
                "task_id": "not-a-task",
            },
        )
        return cast(dict[str, Any], turn_response.json()), missing_task_response.status_code


def test_second_task_uses_its_own_adaptive_path(tmp_path: Path) -> None:
    completed, missing_task_status = asyncio.run(
        _exercise_second_task(tmp_path / "none-events.jsonl")
    )

    assert completed["task"]["task_id"] == "none-versus-falsy"
    turns = cast(list[dict[str, Any]], completed["turns"])
    assert turns[0]["evidence"]["category"] == "correct"
    assert turns[0]["decision"]["action"] == "transfer"
    assert "empty string" in turns[0]["tutor_prompt"]
    assert missing_task_status == 404
