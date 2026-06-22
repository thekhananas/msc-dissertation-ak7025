"""FastAPI health contract tests."""

import asyncio
from typing import Protocol, cast

import httpx

from apps.api.main import app


class _Response(Protocol):
    status_code: int

    def json(self) -> object: ...


async def _get_health() -> _Response:
    transport = httpx.ASGITransport(app=app)  # pyright: ignore[reportUnknownMemberType]
    async with httpx.AsyncClient(  # pyright: ignore[reportUnknownMemberType]
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return cast(
            _Response,
            await client.get("/api/health"),  # pyright: ignore[reportUnknownMemberType]
        )


def test_health_contract() -> None:
    response = asyncio.run(_get_health())
    payload = cast(dict[str, str], response.json())

    assert response.status_code == 200
    assert payload == {
        "status": "ok",
        "service": "socratic-tutor-api",
        "version": "0.1.0",
    }
