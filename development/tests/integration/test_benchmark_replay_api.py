"""HTTP checks for prediction-before-outcome benchmark replay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx

from apps.api.main import create_app

WORKSPACE_ROOT = Path(__file__).parents[2]
REPLAY_ARTIFACT = WORKSPACE_ROOT / "data" / "demo" / "benchmark-replay-v1.json"


async def _exercise_replay_api(log_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    app = create_app(log_path, REPLAY_ARTIFACT)
    transport = httpx.ASGITransport(app=app)  # pyright: ignore[reportUnknownMemberType]
    async with httpx.AsyncClient(  # pyright: ignore[reportUnknownMemberType]
        transport=transport,
        base_url="http://testserver",
    ) as client:
        missing = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/benchmark-replays/{'0' * 64}/reveal"
        )
        assert missing.status_code == 404

        started_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            "/api/benchmark-replays",
            json={"idempotency_key": "api-replay-1"},
        )
        assert started_response.status_code == 201
        started = cast(dict[str, Any], started_response.json())
        assert started["phase"] == "predictions_committed"
        assert started["outcome"] is None
        assert "test_warning" not in json.dumps(started)

        revealed_response = await client.post(  # pyright: ignore[reportUnknownMemberType]
            f"/api/benchmark-replays/{started['replay_id']}/reveal"
        )
        assert revealed_response.status_code == 200
        revealed = cast(dict[str, Any], revealed_response.json())
        return started, revealed


def test_api_returns_committed_predictions_before_recorded_outcome(tmp_path: Path) -> None:
    started, revealed = asyncio.run(_exercise_replay_api(tmp_path / "events.jsonl"))

    assert revealed["phase"] == "outcome_revealed"
    assert revealed["outcome"]["passed_checks"] == 2
    assert revealed["predictions"] == started["predictions"]
    assert revealed["model_calls_made"] == 0
    assert revealed["sandbox_calls_made"] == 0
    assert revealed["post_reveal_policy_calls_made"] == 0
