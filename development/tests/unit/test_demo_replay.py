"""Behavioural checks for the offline benchmark replay state transition."""

from pathlib import Path

import pytest

from socratic_tutor.contracts import ReplayPhase, StartBenchmarkReplayRequest
from socratic_tutor.demo_replay import BenchmarkReplayService, ReplayNotStartedError

WORKSPACE_ROOT = Path(__file__).parents[2]
REPLAY_ARTIFACT = WORKSPACE_ROOT / "data" / "demo" / "benchmark-replay-v1.json"


def test_replay_hides_outcome_until_fixed_predictions_are_returned() -> None:
    service = BenchmarkReplayService(REPLAY_ARTIFACT)

    with pytest.raises(ReplayNotStartedError):
        service.reveal("0" * 64)

    committed = service.start(StartBenchmarkReplayRequest(idempotency_key="demo-replay-1"))
    assert committed.phase is ReplayPhase.PREDICTIONS_COMMITTED
    assert committed.outcome is None
    assert len(committed.predictions) == 4

    revealed = service.reveal(committed.replay_id)
    assert revealed.phase is ReplayPhase.OUTCOME_REVEALED
    assert revealed.outcome is not None
    assert revealed.predictions == committed.predictions
    assert all(
        prediction.committed_at_utc < revealed.outcome.revealed_at_utc
        for prediction in revealed.predictions
    )
    assert revealed.model_calls_made == 0
    assert revealed.sandbox_calls_made == 0
    assert revealed.post_reveal_policy_calls_made == 0
    assert service.reveal(committed.replay_id) == revealed
