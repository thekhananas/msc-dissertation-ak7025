"""Network-free replay of one public benchmark example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from socratic_tutor.contracts.replay import (
    BenchmarkReplayArtifact,
    BenchmarkReplaySnapshot,
    ReplayPhase,
    StartBenchmarkReplayRequest,
)


class ReplayArtifactError(ValueError):
    """The public replay projection is unavailable or invalid."""


class ReplayNotStartedError(KeyError):
    """The requested replay has not been started in this process."""


class BenchmarkReplayService:
    """Reveal a checked-in outcome only after returning its fixed predictions."""

    def __init__(self, artifact_path: Path) -> None:
        self._artifact_path = artifact_path
        self._artifact: BenchmarkReplayArtifact | None = None
        self._started_replay_ids: set[str] = set()

    def start(self, request: StartBenchmarkReplayRequest) -> BenchmarkReplaySnapshot:
        artifact = self._load_artifact()
        replay_id = hashlib.sha256(
            f"{artifact.source_data_hash}:{request.idempotency_key}".encode()
        ).hexdigest()
        self._started_replay_ids.add(replay_id)
        return self._snapshot(replay_id, reveal_outcome=False)

    def reveal(self, replay_id: str) -> BenchmarkReplaySnapshot:
        if replay_id not in self._started_replay_ids:
            raise ReplayNotStartedError(replay_id)
        return self._snapshot(replay_id, reveal_outcome=True)

    def _load_artifact(self) -> BenchmarkReplayArtifact:
        if self._artifact is not None:
            return self._artifact
        try:
            raw = json.loads(self._artifact_path.read_text(encoding="utf-8"))
            self._artifact = BenchmarkReplayArtifact.model_validate(cast(object, raw))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ReplayArtifactError(
                f"Could not load benchmark replay artifact: {self._artifact_path}"
            ) from error
        return self._artifact

    def _snapshot(self, replay_id: str, *, reveal_outcome: bool) -> BenchmarkReplaySnapshot:
        artifact = self._load_artifact()
        return BenchmarkReplaySnapshot(
            replay_id=replay_id,
            phase=(
                ReplayPhase.OUTCOME_REVEALED
                if reveal_outcome
                else ReplayPhase.PREDICTIONS_COMMITTED
            ),
            source_data_hash=artifact.source_data_hash,
            source_manifest_hash=artifact.source_manifest_hash,
            benchmark_version=artifact.benchmark_version,
            case_id=artifact.case_id,
            evaluation_model=artifact.evaluation_model,
            public_summary=artifact.public_summary,
            predictions=artifact.predictions,
            outcome=artifact.outcome if reveal_outcome else None,
            claim_boundary=artifact.claim_boundary,
        )
