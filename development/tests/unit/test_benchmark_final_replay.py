# pyright: reportPrivateUsage=false
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.final_replay import (
    FinalReplayError,
    HashComparison,
    _copy_scoring_sources,
    _extract_historical_source,
    _verify_historical_lock,
)
from socratic_tutor.benchmark.hashing import file_sha256

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_hash_comparison_rejects_a_replay_difference() -> None:
    with pytest.raises(ValidationError, match="differs from canonical"):
        HashComparison(
            artifact="primary_report",
            canonical_hash=HASH_A,
            replayed_hash=HASH_B,
            matches=True,
        )


def test_copies_only_inputs_needed_for_fresh_scoring(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write(source / "run_plan.json", "run")
    _write(source / "probe_summaries.json", "probes")
    _write(source / "decision" / "global.json", "decision")
    _write(source / "criterion" / "records" / "case.json", "criterion")
    _write(source / "analysis" / "primary.json", "must not be copied")
    for stem in ("condition_predictions", "criterion_records"):
        _write(source / "datasets" / f"{stem}.parquet", stem)
        _write(source / "datasets" / f"{stem}.publication.json", stem)

    _copy_scoring_sources(source, destination)

    assert (destination / "run_plan.json").read_text(encoding="utf-8") == "run"
    assert (destination / "decision" / "global.json").exists()
    assert (destination / "criterion" / "records" / "case.json").exists()
    assert not (destination / "analysis").exists()
    assert not (destination / "datasets" / "repeat_metrics.parquet").exists()


def test_rejects_a_historical_lock_that_differs_from_the_plan(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    lock = checkout / "development" / "pixi.lock"
    _write(lock, "historical lock")
    assert file_sha256(lock.read_bytes()) != HASH_A

    with pytest.raises(FinalReplayError, match="differs from its plan"):
        _verify_historical_lock(checkout, HASH_A)


def test_rejects_a_revision_that_is_not_a_git_object_id(tmp_path: Path) -> None:
    with pytest.raises(FinalReplayError, match="not a Git object ID"):
        _extract_historical_source(
            repository_root=tmp_path,
            revision="--exec=unsafe",
            destination=tmp_path / "checkout",
        )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
