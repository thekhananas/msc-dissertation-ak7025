"""Exactly-once condition commitment reducer and filesystem tests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.benchmark.public import (
    BenchmarkCondition,
    BenchmarkSampleKey,
    CommitConflictError,
    CommitParityError,
    CommitStorageError,
    ConditionCommitSet,
    ConditionCommitState,
    DecisionPredictionRecord,
    FilesystemConditionCommitStore,
    IncompleteConditionSetError,
    SampleAlreadySealedError,
    append_condition_record,
    condition_commit_set_hash,
    prediction_record_hash,
    sample_key_hash,
    seal_condition_records,
)
from tests.unit.test_benchmark_predictions import prediction_records

SEALED_AT = datetime(2026, 8, 10, 8, 30, tzinfo=UTC)


def _key(records: tuple[DecisionPredictionRecord, ...]) -> BenchmarkSampleKey:
    return BenchmarkSampleKey.from_record(records[0])


def _rehashed_record(
    record: DecisionPredictionRecord,
    **changes: object,
) -> DecisionPredictionRecord:
    draft = record.model_copy(update=changes)
    payload = draft.model_dump(mode="python")
    payload["record_hash"] = prediction_record_hash(draft)
    return DecisionPredictionRecord.model_validate(payload)


def test_pure_reducer_is_idempotent_and_rejects_conflicts() -> None:
    records = prediction_records()
    state = ConditionCommitState(key=_key(records))

    once, first_result = append_condition_record(state, records[0])
    twice, retry_result = append_condition_record(once, records[0])

    assert twice is once
    assert retry_result == first_result
    conflicting = _rehashed_record(records[0], tracker_mastery_probability=0.123)
    with pytest.raises(CommitConflictError, match="another prediction hash"):
        append_condition_record(once, conflicting)


def test_reducer_requires_complete_ordered_parity_checked_records() -> None:
    records = prediction_records()
    key = _key(records)
    state = ConditionCommitState(key=key)
    for record in records[:-1]:
        state, _ = append_condition_record(state, record)

    with pytest.raises(IncompleteConditionSetError, match="missing"):
        seal_condition_records(state, sealed_at_utc=SEALED_AT)

    mismatched = _rehashed_record(records[1], public_record_hash="f" * 64)
    with pytest.raises(CommitParityError, match="public_record_hash"):
        append_condition_record(ConditionCommitState(key=key, records=(records[0],)), mismatched)

    with pytest.raises(ValidationError, match="canonically ordered"):
        ConditionCommitState(key=key, records=(records[1], records[0]))

    duplicate = _rehashed_record(records[0], tracker_mastery_probability=0.321)
    with pytest.raises(ValidationError, match="unique"):
        ConditionCommitState(key=key, records=(records[0], duplicate))


def test_commit_set_contract_rejects_reordering_and_unknown_condition() -> None:
    records = prediction_records()
    state = ConditionCommitState(key=_key(records), records=records)
    commit_set = seal_condition_records(state, sealed_at_utc=SEALED_AT)

    reordered = commit_set.model_dump(mode="python")
    reordered["conditions"] = tuple(reversed(commit_set.conditions))
    reordered["seal_hash"] = canonical_sha256(
        {name: value for name, value in reordered.items() if name != "seal_hash"}
    )
    with pytest.raises(ValidationError, match="ordered condition set"):
        ConditionCommitSet.model_validate(reordered)

    unknown = commit_set.model_dump(mode="python")
    unknown["conditions"] = (*commit_set.conditions[:-1], "unknown_condition")
    with pytest.raises(ValidationError, match="enum"):
        ConditionCommitSet.model_validate(unknown)


def test_filesystem_store_survives_restart_and_seals_exactly_four(tmp_path: Path) -> None:
    records = prediction_records()
    key = _key(records)
    store = FilesystemConditionCommitStore(tmp_path)

    first = store.append(records[0])
    assert store.append(records[0]) == first
    for record in records[1:]:
        store.append(record)

    restarted = FilesystemConditionCommitStore(tmp_path)
    commit_set = restarted.seal(key)
    assert commit_set.conditions == tuple(BenchmarkCondition)
    assert commit_set.prediction_hashes == tuple(record.record_hash for record in records)
    assert len(commit_set.prediction_hashes) == 4
    assert condition_commit_set_hash(commit_set) == commit_set.seal_hash
    assert restarted.seal(key) == commit_set
    assert FilesystemConditionCommitStore(tmp_path).get(key) == commit_set

    set_path = tmp_path / "sets" / f"{sample_key_hash(key)}.json"
    durable = ConditionCommitSet.model_validate_json(set_path.read_bytes())
    assert durable == commit_set

    with pytest.raises(SampleAlreadySealedError, match="sealed"):
        restarted.append(records[0])


def test_conflicting_retry_fails_before_and_after_restart(tmp_path: Path) -> None:
    record = prediction_records()[0]
    conflict = _rehashed_record(record, tracker_mastery_probability=0.123)
    store = FilesystemConditionCommitStore(tmp_path)
    store.append(record)

    with pytest.raises(CommitConflictError):
        store.append(conflict)
    with pytest.raises(CommitConflictError):
        FilesystemConditionCommitStore(tmp_path).append(conflict)


def test_store_rejects_incomplete_and_parity_mismatched_samples(tmp_path: Path) -> None:
    records = prediction_records()
    key = _key(records)
    store = FilesystemConditionCommitStore(tmp_path / "missing")
    store.append(records[0])
    with pytest.raises(IncompleteConditionSetError):
        store.seal(key)

    parity_store = FilesystemConditionCommitStore(tmp_path / "parity")
    parity_store.append(records[0])
    mismatch = _rehashed_record(records[1], public_evidence_hash="e" * 64)
    with pytest.raises(CommitParityError, match="public_evidence_hash"):
        parity_store.append(mismatch)


def test_restart_rejects_corrupt_or_missing_content_addressed_record(tmp_path: Path) -> None:
    records = prediction_records()
    store = FilesystemConditionCommitStore(tmp_path)
    for record in records:
        store.append(record)

    path = tmp_path / "records" / f"{records[2].record_hash}.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(CommitStorageError, match="Invalid commitment artifact"):
        FilesystemConditionCommitStore(tmp_path)
