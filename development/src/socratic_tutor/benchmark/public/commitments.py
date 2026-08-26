"""Exactly-once, pre-criterion commitments for paired benchmark predictions."""

import fcntl
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public.models import EXPECTED_CONDITIONS, BenchmarkCondition
from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    prediction_record_hash,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.filesystem import fsync_directory


class CommitmentError(ValueError):
    """A condition commitment violates identity or completeness rules."""


class CommitConflictError(CommitmentError):
    """The same sample condition was assigned two prediction hashes."""


class CommitParityError(CommitmentError):
    """Predictions in one paired sample do not share required inputs."""


class IncompleteConditionSetError(CommitmentError):
    """A sample does not contain the exact ordered condition set."""


class SampleAlreadySealedError(CommitmentError):
    """A caller attempted to append to an immutable sample."""


class CommitStorageError(RuntimeError):
    """A durable commitment artifact is missing, corrupt, or inconsistent."""


class BenchmarkSampleKey(ContractModel):
    """Stable identity for one model sample across all four conditions."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.sample_key.v1"] = "benchmark.sample_key.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    model_route_id: str = Field(min_length=1)

    @classmethod
    def from_record(cls, record: DecisionPredictionRecord) -> "BenchmarkSampleKey":
        """Derive identity only from validated prediction fields."""

        return cls(
            benchmark_version=record.benchmark_version,
            run_id=record.run_id,
            case_id=record.case_id,
            sample_id=record.sample_id,
            model_route_id=record.model_route_id,
        )


def sample_key_hash(key: BenchmarkSampleKey) -> Sha256:
    """Return the path-safe content digest for a sample identity."""

    return model_content_hash(key)


class AppendResult(ContractModel):
    """Stable result returned for both an initial append and its retry."""

    key: BenchmarkSampleKey
    condition: BenchmarkCondition
    record_hash: Sha256


class ConditionCommitState(ContractModel):
    """Immutable in-memory state used by the append and seal reducer."""

    key: BenchmarkSampleKey
    records: tuple[DecisionPredictionRecord, ...] = ()

    @model_validator(mode="after")
    def validate_records(self) -> "ConditionCommitState":
        _validate_record_sequence(self.key, self.records)
        return self


class _ConditionCommitSetContent(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.condition_commit_set.v1"] = "benchmark.condition_commit_set.v1"
    key: BenchmarkSampleKey
    conditions: tuple[BenchmarkCondition, ...]
    prediction_hashes: tuple[Sha256, ...]
    case_content_hash: Sha256
    initial_state_hash: Sha256
    public_record_hash: Sha256
    public_evidence_hash: Sha256
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    observation_schema: str = Field(min_length=1)
    sealed_at_utc: datetime

    @model_validator(mode="after")
    def validate_complete_set(self) -> "_ConditionCommitSetContent":
        if self.conditions != EXPECTED_CONDITIONS:
            raise ValueError("Commit set must contain the exact ordered condition set")
        if len(self.prediction_hashes) != len(EXPECTED_CONDITIONS):
            raise ValueError("Commit set must contain exactly four prediction hashes")
        if len(set(self.prediction_hashes)) != len(self.prediction_hashes):
            raise ValueError("Commit set prediction hashes must be unique")
        _require_utc(self.sealed_at_utc, field_name="Seal time")
        return self


class ConditionCommitSet(_ConditionCommitSetContent):
    """Immutable publication proving four predictions preceded criterion access."""

    seal_hash: Sha256

    @model_validator(mode="after")
    def validate_seal_hash(self) -> "ConditionCommitSet":
        if self.seal_hash != condition_commit_set_hash(self):
            raise ValueError("Condition commit seal hash does not match its content")
        return self


def condition_commit_set_hash(commit_set: ConditionCommitSet) -> Sha256:
    """Hash a sealed set without its self-referential digest."""

    return model_content_hash(commit_set, exclude={"seal_hash"})


def append_condition_record(
    state: ConditionCommitState,
    record: DecisionPredictionRecord,
) -> tuple[ConditionCommitState, AppendResult]:
    """Apply one idempotent append to immutable condition state."""

    _validate_prediction(record)
    record_key = BenchmarkSampleKey.from_record(record)
    if record_key != state.key:
        raise CommitParityError("Prediction does not match the sample key")

    existing = next((item for item in state.records if item.condition is record.condition), None)
    result = AppendResult(
        key=state.key,
        condition=record.condition,
        record_hash=record.record_hash,
    )
    if existing is not None:
        if existing.record_hash != record.record_hash:
            raise CommitConflictError(
                f"Condition {record.condition.value} already has another prediction hash"
            )
        return state, result

    records = tuple(
        sorted(
            (*state.records, record),
            key=lambda item: EXPECTED_CONDITIONS.index(item.condition),
        )
    )
    _validate_record_sequence(state.key, records)
    return ConditionCommitState(key=state.key, records=records), result


def seal_condition_records(
    state: ConditionCommitState,
    *,
    sealed_at_utc: datetime,
) -> ConditionCommitSet:
    """Seal a complete state into an immutable, content-addressed set."""

    _validate_record_sequence(state.key, state.records)
    conditions = tuple(record.condition for record in state.records)
    if conditions != EXPECTED_CONDITIONS:
        missing = tuple(
            condition.value for condition in EXPECTED_CONDITIONS if condition not in conditions
        )
        raise IncompleteConditionSetError(
            f"Cannot seal without the exact condition set; missing={missing}"
        )
    _require_utc(sealed_at_utc, field_name="Seal time")

    first = state.records[0]
    content = _ConditionCommitSetContent(
        key=state.key,
        conditions=conditions,
        prediction_hashes=tuple(record.record_hash for record in state.records),
        case_content_hash=first.case_content_hash,
        initial_state_hash=first.initial_state_hash,
        public_record_hash=first.public_record_hash,
        public_evidence_hash=first.public_evidence_hash,
        tracker_version=first.tracker_version,
        policy_version=first.policy_version,
        observation_schema=first.observation_schema,
        sealed_at_utc=sealed_at_utc,
    )
    return ConditionCommitSet.model_validate(
        {
            **content.model_dump(mode="python"),
            "seal_hash": model_content_hash(content),
        }
    )


class ConditionCommitStore(Protocol):
    """Storage contract for pre-criterion sample commitments."""

    def append(self, record: DecisionPredictionRecord) -> AppendResult: ...

    def seal(self, key: BenchmarkSampleKey) -> ConditionCommitSet: ...

    def get(self, key: BenchmarkSampleKey) -> ConditionCommitSet | None: ...

    def get_records(
        self,
        key: BenchmarkSampleKey,
    ) -> tuple[DecisionPredictionRecord, ...]: ...

    def get_staged_records(
        self,
        key: BenchmarkSampleKey,
    ) -> tuple[DecisionPredictionRecord, ...]: ...

    def sealed_keys(self) -> tuple[BenchmarkSampleKey, ...]: ...


class _ConditionRecordRef(ContractModel):
    condition: BenchmarkCondition
    record_hash: Sha256


class _StagingIndexContent(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.condition_staging.v1"] = "benchmark.condition_staging.v1"
    key: BenchmarkSampleKey
    records: tuple[_ConditionRecordRef, ...] = Field(min_length=1)
    updated_at_utc: datetime

    @model_validator(mode="after")
    def validate_index(self) -> "_StagingIndexContent":
        conditions = tuple(item.condition for item in self.records)
        expected_order = tuple(
            condition for condition in EXPECTED_CONDITIONS if condition in conditions
        )
        if conditions != expected_order or len(set(conditions)) != len(conditions):
            raise ValueError("Staging conditions must be unique and canonically ordered")
        _require_utc(self.updated_at_utc, field_name="Staging update time")
        return self


class _StagingIndex(_StagingIndexContent):
    state_hash: Sha256

    @model_validator(mode="after")
    def validate_state_hash(self) -> "_StagingIndex":
        if self.state_hash != model_content_hash(self, exclude={"state_hash"}):
            raise ValueError("Staging index hash does not match its content")
        return self


class FilesystemConditionCommitStore:
    """Atomic filesystem implementation of per-sample prediction commitments."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = root
        self._records_dir = root / "records"
        self._staging_dir = root / "staging"
        self._sets_dir = root / "sets"
        self._locks_dir = root / "locks"
        self._clock = clock or _utc_now
        self._lock = RLock()
        for directory in (
            self._records_dir,
            self._staging_dir,
            self._sets_dir,
            self._locks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._verify_existing_artifacts()

    def append(self, record: DecisionPredictionRecord) -> AppendResult:
        """Persist a condition once; an identical retry returns the same result."""

        validated = _validate_prediction(record)
        key = BenchmarkSampleKey.from_record(validated)
        with self._sample_lock(key):
            if self._set_path(key).exists():
                self._load_and_verify_set(key)
                raise SampleAlreadySealedError("Cannot append after the sample has been sealed")

            state = self._load_state(key)
            if state is None:
                state = ConditionCommitState(key=key)
            next_state, result = append_condition_record(state, validated)
            if next_state is state:
                return result

            self._write_record(validated)
            self._write_staging(next_state)
            return result

    def seal(self, key: BenchmarkSampleKey) -> ConditionCommitSet:
        """Atomically publish an exact four-condition set."""

        with self._sample_lock(key):
            if self._set_path(key).exists():
                return self._load_and_verify_set(key)
            state = self._load_state(key)
            if state is None:
                raise IncompleteConditionSetError("Cannot seal a sample with no predictions")
            commit_set = seal_condition_records(state, sealed_at_utc=self._clock())
            self._atomic_write_model(self._set_path(key), commit_set)
            return commit_set

    def get(self, key: BenchmarkSampleKey) -> ConditionCommitSet | None:
        """Return a verified sealed set, or none when the sample is still pending."""

        with self._sample_lock(key):
            if not self._set_path(key).exists():
                return None
            return self._load_and_verify_set(key)

    def get_records(
        self,
        key: BenchmarkSampleKey,
    ) -> tuple[DecisionPredictionRecord, ...]:
        """Return all records referenced by a verified sealed sample."""

        with self._sample_lock(key):
            if not self._set_path(key).exists():
                raise IncompleteConditionSetError("Sample has no sealed prediction set")
            commit_set = self._load_and_verify_set(key)
            return tuple(self._load_record(item) for item in commit_set.prediction_hashes)

    def get_staged_records(
        self,
        key: BenchmarkSampleKey,
    ) -> tuple[DecisionPredictionRecord, ...]:
        """Return verified records staged for a complete or incomplete sample."""

        with self._sample_lock(key):
            state = self._load_state(key)
            return () if state is None else state.records

    def sealed_keys(self) -> tuple[BenchmarkSampleKey, ...]:
        """List verified sample keys with published condition sets."""

        with self._lock:
            keys: list[BenchmarkSampleKey] = []
            for path in sorted(self._sets_dir.glob("*.json")):
                commit_set = self._read_model(path, ConditionCommitSet)
                if path.stem != sample_key_hash(commit_set.key):
                    raise CommitStorageError(
                        f"Commit-set filename does not match sample key: {path}"
                    )
                self._verify_commit_set(commit_set)
                keys.append(commit_set.key)
            return tuple(sorted(keys, key=_sample_key_order))

    @contextmanager
    def _sample_lock(self, key: BenchmarkSampleKey) -> Generator[None]:
        lock_path = self._locks_dir / f"{sample_key_hash(key)}.lock"
        with self._lock, lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _verify_existing_artifacts(self) -> None:
        for path in sorted(self._records_dir.glob("*.json")):
            record = self._read_model(path, DecisionPredictionRecord)
            if path.stem != record.record_hash:
                raise CommitStorageError(f"Prediction filename does not match content: {path}")

        for path in sorted(self._staging_dir.glob("*.json")):
            index = self._read_model(path, _StagingIndex)
            if path.stem != sample_key_hash(index.key):
                raise CommitStorageError(f"Staging filename does not match sample key: {path}")
            self._state_from_index(index)

        for path in sorted(self._sets_dir.glob("*.json")):
            commit_set = self._read_model(path, ConditionCommitSet)
            if path.stem != sample_key_hash(commit_set.key):
                raise CommitStorageError(f"Commit-set filename does not match sample key: {path}")
            self._verify_commit_set(commit_set)

    def _load_state(self, key: BenchmarkSampleKey) -> ConditionCommitState | None:
        path = self._staging_path(key)
        if not path.exists():
            return None
        index = self._read_model(path, _StagingIndex)
        if index.key != key:
            raise CommitStorageError("Staging index contains another sample key")
        return self._state_from_index(index)

    def _state_from_index(self, index: _StagingIndex) -> ConditionCommitState:
        records = tuple(self._load_record(item.record_hash) for item in index.records)
        for reference, record in zip(index.records, records, strict=True):
            if reference.condition is not record.condition:
                raise CommitStorageError("Staging condition does not match its prediction")
        try:
            return ConditionCommitState(key=index.key, records=records)
        except ValueError as error:
            raise CommitStorageError("Staging index contains inconsistent predictions") from error

    def _load_record(self, record_hash: Sha256) -> DecisionPredictionRecord:
        path = self._records_dir / f"{record_hash}.json"
        if not path.exists():
            raise CommitStorageError(f"Missing prediction record: {record_hash}")
        record = self._read_model(path, DecisionPredictionRecord)
        if path.stem != record.record_hash or record.record_hash != record_hash:
            raise CommitStorageError("Prediction path and content hashes differ")
        return record

    def _write_record(self, record: DecisionPredictionRecord) -> None:
        path = self._records_dir / f"{record.record_hash}.json"
        if path.exists():
            stored = self._read_model(path, DecisionPredictionRecord)
            if stored != record:
                raise CommitStorageError("Existing content-addressed prediction differs")
            return
        self._atomic_write_model(path, record)

    def _write_staging(self, state: ConditionCommitState) -> None:
        content = _StagingIndexContent(
            key=state.key,
            records=tuple(
                _ConditionRecordRef(condition=record.condition, record_hash=record.record_hash)
                for record in state.records
            ),
            updated_at_utc=self._clock(),
        )
        index = _StagingIndex.model_validate(
            {
                **content.model_dump(mode="python"),
                "state_hash": model_content_hash(content),
            }
        )
        self._atomic_write_model(self._staging_path(state.key), index)

    def _load_and_verify_set(self, key: BenchmarkSampleKey) -> ConditionCommitSet:
        commit_set = self._read_model(self._set_path(key), ConditionCommitSet)
        if commit_set.key != key:
            raise CommitStorageError("Commit set contains another sample key")
        self._verify_commit_set(commit_set)
        return commit_set

    def _verify_commit_set(self, commit_set: ConditionCommitSet) -> None:
        records = tuple(self._load_record(item) for item in commit_set.prediction_hashes)
        try:
            state = ConditionCommitState(key=commit_set.key, records=records)
            expected = seal_condition_records(state, sealed_at_utc=commit_set.sealed_at_utc)
        except ValueError as error:
            raise CommitStorageError("Commit set references inconsistent predictions") from error
        if expected != commit_set:
            raise CommitStorageError("Commit set content does not match its predictions")

        staging_path = self._staging_path(commit_set.key)
        if staging_path.exists():
            staged = self._load_state(commit_set.key)
            if staged is None or tuple(record.record_hash for record in staged.records) != (
                commit_set.prediction_hashes
            ):
                raise CommitStorageError("Sealed and staged prediction sets differ")

    def _staging_path(self, key: BenchmarkSampleKey) -> Path:
        return self._staging_dir / f"{sample_key_hash(key)}.json"

    def _set_path(self, key: BenchmarkSampleKey) -> Path:
        return self._sets_dir / f"{sample_key_hash(key)}.json"

    def _atomic_write_model(self, path: Path, model: ContractModel) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(model.model_dump_json(indent=2).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_model[ModelT: ContractModel](path: Path, model_type: type[ModelT]) -> ModelT:
        try:
            return model_type.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as error:
            raise CommitStorageError(f"Invalid commitment artifact: {path}") from error


def _validate_prediction(record: DecisionPredictionRecord) -> DecisionPredictionRecord:
    try:
        validated = DecisionPredictionRecord.model_validate(record.model_dump(mode="python"))
    except ValueError as error:
        raise CommitmentError("Prediction record failed validation") from error
    if validated.record_hash != prediction_record_hash(validated):
        raise CommitmentError("Prediction record hash does not match its content")
    return validated


def _validate_record_sequence(
    key: BenchmarkSampleKey,
    records: tuple[DecisionPredictionRecord, ...],
) -> None:
    conditions = tuple(record.condition for record in records)
    expected_order = tuple(
        condition for condition in EXPECTED_CONDITIONS if condition in conditions
    )
    if conditions != expected_order or len(set(conditions)) != len(conditions):
        raise CommitConflictError("Conditions must be unique and canonically ordered")

    parity_fields = (
        "case_content_hash",
        "initial_state_hash",
        "public_record_hash",
        "public_evidence_hash",
        "tracker_version",
        "policy_version",
        "observation_schema",
    )
    for record in records:
        _validate_prediction(record)
        if BenchmarkSampleKey.from_record(record) != key:
            raise CommitParityError("Prediction does not match the sample key")
    if records:
        first = records[0]
        for record in records[1:]:
            mismatches = tuple(
                field_name
                for field_name in parity_fields
                if getattr(record, field_name) != getattr(first, field_name)
            )
            if mismatches:
                raise CommitParityError(f"Prediction parity mismatch: {', '.join(mismatches)}")


def _sample_key_order(key: BenchmarkSampleKey) -> tuple[str, str, str, str, str]:
    return (
        key.benchmark_version,
        key.run_id,
        key.case_id,
        key.sample_id,
        key.model_route_id,
    )


def _require_utc(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")


def _utc_now() -> datetime:
    return datetime.now(UTC)
