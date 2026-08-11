"""Run-wide decision commitments published before criterion access."""

import fcntl
import os
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.public.commitments import (
    BenchmarkSampleKey,
    ConditionCommitSet,
    ConditionCommitStore,
)
from socratic_tutor.benchmark.public.models import EXPECTED_CONDITIONS, BenchmarkCondition
from socratic_tutor.benchmark.public.predictions import DecisionPredictionRecord
from socratic_tutor.contracts import ContractModel


class GlobalSealError(ValueError):
    """A run-wide seal violates plan, accounting, or lifecycle rules."""


class GlobalSealAccountingError(GlobalSealError):
    """Planned and observed sample outcomes do not agree."""


class GlobalSealConflictError(GlobalSealError):
    """An immutable global seal already exists for another plan."""


class GlobalSealStorageError(RuntimeError):
    """A persisted global seal is missing, corrupt, or inconsistent."""


class SampleDecisionStatus(StrEnum):
    """Terminal decision-phase state for one planned sample."""

    COMPLETE = "complete"
    PRECRITERION_MISSING = "precriterion_missing"
    INVALID = "invalid"


class SampleDecisionReason(StrEnum):
    """Frozen reason codes for samples that cannot reach criterion execution."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    GENERATION_MISSING = "generation_missing"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INCOMPLETE_CONDITION_SET = "incomplete_condition_set"
    COMMIT_CONFLICT = "commit_conflict"
    PARITY_MISMATCH = "parity_mismatch"
    INTEGRITY_FAILURE = "integrity_failure"
    DECISION_FAILURE = "decision_failure"


_MISSING_REASONS = frozenset(
    {
        SampleDecisionReason.PROVIDER_UNAVAILABLE,
        SampleDecisionReason.GENERATION_MISSING,
        SampleDecisionReason.BUDGET_EXHAUSTED,
        SampleDecisionReason.INCOMPLETE_CONDITION_SET,
    }
)
_INVALID_REASONS = frozenset(
    {
        SampleDecisionReason.COMMIT_CONFLICT,
        SampleDecisionReason.PARITY_MISMATCH,
        SampleDecisionReason.INTEGRITY_FAILURE,
        SampleDecisionReason.DECISION_FAILURE,
    }
)


class DecisionRunProvenance(ContractModel):
    """Frozen software, data, component, and analysis identity for a run."""

    code_revision: str = Field(min_length=1)
    dirty_worktree: bool
    pixi_lock_hash: Sha256
    resolved_config_hash: Sha256
    public_manifest_hash: Sha256
    split_hash: Sha256
    prompt_version: str = Field(min_length=1)
    prompt_hash: Sha256
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    observation_schema: str = Field(min_length=1)
    action_mapping_hash: Sha256
    calibration_decision_hash: Sha256 | None = None
    analysis_plan_hash: Sha256
    review_gate_hash: Sha256
    root_seed: int = Field(ge=0, le=2**32 - 1)


class _DecisionRunPlanContent(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.run_plan.v1"] = "benchmark.run_plan.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    provenance: DecisionRunProvenance
    expected_conditions: tuple[BenchmarkCondition, ...]
    planned_samples: tuple[BenchmarkSampleKey, ...] = Field(min_length=1)
    created_at_utc: datetime

    @model_validator(mode="after")
    def validate_plan(self) -> "_DecisionRunPlanContent":
        if self.expected_conditions != EXPECTED_CONDITIONS:
            raise ValueError("Run plan must use the exact ordered benchmark conditions")
        _require_utc(self.created_at_utc, field_name="Run-plan creation time")
        _validate_planned_samples(
            self.planned_samples,
            benchmark_version=self.benchmark_version,
            run_id=self.run_id,
        )
        return self


class DecisionRunPlan(_DecisionRunPlanContent):
    """Content-addressed sample matrix frozen before decision generation."""

    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan_hash(self) -> "DecisionRunPlan":
        if self.plan_hash != decision_run_plan_hash(self):
            raise ValueError("Run-plan hash does not match its content")
        return self


def decision_run_plan_hash(plan: DecisionRunPlan) -> Sha256:
    """Hash a run plan without its self-referential digest."""

    return model_content_hash(plan, exclude={"plan_hash"})


def create_decision_run_plan(
    *,
    benchmark_version: str,
    run_id: str,
    provenance: DecisionRunProvenance,
    planned_samples: Iterable[BenchmarkSampleKey],
    created_at_utc: datetime,
) -> DecisionRunPlan:
    """Create a canonically ordered, content-addressed run plan."""

    content = _DecisionRunPlanContent(
        benchmark_version=benchmark_version,
        run_id=run_id,
        provenance=provenance,
        expected_conditions=EXPECTED_CONDITIONS,
        planned_samples=tuple(sorted(planned_samples, key=_sample_key_order)),
        created_at_utc=created_at_utc,
    )
    return DecisionRunPlan.model_validate(
        {
            **content.model_dump(mode="python"),
            "plan_hash": model_content_hash(content),
        }
    )


class SampleDecisionFailure(ContractModel):
    """Prespecified terminal outcome for a sample without a valid commit set."""

    key: BenchmarkSampleKey
    status: SampleDecisionStatus
    reason_code: SampleDecisionReason
    failure_record_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "SampleDecisionFailure":
        _validate_failure_reason(self.status, self.reason_code)
        return self


class SampleDecisionStatusRecord(ContractModel):
    """One planned sample's immutable decision-phase accounting record."""

    key: BenchmarkSampleKey
    status: SampleDecisionStatus
    condition_set_hash: Sha256 | None = None
    prediction_hashes: tuple[Sha256, ...] = ()
    reason_code: SampleDecisionReason | None = None
    failure_record_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "SampleDecisionStatusRecord":
        if self.status is SampleDecisionStatus.COMPLETE:
            if self.condition_set_hash is None:
                raise ValueError("Complete sample requires a condition-set hash")
            if len(self.prediction_hashes) != len(EXPECTED_CONDITIONS):
                raise ValueError("Complete sample requires exactly four prediction hashes")
            if len(set(self.prediction_hashes)) != len(self.prediction_hashes):
                raise ValueError("Complete sample prediction hashes must be unique")
            if self.reason_code is not None or self.failure_record_hash is not None:
                raise ValueError("Complete sample cannot contain failure metadata")
        else:
            if self.condition_set_hash is not None:
                raise ValueError("Failed sample cannot reference a condition-set hash")
            if len(self.prediction_hashes) > len(EXPECTED_CONDITIONS):
                raise ValueError("Failed sample cannot contain more than four prediction hashes")
            if len(set(self.prediction_hashes)) != len(self.prediction_hashes):
                raise ValueError("Failed sample prediction hashes must be unique")
            if self.reason_code is None:
                raise ValueError("Failed sample requires a frozen reason code")
            _validate_failure_reason(self.status, self.reason_code)
        return self


class _GlobalDecisionSealContent(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.global_decision_seal.v1"] = "benchmark.global_decision_seal.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_plan_hash: Sha256
    planned_sample_count: int = Field(ge=1)
    sample_statuses: tuple[SampleDecisionStatusRecord, ...] = Field(min_length=1)
    complete_set_hashes: tuple[Sha256, ...]
    decision_record_root_hash: Sha256
    sealed_at_utc: datetime

    @model_validator(mode="after")
    def validate_accounting(self) -> "_GlobalDecisionSealContent":
        _require_utc(self.sealed_at_utc, field_name="Global seal time")
        if len(self.sample_statuses) != self.planned_sample_count:
            raise ValueError("Global seal must account for every planned sample")
        keys = tuple(record.key for record in self.sample_statuses)
        if len(set(keys)) != len(keys):
            raise ValueError("Every planned sample key must appear exactly once")
        if keys != tuple(sorted(keys, key=_sample_key_order)):
            raise ValueError("Global sample statuses must be canonically ordered")
        if any(
            key.benchmark_version != self.benchmark_version or key.run_id != self.run_id
            for key in keys
        ):
            raise ValueError("Global sample identity must match benchmark and run")
        expected_set_hashes = tuple(
            record.condition_set_hash
            for record in self.sample_statuses
            if record.status is SampleDecisionStatus.COMPLETE
        )
        if self.complete_set_hashes != expected_set_hashes:
            raise ValueError("Complete-set hashes must match complete sample statuses")
        if len(set(self.complete_set_hashes)) != len(self.complete_set_hashes):
            raise ValueError("Complete condition-set hashes must be unique")
        if self.decision_record_root_hash != decision_record_root_hash(self.sample_statuses):
            raise ValueError("Decision record root hash does not match sample accounting")
        return self


class GlobalDecisionSeal(_GlobalDecisionSealContent):
    """Immutable run-wide barrier between decisions and criterion outcomes."""

    seal_hash: Sha256

    @model_validator(mode="after")
    def validate_seal_hash(self) -> "GlobalDecisionSeal":
        if self.seal_hash != global_decision_seal_hash(self):
            raise ValueError("Global decision seal hash does not match its content")
        return self


def decision_record_root_hash(
    statuses: tuple[SampleDecisionStatusRecord, ...],
) -> Sha256:
    """Bind statuses, set hashes, and underlying prediction hashes."""

    return canonical_sha256(
        {
            "schema_id": "benchmark.decision_record_root.v1",
            "sample_statuses": [record.model_dump(mode="json") for record in statuses],
            "complete_set_hashes": [
                record.condition_set_hash
                for record in statuses
                if record.status is SampleDecisionStatus.COMPLETE
            ],
            "prediction_hashes": [
                prediction_hash
                for record in statuses
                for prediction_hash in record.prediction_hashes
            ],
        }
    )


def global_decision_seal_hash(seal: GlobalDecisionSeal) -> Sha256:
    """Hash a global seal without its self-referential digest."""

    return model_content_hash(seal, exclude={"seal_hash"})


class FilesystemGlobalDecisionSealStore:
    """Publish and verify one atomic global decision manifest."""

    def __init__(
        self,
        decision_root: Path,
        condition_store: ConditionCommitStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.decision_root = decision_root
        self.path = decision_root / "commitment_manifest.json"
        self._lock_path = decision_root / "commitment_manifest.lock"
        self._condition_store = condition_store
        self._clock = clock or _utc_now
        self._lock = RLock()
        decision_root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        plan: DecisionRunPlan,
        *,
        failures: Iterable[SampleDecisionFailure] = (),
    ) -> GlobalDecisionSeal:
        """Account for the frozen matrix and atomically publish its barrier."""

        validated_plan = DecisionRunPlan.model_validate(plan.model_dump(mode="python"))
        with self._file_lock():
            if self.path.exists():
                existing = self._load_and_verify()
                if existing.run_plan_hash != validated_plan.plan_hash:
                    raise GlobalSealConflictError("Another run plan is already sealed")
                if tuple(record.key for record in existing.sample_statuses) != (
                    validated_plan.planned_samples
                ):
                    raise GlobalSealConflictError("Sealed sample matrix differs from the run plan")
                return existing

            failure_map = _failure_map(failures, validated_plan)
            statuses, latest_commit = self._account_for_samples(validated_plan, failure_map)
            sealed_at = self._clock()
            _require_utc(sealed_at, field_name="Global seal time")
            if sealed_at <= latest_commit:
                raise GlobalSealAccountingError(
                    "Global seal time must follow every prediction and condition-set commit"
                )

            content = _GlobalDecisionSealContent(
                benchmark_version=validated_plan.benchmark_version,
                run_id=validated_plan.run_id,
                run_plan_hash=validated_plan.plan_hash,
                planned_sample_count=len(validated_plan.planned_samples),
                sample_statuses=statuses,
                complete_set_hashes=_complete_set_hashes(statuses),
                decision_record_root_hash=decision_record_root_hash(statuses),
                sealed_at_utc=sealed_at,
            )
            seal = GlobalDecisionSeal.model_validate(
                {
                    **content.model_dump(mode="python"),
                    "seal_hash": model_content_hash(content),
                }
            )
            self._atomic_write(seal)
            return seal

    def load(self) -> GlobalDecisionSeal | None:
        """Return the fully verified final seal, or none before publication."""

        with self._file_lock():
            if not self.path.exists():
                return None
            return self._load_and_verify()

    def _account_for_samples(
        self,
        plan: DecisionRunPlan,
        failures: dict[BenchmarkSampleKey, SampleDecisionFailure],
    ) -> tuple[tuple[SampleDecisionStatusRecord, ...], datetime]:
        planned = set(plan.planned_samples)
        observed = set(self._condition_store.sealed_keys())
        if observed - planned:
            raise GlobalSealAccountingError("Condition store contains an unplanned sealed sample")

        statuses: list[SampleDecisionStatusRecord] = []
        latest_commit = plan.created_at_utc
        for key in plan.planned_samples:
            commit_set = self._condition_store.get(key)
            failure = failures.get(key)
            if commit_set is None:
                if failure is None:
                    raise GlobalSealAccountingError(
                        f"Planned sample has no terminal outcome: {key.case_id}/{key.sample_id}"
                    )
                records = self._condition_store.get_staged_records(key)
                _require_plan_precedes_records(plan, records)
                statuses.append(_failure_status(failure, records=records))
                if records:
                    latest_commit = _latest_time(
                        latest_commit,
                        *(record.committed_at_utc for record in records),
                    )
                continue
            if failure is not None:
                raise GlobalSealAccountingError(
                    f"Complete sample also has a failure declaration: {key.case_id}/{key.sample_id}"
                )

            records = self._condition_store.get_records(key)
            _verify_complete_sample(key, commit_set, records)
            _require_plan_precedes_records(plan, records, commit_set=commit_set)
            statuses.append(
                SampleDecisionStatusRecord(
                    key=key,
                    status=SampleDecisionStatus.COMPLETE,
                    condition_set_hash=commit_set.seal_hash,
                    prediction_hashes=commit_set.prediction_hashes,
                )
            )
            latest_commit = _latest_time(
                latest_commit,
                commit_set.sealed_at_utc,
                *(record.committed_at_utc for record in records),
            )

        return tuple(statuses), latest_commit

    def _load_and_verify(self) -> GlobalDecisionSeal:
        try:
            seal = GlobalDecisionSeal.model_validate_json(self.path.read_bytes())
        except (OSError, ValueError) as error:
            raise GlobalSealStorageError("Invalid global decision seal") from error

        expected_keys = tuple(record.key for record in seal.sample_statuses)
        observed_keys = self._condition_store.sealed_keys()
        complete_keys = tuple(
            record.key
            for record in seal.sample_statuses
            if record.status is SampleDecisionStatus.COMPLETE
        )
        if observed_keys != complete_keys:
            raise GlobalSealStorageError("Condition sets changed after global sealing")
        if len(expected_keys) != seal.planned_sample_count:
            raise GlobalSealStorageError("Global seal sample count is inconsistent")

        for status in seal.sample_statuses:
            if status.status is not SampleDecisionStatus.COMPLETE:
                records = self._condition_store.get_staged_records(status.key)
                if tuple(record.record_hash for record in records) != status.prediction_hashes:
                    raise GlobalSealStorageError(
                        "Failed-sample predictions changed after global sealing"
                    )
                if any(record.committed_at_utc >= seal.sealed_at_utc for record in records):
                    raise GlobalSealStorageError(
                        "Failed-sample predictions do not precede the global seal"
                    )
                continue
            commit_set = self._condition_store.get(status.key)
            if commit_set is None:
                raise GlobalSealStorageError("Global seal references a missing condition set")
            records = self._condition_store.get_records(status.key)
            try:
                _verify_complete_sample(status.key, commit_set, records)
            except GlobalSealAccountingError as error:
                raise GlobalSealStorageError(
                    "Global seal references invalid predictions"
                ) from error
            if (
                status.condition_set_hash != commit_set.seal_hash
                or status.prediction_hashes != commit_set.prediction_hashes
            ):
                raise GlobalSealStorageError("Global seal references changed condition content")
            if commit_set.sealed_at_utc >= seal.sealed_at_utc or any(
                record.committed_at_utc >= seal.sealed_at_utc for record in records
            ):
                raise GlobalSealStorageError("Decision records do not precede the global seal")
        return seal

    @contextmanager
    def _file_lock(self) -> Generator[None]:
        with self._lock, self._lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _atomic_write(self, seal: GlobalDecisionSeal) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(seal.model_dump_json(indent=2).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _failure_map(
    failures: Iterable[SampleDecisionFailure],
    plan: DecisionRunPlan,
) -> dict[BenchmarkSampleKey, SampleDecisionFailure]:
    values: dict[BenchmarkSampleKey, SampleDecisionFailure] = {}
    planned = set(plan.planned_samples)
    for failure in failures:
        if failure.key not in planned:
            raise GlobalSealAccountingError("Failure declaration contains an unplanned sample")
        if failure.key in values:
            raise GlobalSealAccountingError("Sample has duplicate failure declarations")
        values[failure.key] = failure
    return values


def _failure_status(
    failure: SampleDecisionFailure,
    *,
    records: tuple[DecisionPredictionRecord, ...],
) -> SampleDecisionStatusRecord:
    return SampleDecisionStatusRecord(
        key=failure.key,
        status=failure.status,
        prediction_hashes=tuple(record.record_hash for record in records),
        reason_code=failure.reason_code,
        failure_record_hash=failure.failure_record_hash,
    )


def _complete_set_hashes(
    statuses: tuple[SampleDecisionStatusRecord, ...],
) -> tuple[Sha256, ...]:
    hashes: list[Sha256] = []
    for status in statuses:
        if status.status is not SampleDecisionStatus.COMPLETE:
            continue
        if status.condition_set_hash is None:
            raise GlobalSealAccountingError("Complete sample is missing its condition-set hash")
        hashes.append(status.condition_set_hash)
    return tuple(hashes)


def _verify_complete_sample(
    key: BenchmarkSampleKey,
    commit_set: ConditionCommitSet,
    records: tuple[DecisionPredictionRecord, ...],
) -> None:
    if commit_set.key != key:
        raise GlobalSealAccountingError("Condition set does not match its planned sample")
    if len(records) != len(EXPECTED_CONDITIONS):
        raise GlobalSealAccountingError("Complete sample does not contain four predictions")
    if tuple(record.record_hash for record in records) != commit_set.prediction_hashes:
        raise GlobalSealAccountingError("Condition set prediction hashes do not match records")
    if tuple(record.condition for record in records) != EXPECTED_CONDITIONS:
        raise GlobalSealAccountingError("Complete sample conditions are not canonically ordered")


def _require_plan_precedes_records(
    plan: DecisionRunPlan,
    records: tuple[DecisionPredictionRecord, ...],
    *,
    commit_set: ConditionCommitSet | None = None,
) -> None:
    if any(record.committed_at_utc <= plan.created_at_utc for record in records):
        raise GlobalSealAccountingError("Run plan must precede every prediction commit")
    if commit_set is not None and commit_set.sealed_at_utc <= plan.created_at_utc:
        raise GlobalSealAccountingError("Run plan must precede every condition-set commit")


def _validate_failure_reason(
    status: SampleDecisionStatus,
    reason: SampleDecisionReason,
) -> None:
    if status is SampleDecisionStatus.COMPLETE:
        raise ValueError("Complete status cannot be used as a failure")
    allowed = (
        _MISSING_REASONS
        if status is SampleDecisionStatus.PRECRITERION_MISSING
        else (_INVALID_REASONS)
    )
    if reason not in allowed:
        raise ValueError(f"Reason {reason.value} is not valid for status {status.value}")


def _validate_planned_samples(
    samples: tuple[BenchmarkSampleKey, ...],
    *,
    benchmark_version: str,
    run_id: str,
) -> None:
    if len(set(samples)) != len(samples):
        raise ValueError("Run-plan sample keys must be unique")
    if samples != tuple(sorted(samples, key=_sample_key_order)):
        raise ValueError("Run-plan samples must be canonically ordered")
    if any(
        sample.benchmark_version != benchmark_version or sample.run_id != run_id
        for sample in samples
    ):
        raise ValueError("Planned samples must match benchmark and run identity")

    case_ids = tuple(sorted({sample.case_id for sample in samples}))
    sample_ids = tuple(sorted({sample.sample_id for sample in samples}))
    routes = tuple(sorted({sample.model_route_id for sample in samples}))
    expected_matrix = {
        (case_id, sample_id, route)
        for case_id in case_ids
        for sample_id in sample_ids
        for route in routes
    }
    observed_matrix = {
        (sample.case_id, sample.sample_id, sample.model_route_id) for sample in samples
    }
    if observed_matrix != expected_matrix:
        raise ValueError("Run-plan samples must form the complete case x sample x route matrix")


def _sample_key_order(key: BenchmarkSampleKey) -> tuple[str, str, str]:
    return (key.case_id, key.sample_id, key.model_route_id)


def _latest_time(current: datetime, *values: datetime) -> datetime:
    return max(current, *values)


def _require_utc(value: datetime, *, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
