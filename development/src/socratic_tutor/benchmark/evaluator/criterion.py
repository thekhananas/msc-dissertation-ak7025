"""Immutable criterion outcomes created only after the decision barrier."""

from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.gate import VerifiedCriterionAccess
from socratic_tutor.benchmark.evaluator.models import CriterionSpec
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationChannel,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public import BenchmarkSampleKey
from socratic_tutor.benchmark.replay import (
    RecordedGenerationResponse,
    ReplayedGenerationResult,
)
from socratic_tutor.contracts import ContractModel


class CriterionRecordError(ValueError):
    """A criterion result violates identity, timing, or outcome semantics."""


class CriterionRecordConflictError(CriterionRecordError):
    """One sample was assigned two different criterion outcomes."""


class CriterionExecutionStatus(StrEnum):
    """Normalized terminal states for the criterion generation/execution path."""

    COMPLETED = "completed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    SANDBOX_TIMEOUT = "sandbox_timeout"
    SANDBOX_RESOURCE_LIMIT = "sandbox_resource_limit"
    SANDBOX_ERROR = "sandbox_error"


class CriterionMissingReason(StrEnum):
    """Frozen missingness reasons that are never converted to failed performance."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    SANDBOX_TIMEOUT = "sandbox_timeout"
    SANDBOX_RESOURCE_LIMIT = "sandbox_resource_limit"
    SANDBOX_ERROR = "sandbox_error"


_MISSING_REASON_BY_STATUS: dict[CriterionExecutionStatus, CriterionMissingReason] = {
    CriterionExecutionStatus.PROVIDER_UNAVAILABLE: CriterionMissingReason.PROVIDER_UNAVAILABLE,
    CriterionExecutionStatus.SANDBOX_UNAVAILABLE: CriterionMissingReason.SANDBOX_UNAVAILABLE,
    CriterionExecutionStatus.SANDBOX_TIMEOUT: CriterionMissingReason.SANDBOX_TIMEOUT,
    CriterionExecutionStatus.SANDBOX_RESOURCE_LIMIT: (
        CriterionMissingReason.SANDBOX_RESOURCE_LIMIT
    ),
    CriterionExecutionStatus.SANDBOX_ERROR: CriterionMissingReason.SANDBOX_ERROR,
}


class NormalizedCriterionExecution(ContractModel):
    """Provider-neutral execution result suitable for deterministic scoring."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.criterion_execution.v1"] = "benchmark.criterion_execution.v1"
    operation_id: str | None = Field(default=None, min_length=1)
    status: CriterionExecutionStatus
    test_bundle_sha256: Sha256
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    exit_code: int | None = None
    timed_out: bool
    resource_limited: bool
    stdout_sha256: Sha256 | None = None
    stderr_sha256: Sha256 | None = None
    requested_at_utc: datetime
    executed_at_utc: datetime | None = None
    execution_hash: Sha256

    @model_validator(mode="after")
    def validate_execution(self) -> "NormalizedCriterionExecution":
        _require_utc(self.requested_at_utc, "Criterion request time")
        if self.executed_at_utc is not None:
            _require_utc(self.executed_at_utc, "Criterion execution time")
            if self.executed_at_utc < self.requested_at_utc:
                raise ValueError("Criterion execution cannot precede its request")

        if self.status is CriterionExecutionStatus.COMPLETED:
            if self.operation_id is None or self.executed_at_utc is None:
                raise ValueError("Completed criterion execution requires operation and time")
            if self.passed + self.failed == 0:
                raise ValueError("Completed criterion execution requires at least one test")
            if self.exit_code is None:
                raise ValueError("Completed criterion execution requires an exit code")
            if self.timed_out or self.resource_limited:
                raise ValueError("Completed criterion execution cannot have limit flags")
            if self.stdout_sha256 is None or self.stderr_sha256 is None:
                raise ValueError("Completed criterion execution requires output hashes")
        elif self.status is CriterionExecutionStatus.PROVIDER_UNAVAILABLE:
            if (
                any(
                    value is not None
                    for value in (
                        self.operation_id,
                        self.exit_code,
                        self.stdout_sha256,
                        self.stderr_sha256,
                        self.executed_at_utc,
                    )
                )
                or self.passed
                or self.failed
                or self.timed_out
                or self.resource_limited
            ):
                raise ValueError("Provider unavailability cannot contain sandbox execution data")
        elif self.status is CriterionExecutionStatus.SANDBOX_UNAVAILABLE:
            if (
                self.operation_id is not None
                or self.stdout_sha256 is not None
                or self.stderr_sha256 is not None
                or self.passed
                or self.failed
                or self.timed_out
                or self.resource_limited
            ):
                raise ValueError("Sandbox unavailability cannot contain test results or limits")
            if self.exit_code is not None or self.executed_at_utc is not None:
                raise ValueError("Unavailable sandbox cannot contain execution completion data")
        else:
            if self.operation_id is None or self.executed_at_utc is None:
                raise ValueError("Attempted sandbox execution requires operation and time")
            if self.stdout_sha256 is None or self.stderr_sha256 is None:
                raise ValueError("Attempted sandbox execution requires output hashes")
            if self.status is CriterionExecutionStatus.SANDBOX_TIMEOUT and not self.timed_out:
                raise ValueError("Sandbox timeout status requires timed_out")
            if (
                self.status is CriterionExecutionStatus.SANDBOX_RESOURCE_LIMIT
                and not self.resource_limited
            ):
                raise ValueError("Sandbox resource status requires resource_limited")
            if self.status is CriterionExecutionStatus.SANDBOX_ERROR and (
                self.timed_out or self.resource_limited
            ):
                raise ValueError("Sandbox error cannot carry timeout or resource-limit flags")

        if self.execution_hash != normalized_execution_hash(self):
            raise ValueError("Criterion execution hash does not match its content")
        return self


def normalized_execution_hash(execution: NormalizedCriterionExecution) -> Sha256:
    """Hash all normalized execution fields except the self-referential digest."""

    return model_content_hash(execution, exclude={"execution_hash"})


def create_normalized_criterion_execution(
    *,
    status: CriterionExecutionStatus,
    test_bundle_sha256: Sha256,
    requested_at_utc: datetime,
    operation_id: str | None = None,
    passed: int = 0,
    failed: int = 0,
    exit_code: int | None = None,
    timed_out: bool = False,
    resource_limited: bool = False,
    stdout_sha256: Sha256 | None = None,
    stderr_sha256: Sha256 | None = None,
    executed_at_utc: datetime | None = None,
) -> NormalizedCriterionExecution:
    """Create one content-addressed provider-neutral execution result."""

    content = {
        "schema_version": 1,
        "schema_id": "benchmark.criterion_execution.v1",
        "operation_id": operation_id,
        "status": status,
        "test_bundle_sha256": test_bundle_sha256,
        "passed": passed,
        "failed": failed,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "resource_limited": resource_limited,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "requested_at_utc": requested_at_utc,
        "executed_at_utc": executed_at_utc,
    }
    draft = NormalizedCriterionExecution.model_construct(
        _fields_set=set(content),
        **content,
        execution_hash="0" * 64,
    )
    return NormalizedCriterionExecution.model_validate(
        {**content, "execution_hash": normalized_execution_hash(draft)}
    )


class CriterionRecord(ContractModel):
    """One immutable held-out transfer outcome tied to the sealed decisions."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.criterion_record.v1"] = "benchmark.criterion_record.v1"
    key: BenchmarkSampleKey
    global_seal_hash: Sha256
    condition_set_hash: Sha256
    criterion_probe_id: str = Field(min_length=1)
    request_hash: Sha256
    response_hash: Sha256 | None
    test_bundle_sha256: Sha256
    rubric_sha256: Sha256
    execution: NormalizedCriterionExecution
    demonstrated_performance: bool | None
    missing_reason: CriterionMissingReason | None
    revealed_at_utc: datetime
    record_hash: Sha256

    @model_validator(mode="after")
    def validate_outcome(self) -> "CriterionRecord":
        _require_utc(self.revealed_at_utc, "Criterion reveal time")
        if self.revealed_at_utc < self.execution.requested_at_utc:
            raise ValueError("Criterion reveal cannot precede its request")
        if (
            self.execution.executed_at_utc is not None
            and self.revealed_at_utc < self.execution.executed_at_utc
        ):
            raise ValueError("Criterion reveal cannot precede execution")
        if self.execution.test_bundle_sha256 != self.test_bundle_sha256:
            raise ValueError("Criterion execution used another test bundle")

        expected_missing = _MISSING_REASON_BY_STATUS.get(self.execution.status)
        if self.execution.status is CriterionExecutionStatus.COMPLETED:
            if self.response_hash is None:
                raise ValueError("Completed criterion execution requires a response")
            if self.demonstrated_performance is None or self.missing_reason is not None:
                raise ValueError("Completed criterion execution requires one binary outcome")
            expected_outcome = self.execution.failed == 0
            if self.demonstrated_performance is not expected_outcome:
                raise ValueError("Criterion outcome does not match reviewed test results")
        else:
            if (
                self.demonstrated_performance is not None
                or self.missing_reason is not expected_missing
            ):
                raise ValueError("Unavailable criterion requires its exact typed missing reason")
            if (
                self.execution.status is CriterionExecutionStatus.PROVIDER_UNAVAILABLE
                and self.response_hash is not None
            ):
                raise ValueError("Provider unavailability cannot contain a response")
            if (
                self.execution.status is not CriterionExecutionStatus.PROVIDER_UNAVAILABLE
                and self.response_hash is None
            ):
                raise ValueError("Sandbox-stage missingness requires a generated response")

        if self.record_hash != criterion_record_hash(self):
            raise ValueError("Criterion record hash does not match its content")
        return self


def criterion_record_hash(record: CriterionRecord) -> Sha256:
    """Hash every criterion record field except its self-referential digest."""

    return model_content_hash(record, exclude={"record_hash"})


class CriterionRecordFactory:
    """Build criterion records from one consumed capability and exact request."""

    def build(
        self,
        access: VerifiedCriterionAccess,
        *,
        request: StudentGenerationRequest,
        response: RecordedGenerationResponse | ReplayedGenerationResult | None,
        execution: NormalizedCriterionExecution,
        revealed_at_utc: datetime,
    ) -> CriterionRecord:
        """Derive performance or missingness without caller-supplied labels."""

        criterion = _criterion_for_access(access)
        _validate_request(access, request, criterion.criterion_probe_id)
        response_hash = _validated_response_hash(request, response)
        if execution.test_bundle_sha256 != criterion.test_bundle_sha256:
            raise CriterionRecordError("Execution test bundle does not match the criterion")
        if execution.requested_at_utc <= access.global_sealed_at_utc:
            raise CriterionRecordError("Criterion request must follow the global decision seal")
        if execution.requested_at_utc <= access.condition_sealed_at_utc:
            raise CriterionRecordError("Criterion request must follow the condition seal")
        if revealed_at_utc <= access.global_sealed_at_utc:
            raise CriterionRecordError("Criterion reveal must follow the global decision seal")
        if revealed_at_utc <= access.condition_sealed_at_utc:
            raise CriterionRecordError("Criterion reveal must follow the condition seal")
        if (
            execution.status is not CriterionExecutionStatus.PROVIDER_UNAVAILABLE
            and response_hash is None
        ):
            raise CriterionRecordError("Sandbox-stage criterion results require a response")

        performance = (
            execution.failed == 0
            if execution.status is CriterionExecutionStatus.COMPLETED
            else None
        )
        missing_reason = _MISSING_REASON_BY_STATUS.get(execution.status)
        content = {
            "schema_version": 1,
            "schema_id": "benchmark.criterion_record.v1",
            "key": access.sample_key,
            "global_seal_hash": access.global_seal_hash,
            "condition_set_hash": access.condition_set_hash,
            "criterion_probe_id": criterion.criterion_probe_id,
            "request_hash": request.request_hash,
            "response_hash": response_hash,
            "test_bundle_sha256": criterion.test_bundle_sha256,
            "rubric_sha256": criterion.rubric_sha256,
            "execution": execution,
            "demonstrated_performance": performance,
            "missing_reason": missing_reason,
            "revealed_at_utc": revealed_at_utc,
        }
        draft = CriterionRecord.model_construct(
            _fields_set=set(content),
            **content,
            record_hash="0" * 64,
        )
        return CriterionRecord.model_validate(
            {**content, "record_hash": criterion_record_hash(draft)}
        )


class CriterionRecordIndex:
    """Thread-safe exactly-once index used before atomic dataset publication."""

    def __init__(self, records: Iterable[CriterionRecord] = ()) -> None:
        self._records: dict[BenchmarkSampleKey, CriterionRecord] = {}
        self._lock = RLock()
        for record in records:
            self.add(record)

    def exists(self, key: BenchmarkSampleKey) -> bool:
        """Return whether the sample already has a validated criterion record."""

        with self._lock:
            return key in self._records

    def add(self, record: CriterionRecord) -> CriterionRecord:
        """Append once; exact retries are idempotent and conflicts fail closed."""

        with self._lock:
            existing = self._records.get(record.key)
            if existing is None:
                self._records[record.key] = record
                return record
            if existing.record_hash == record.record_hash:
                return existing
            raise CriterionRecordConflictError("Sample already has another criterion record")

    def records(self) -> tuple[CriterionRecord, ...]:
        """Return records in stable sample-identity order."""

        with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda record: (
                        record.key.benchmark_version,
                        record.key.run_id,
                        record.key.case_id,
                        record.key.sample_id,
                        record.key.model_route_id,
                    ),
                )
            )


def _criterion_for_access(access: VerifiedCriterionAccess) -> CriterionSpec:
    matches = tuple(
        criterion
        for criterion in access.manifest.criteria
        if criterion.case_id == access.sample_key.case_id
    )
    if len(matches) != 1:
        raise CriterionRecordError("Capability must resolve to exactly one criterion")
    return matches[0]


def _validate_request(
    access: VerifiedCriterionAccess,
    request: StudentGenerationRequest,
    criterion_probe_id: str,
) -> None:
    key = access.sample_key
    route_id = f"{request.model_route.provider}/{request.model_route.model}"
    if request.channel is not GenerationChannel.CRITERION:
        raise CriterionRecordError("Criterion record requires a criterion-channel request")
    if not isinstance(request.task_payload, CriterionTaskPayload):
        raise CriterionRecordError("Criterion request has the wrong payload type")
    if request.task_payload.criterion_probe_id != criterion_probe_id:
        raise CriterionRecordError("Criterion request used another held-out probe")
    if (
        request.task_payload.benchmark_version != key.benchmark_version
        or request.run_id != key.run_id
        or request.case_id != key.case_id
        or request.sample_id != key.sample_id
        or route_id != key.model_route_id
    ):
        raise CriterionRecordError("Criterion request identity does not match the capability")


def _validated_response_hash(
    request: StudentGenerationRequest,
    response: RecordedGenerationResponse | ReplayedGenerationResult | None,
) -> Sha256 | None:
    if response is None:
        return None
    if isinstance(response, RecordedGenerationResponse):
        if response.request != request:
            raise CriterionRecordError("Criterion response belongs to another request")
        response_channel = response.request.channel
    else:
        if response.request_hash != request.request_hash:
            raise CriterionRecordError("Criterion response belongs to another request")
        response_channel = response.channel
    if response_channel is not GenerationChannel.CRITERION:
        raise CriterionRecordError("Criterion record requires a criterion-channel response")
    return response.response_hash


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
