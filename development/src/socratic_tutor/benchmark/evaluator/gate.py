"""One-time criterion access issued only after the decision phase is sealed."""

import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Protocol

from socratic_tutor.benchmark.evaluator.models import EvaluatorBenchmarkManifest
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public import (
    BenchmarkSampleKey,
    ConditionCommitSet,
    ConditionCommitStore,
    DecisionRunPlan,
    GlobalDecisionSeal,
    SampleDecisionStatus,
)


class CriterionGateError(ValueError):
    """Criterion access is unavailable or violates the sealed decision state."""


class CriterionNotSealedError(CriterionGateError):
    """The global decision phase has not produced a valid final seal."""


class CriterionIneligibleError(CriterionGateError):
    """The requested sample is not a complete sealed decision sample."""


class CriterionCapabilityError(CriterionGateError):
    """A criterion capability is unknown, altered, reused, or already consumed."""


class EvaluatorManifestAccessError(CriterionGateError):
    """The gated evaluator manifest does not match the capability."""


class GlobalDecisionSealReader(Protocol):
    """Minimum verified global-seal interface required by the gate."""

    def load(self) -> GlobalDecisionSeal | None: ...


type CriterionRecordExists = Callable[[BenchmarkSampleKey], bool]
type EvaluatorManifestLoader = Callable[[], EvaluatorBenchmarkManifest]
type NonceFactory = Callable[[], bytes]


@dataclass(frozen=True, slots=True)
class CriterionCapability:
    """One-use in-process authority bound to one complete prediction set."""

    global_seal_hash: str
    condition_set_hash: str
    sample_key: BenchmarkSampleKey
    nonce: bytes


@dataclass(frozen=True, slots=True)
class VerifiedCriterionAccess:
    """Private manifest access obtained by consuming a valid capability."""

    global_seal_hash: str
    condition_set_hash: str
    global_sealed_at_utc: datetime
    condition_sealed_at_utc: datetime
    sample_key: BenchmarkSampleKey
    manifest: EvaluatorBenchmarkManifest


class CriterionGate:
    """Verify decision commitments before constructing evaluator-only readers."""

    def __init__(
        self,
        global_seals: GlobalDecisionSealReader,
        condition_store: ConditionCommitStore,
        run_plan: DecisionRunPlan,
        *,
        criterion_record_exists: CriterionRecordExists,
        nonce_factory: NonceFactory | None = None,
    ) -> None:
        self._global_seals = global_seals
        self._condition_store = condition_store
        self._run_plan = DecisionRunPlan.model_validate(run_plan.model_dump(mode="python"))
        self._criterion_record_exists = criterion_record_exists
        self._nonce_factory = nonce_factory or _secure_nonce
        self._pending: dict[BenchmarkSampleKey, CriterionCapability] = {}
        self._consumed: set[BenchmarkSampleKey] = set()
        self._used_nonces: set[bytes] = set()
        self._lock = RLock()

    def issue(self, key: BenchmarkSampleKey) -> CriterionCapability:
        """Issue one capability for an eligible sample without reading evaluator data."""

        with self._lock:
            if key in self._pending or key in self._consumed:
                raise CriterionCapabilityError("Criterion capability was already issued")
            seal, commit_set = self._verify_eligibility(key)
            nonce = bytes(self._nonce_factory())
            if len(nonce) < 16:
                raise CriterionCapabilityError("Criterion nonce must contain at least 16 bytes")
            if nonce in self._used_nonces:
                raise CriterionCapabilityError("Criterion nonce was already used")
            capability = CriterionCapability(
                global_seal_hash=seal.seal_hash,
                condition_set_hash=commit_set.seal_hash,
                sample_key=key,
                nonce=nonce,
            )
            self._pending[key] = capability
            self._used_nonces.add(nonce)
            return capability

    def open(
        self,
        capability: CriterionCapability,
        *,
        manifest_loader: EvaluatorManifestLoader,
    ) -> VerifiedCriterionAccess:
        """Consume a capability, then and only then construct the private manifest."""

        with self._lock:
            pending = self._pending.get(capability.sample_key)
            if pending is None or not _same_capability(pending, capability):
                raise CriterionCapabilityError("Criterion capability is unknown or altered")
            seal, commit_set = self._verify_eligibility(capability.sample_key)
            if (
                seal.seal_hash != capability.global_seal_hash
                or commit_set.seal_hash != capability.condition_set_hash
            ):
                raise CriterionCapabilityError("Criterion capability no longer matches the seal")

            del self._pending[capability.sample_key]
            self._consumed.add(capability.sample_key)
            manifest = manifest_loader()
            _validate_evaluator_manifest(
                manifest,
                capability.sample_key,
                public_manifest_hash=self._run_plan.provenance.public_manifest_hash,
            )
            return VerifiedCriterionAccess(
                global_seal_hash=capability.global_seal_hash,
                condition_set_hash=capability.condition_set_hash,
                global_sealed_at_utc=seal.sealed_at_utc,
                condition_sealed_at_utc=commit_set.sealed_at_utc,
                sample_key=capability.sample_key,
                manifest=manifest,
            )

    def _verify_eligibility(
        self,
        key: BenchmarkSampleKey,
    ) -> tuple[GlobalDecisionSeal, ConditionCommitSet]:
        try:
            seal = self._global_seals.load()
        except (RuntimeError, ValueError) as error:
            raise CriterionNotSealedError("Global decision seal failed verification") from error
        if seal is None:
            raise CriterionNotSealedError("Global decision phase is not sealed")
        if seal.run_plan_hash != self._run_plan.plan_hash:
            raise CriterionNotSealedError("Global seal belongs to another run plan")
        if key not in self._run_plan.planned_samples:
            raise CriterionIneligibleError("Sample is absent from the frozen run plan")

        matches = tuple(status for status in seal.sample_statuses if status.key == key)
        if len(matches) != 1:
            raise CriterionIneligibleError("Sample does not appear exactly once in the global seal")
        status = matches[0]
        if status.status is not SampleDecisionStatus.COMPLETE:
            raise CriterionIneligibleError(
                f"Sample is not eligible for criterion access: {status.status.value}"
            )
        condition_set_hash = status.condition_set_hash
        if condition_set_hash is None or seal.complete_set_hashes.count(condition_set_hash) != 1:
            raise CriterionIneligibleError("Sample does not have one globally sealed condition set")

        try:
            commit_set = self._condition_store.get(key)
            records = self._condition_store.get_records(key)
        except (RuntimeError, ValueError) as error:
            raise CriterionIneligibleError("Sample commitment failed verification") from error
        if commit_set is None or commit_set.seal_hash != condition_set_hash:
            raise CriterionIneligibleError("Sample commitment does not match the global seal")
        if status.prediction_hashes != tuple(record.record_hash for record in records):
            raise CriterionIneligibleError("Prediction records do not match the global seal")
        if commit_set.sealed_at_utc >= seal.sealed_at_utc or any(
            record.committed_at_utc >= seal.sealed_at_utc for record in records
        ):
            raise CriterionIneligibleError("Prediction records do not precede the global seal")
        if self._criterion_record_exists(key):
            raise CriterionIneligibleError("Criterion record already exists for this sample")
        return seal, commit_set


def _validate_evaluator_manifest(
    manifest: EvaluatorBenchmarkManifest,
    key: BenchmarkSampleKey,
    *,
    public_manifest_hash: str,
) -> None:
    expected_hash = model_content_hash(
        manifest,
        exclude={"projection_hash", "projected_at_utc"},
    )
    if manifest.projection_hash != expected_hash:
        raise EvaluatorManifestAccessError("Evaluator projection hash does not match its content")
    if manifest.benchmark_version != key.benchmark_version:
        raise EvaluatorManifestAccessError("Evaluator manifest belongs to another benchmark")
    if manifest.public_projection_hash != public_manifest_hash:
        raise EvaluatorManifestAccessError(
            "Evaluator manifest does not match the run's public projection"
        )
    matches = tuple(
        criterion for criterion in manifest.criteria if criterion.case_id == key.case_id
    )
    if len(matches) != 1:
        raise EvaluatorManifestAccessError(
            "Evaluator manifest must contain exactly one criterion for the sample"
        )


def _same_capability(first: CriterionCapability, second: CriterionCapability) -> bool:
    return (
        first.global_seal_hash == second.global_seal_hash
        and first.condition_set_hash == second.condition_set_hash
        and first.sample_key == second.sample_key
        and hmac.compare_digest(first.nonce, second.nonce)
    )


def _secure_nonce() -> bytes:
    return secrets.token_bytes(32)
