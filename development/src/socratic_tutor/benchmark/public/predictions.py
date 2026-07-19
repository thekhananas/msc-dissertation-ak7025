"""Decision-safe, content-addressed benchmark prediction records."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public.models import (
    EXPECTED_CONDITIONS,
    BenchmarkCondition,
    PublicBenchmarkManifest,
)
from socratic_tutor.benchmark.public.runner import ConditionOutcome, PairedConditionRun
from socratic_tutor.benchmark.public.safety import assert_public_payload_safe
from socratic_tutor.contracts import ContractModel, PolicyDecision


class PredictionRecordError(ValueError):
    """A prediction record violates identity, provenance, or parity constraints."""


class TrackerSecondaryPrediction(ContractModel):
    """Optional outputs kept explicit when a tracker does not estimate them."""

    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty_method: str = Field(min_length=1)
    misconception_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_uncertainty_provenance(self) -> "TrackerSecondaryPrediction":
        if self.uncertainty is None and self.uncertainty_method != "not_estimated":
            raise ValueError("Missing uncertainty must use method not_estimated")
        if self.uncertainty is not None and self.uncertainty_method == "not_estimated":
            raise ValueError("Estimated uncertainty requires a declared method")
        return self


class _DecisionPredictionContent(ContractModel):
    """Prediction fields included in the canonical record digest."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.condition_prediction.v1"] = "benchmark.condition_prediction.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    model_route_id: str = Field(min_length=1)
    condition: BenchmarkCondition
    case_content_hash: Sha256
    initial_state_hash: Sha256
    public_record_hash: Sha256
    public_evidence_hash: Sha256
    additional_evidence_hash: Sha256 | None
    tracker_mastery_probability: float = Field(ge=0.0, le=1.0)
    tracker_uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    tracker_uncertainty_method: str = Field(min_length=1)
    tracker_misconception_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    decision: PolicyDecision
    policy_target_concept: str = Field(min_length=1)
    policy_propensity: float = Field(gt=0.0, le=1.0)
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    observation_schema: str = Field(min_length=1)
    input_hash: Sha256
    committed_at_utc: datetime

    @model_validator(mode="after")
    def validate_semantics(self) -> "_DecisionPredictionContent":
        if self.committed_at_utc.tzinfo is None or self.committed_at_utc.utcoffset() != timedelta(
            0
        ):
            raise ValueError("Prediction commit time must use UTC")
        if self.condition is BenchmarkCondition.DIALOGUE_ONLY:
            if self.additional_evidence_hash is not None:
                raise ValueError("Dialogue-only prediction cannot have additional evidence")
        elif self.additional_evidence_hash is None:
            raise ValueError("Probe prediction requires an additional evidence hash")
        TrackerSecondaryPrediction(
            uncertainty=self.tracker_uncertainty,
            uncertainty_method=self.tracker_uncertainty_method,
            misconception_probability=self.tracker_misconception_probability,
        )

        return self


class DecisionPredictionRecord(_DecisionPredictionContent):
    """One tracker prediction and policy action made before outcome reveal."""

    record_hash: Sha256

    @model_validator(mode="after")
    def validate_hash_and_safety(self) -> "DecisionPredictionRecord":
        if self.record_hash != prediction_record_hash(self):
            raise ValueError("Prediction record hash does not match its content")
        assert_public_payload_safe(self.model_dump(mode="json"))
        return self


def prediction_record_hash(record: DecisionPredictionRecord) -> Sha256:
    """Hash every prediction field except its self-referential record digest."""

    return model_content_hash(record, exclude={"record_hash"})


class PredictionRecordFactory:
    """Build a complete prediction set from one parity-checked condition run."""

    def __init__(self, manifest: PublicBenchmarkManifest) -> None:
        self._benchmark_version = manifest.benchmark_version
        self._cases = {case.case_id: case for case in manifest.cases}

    def build(
        self,
        paired_run: PairedConditionRun,
        *,
        run_id: str,
        sample_id: str,
        model_route_id: str,
        public_record_hash: Sha256,
        committed_at_utc: datetime,
        secondary_predictions: Mapping[
            BenchmarkCondition,
            TrackerSecondaryPrediction,
        ]
        | None = None,
        policy_propensities: Mapping[BenchmarkCondition, float] | None = None,
    ) -> tuple[DecisionPredictionRecord, ...]:
        """Create all four records with shared identity and source provenance."""

        case = self._cases.get(paired_run.case_id)
        if case is None or case.case_content_hash != paired_run.case_content_hash:
            raise PredictionRecordError("Paired run does not match the public manifest")
        secondary = (
            {
                condition: TrackerSecondaryPrediction(uncertainty_method="not_estimated")
                for condition in EXPECTED_CONDITIONS
            }
            if secondary_predictions is None
            else secondary_predictions
        )
        propensities = (
            {condition: 1.0 for condition in EXPECTED_CONDITIONS}
            if policy_propensities is None
            else policy_propensities
        )
        _require_complete_mapping("secondary predictions", secondary)
        _require_complete_mapping("policy propensities", propensities)

        records = tuple(
            _create_prediction_record(
                benchmark_version=self._benchmark_version,
                run_id=run_id,
                sample_id=sample_id,
                model_route_id=model_route_id,
                public_record_hash=public_record_hash,
                paired_run=paired_run,
                outcome=outcome,
                secondary=secondary[outcome.condition],
                policy_propensity=propensities[outcome.condition],
                committed_at_utc=committed_at_utc,
            )
            for outcome in paired_run.outcomes
        )
        _validate_record_set(records, paired_run)
        return records


def _require_complete_mapping(name: str, values: Mapping[BenchmarkCondition, object]) -> None:
    if set(values) != set(EXPECTED_CONDITIONS):
        raise PredictionRecordError(f"{name.capitalize()} must match the condition set")


def _create_prediction_record(
    *,
    benchmark_version: str,
    run_id: str,
    sample_id: str,
    model_route_id: str,
    public_record_hash: Sha256,
    paired_run: PairedConditionRun,
    outcome: ConditionOutcome,
    secondary: TrackerSecondaryPrediction,
    policy_propensity: float,
    committed_at_utc: datetime,
) -> DecisionPredictionRecord:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.condition_prediction.v1",
        "benchmark_version": benchmark_version,
        "run_id": run_id,
        "case_id": paired_run.case_id,
        "sample_id": sample_id,
        "model_route_id": model_route_id,
        "condition": outcome.condition,
        "case_content_hash": paired_run.case_content_hash,
        "initial_state_hash": outcome.initial_state_hash,
        "public_record_hash": public_record_hash,
        "public_evidence_hash": outcome.public_evidence_hash,
        "additional_evidence_hash": outcome.evidence_source_hash,
        "tracker_mastery_probability": outcome.tracker_after.mastery_probability,
        "tracker_uncertainty": secondary.uncertainty,
        "tracker_uncertainty_method": secondary.uncertainty_method,
        "tracker_misconception_probability": secondary.misconception_probability,
        "decision": outcome.decision,
        "policy_target_concept": outcome.tracker_after.concept,
        "policy_propensity": policy_propensity,
        "tracker_version": outcome.tracker_version,
        "policy_version": outcome.policy_version,
        "observation_schema": outcome.schema_id,
        "input_hash": outcome.input_hash,
        "committed_at_utc": committed_at_utc,
    }
    draft = _DecisionPredictionContent.model_validate(content)
    return DecisionPredictionRecord.model_validate(
        {
            **draft.model_dump(mode="python"),
            "record_hash": model_content_hash(draft),
        }
    )


def _validate_record_set(
    records: tuple[DecisionPredictionRecord, ...],
    paired_run: PairedConditionRun,
) -> None:
    if tuple(record.condition for record in records) != EXPECTED_CONDITIONS:
        raise PredictionRecordError("Prediction records do not contain the ordered condition set")
    if len({record.record_hash for record in records}) != len(EXPECTED_CONDITIONS):
        raise PredictionRecordError("Prediction record hashes must be unique")
    if {record.initial_state_hash for record in records} != {paired_run.initial_state_hash}:
        raise PredictionRecordError("Prediction records do not share the paired initial state")
    shared_identity = {
        (
            record.benchmark_version,
            record.run_id,
            record.case_id,
            record.sample_id,
            record.model_route_id,
            record.case_content_hash,
            record.public_record_hash,
            record.public_evidence_hash,
            record.tracker_version,
            record.policy_version,
            record.observation_schema,
            record.committed_at_utc,
        )
        for record in records
    }
    if len(shared_identity) != 1:
        raise PredictionRecordError("Prediction records do not share run and component identity")


def prediction_arrow_row(record: DecisionPredictionRecord) -> dict[str, object]:
    """Flatten one validated prediction without introducing undeclared columns."""

    return {
        "schema_version": record.schema_version,
        "schema_id": record.schema_id,
        "benchmark_version": record.benchmark_version,
        "run_id": record.run_id,
        "case_id": record.case_id,
        "sample_id": record.sample_id,
        "model_route_id": record.model_route_id,
        "condition": record.condition.value,
        "case_content_hash": record.case_content_hash,
        "initial_state_hash": record.initial_state_hash,
        "public_record_hash": record.public_record_hash,
        "public_evidence_hash": record.public_evidence_hash,
        "additional_evidence_hash": record.additional_evidence_hash,
        "tracker_mastery_probability": record.tracker_mastery_probability,
        "tracker_uncertainty": record.tracker_uncertainty,
        "tracker_uncertainty_method": record.tracker_uncertainty_method,
        "tracker_misconception_probability": record.tracker_misconception_probability,
        "directive": record.decision.action.value,
        "decision_rationale": record.decision.rationale,
        "policy_target_concept": record.policy_target_concept,
        "policy_propensity": record.policy_propensity,
        "tracker_version": record.tracker_version,
        "policy_version": record.policy_version,
        "observation_schema": record.observation_schema,
        "input_hash": record.input_hash,
        "committed_at_utc": record.committed_at_utc,
        "record_hash": record.record_hash,
    }
