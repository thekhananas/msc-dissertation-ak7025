"""Public contracts for the deterministic benchmark replay."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.contracts.models import ContractModel


class ReplayPhase(StrEnum):
    """The two visible stages of the benchmark replay."""

    PREDICTIONS_COMMITTED = "predictions_committed"
    OUTCOME_REVEALED = "outcome_revealed"


class StartBenchmarkReplayRequest(ContractModel):
    """Idempotency input for starting one local replay."""

    idempotency_key: str = Field(min_length=1, max_length=200)


class CommittedPredictionView(ContractModel):
    """One prediction saved before the later task became available."""

    condition: Literal[
        "dialogue_only",
        "probe_informed",
        "unrelated_probe",
        "corrupted_probe",
    ]
    label: str = Field(min_length=1)
    tracker_score: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    predicts_success: bool
    committed_at_utc: datetime
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReplayEvidenceView(ContractModel):
    """Recorded evidence that was available before the later task."""

    condition: Literal["probe_informed", "unrelated_probe"]
    label: str = Field(min_length=1)
    response_summary: str = Field(min_length=1)
    execution_summary: str = Field(min_length=1)
    passed_checks: int = Field(ge=0)
    failed_checks: int = Field(ge=0)


class RecordedOutcomeView(ContractModel):
    """The later result, returned only by the reveal action."""

    summary: str = Field(min_length=1)
    test_warning: str = Field(min_length=1)
    passed_checks: int = Field(ge=0)
    failed_checks: int = Field(ge=0)
    revealed_at_utc: datetime


class BenchmarkReplayArtifact(ContractModel):
    """Checked-in public projection of one frozen benchmark case."""

    schema_version: Literal[1] = 1
    schema_id: Literal["demo.benchmark_replay_artifact.v1"] = "demo.benchmark_replay_artifact.v1"
    source_data_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    evaluation_model: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    evidence: tuple[ReplayEvidenceView, ...] = Field(min_length=1)
    predictions: tuple[CommittedPredictionView, ...] = Field(min_length=1)
    outcome: RecordedOutcomeView
    interpretation: str = Field(min_length=1)
    predictions_sealed_before_outcome_reveal: Literal[True] = True
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reveal_order(self) -> BenchmarkReplayArtifact:
        conditions = [prediction.condition for prediction in self.predictions]
        if len(set(conditions)) != len(conditions):
            raise ValueError("Replay artifact contains duplicate prediction conditions")
        if any(
            prediction.committed_at_utc >= self.outcome.revealed_at_utc
            for prediction in self.predictions
        ):
            raise ValueError("Replay predictions must precede the recorded outcome")
        return self


class BenchmarkReplaySnapshot(ContractModel):
    """One visible replay stage with explicit offline-call accounting."""

    replay_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase: ReplayPhase
    source: Literal["recorded_artifact"] = "recorded_artifact"
    source_data_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_version: str
    case_id: str
    evaluation_model: str
    public_summary: str
    evidence: tuple[ReplayEvidenceView, ...] = Field(min_length=1)
    predictions: tuple[CommittedPredictionView, ...] = Field(min_length=1)
    outcome: RecordedOutcomeView | None
    interpretation: str | None
    predictions_sealed_before_outcome_reveal: Literal[True] = True
    model_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    post_reveal_policy_calls_made: Literal[0] = 0
    claim_boundary: str

    @model_validator(mode="after")
    def validate_phase(self) -> BenchmarkReplaySnapshot:
        if self.phase is ReplayPhase.PREDICTIONS_COMMITTED and self.outcome is not None:
            raise ValueError("Committed replay stage cannot include the later outcome")
        if self.phase is ReplayPhase.OUTCOME_REVEALED and self.outcome is None:
            raise ValueError("Revealed replay stage must include the later outcome")
        if self.phase is ReplayPhase.PREDICTIONS_COMMITTED and self.interpretation is not None:
            raise ValueError("Committed replay stage cannot interpret the hidden outcome")
        if self.phase is ReplayPhase.OUTCOME_REVEALED and self.interpretation is None:
            raise ValueError("Revealed replay stage must include its interpretation")
        return self


class ExperimentFindingView(ContractModel):
    """One checked finding prepared for the public experiment view."""

    study_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    question: str = Field(min_length=1)
    result: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    sample: str = Field(min_length=1)
    evidence_type: Literal[
        "external_model_benchmark",
        "glass_box_simulation",
        "external_learner_records",
    ]
    status: str = Field(min_length=1)
    source_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentSummaryArtifact(ContractModel):
    """Small public projection of the dissertation's completed studies."""

    schema_version: Literal[1] = 1
    schema_id: Literal["demo.experiment_summary.v1"] = "demo.experiment_summary.v1"
    overarching_question: str = Field(min_length=1)
    findings: tuple[ExperimentFindingView, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
