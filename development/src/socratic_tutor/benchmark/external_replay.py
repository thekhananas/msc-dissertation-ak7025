"""Network-free reconstruction of a sealed external run before analysis."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.criterion import CriterionRecord
from socratic_tutor.benchmark.evaluator.datasets import (
    publish_criterion_records,
    read_criterion_records,
)
from socratic_tutor.benchmark.external_criterion_audit import (
    ExternalCriterionIntegrityReport,
)
from socratic_tutor.benchmark.generation import GenerationChannel
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.datasets import (
    publish_condition_predictions,
    read_condition_predictions,
)
from socratic_tutor.benchmark.public.global_seal import (
    FilesystemGlobalDecisionSealStore,
    SampleDecisionStatus,
)
from socratic_tutor.benchmark.replay import (
    RecordedGenerationResponse,
    RecordedResponseGateway,
)
from socratic_tutor.contracts import ContractModel


class ExternalReplayError(ValueError):
    """The network-free external replay differs from the sealed run."""


class ExternalReplayPlan(ContractModel):
    """Exact sources and software identity for one pre-analysis replay."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_replay_plan.v1"] = "benchmark.external_replay_plan.v1"
    run_id: str = Field(min_length=1)
    postcriterion_integrity_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_seal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_response_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_response_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_code_revision: str = Field(min_length=1)
    replay_pixi_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: datetime
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_plan(self) -> ExternalReplayPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("External replay-plan hash does not match its content")
        return self


class ExternalReplayReport(ContractModel):
    """Hash comparison proving a replay made no external calls."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_replay_report.v1"] = (
        "benchmark.external_replay_report.v1"
    )
    run_id: str = Field(min_length=1)
    replay_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    postcriterion_integrity_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    replayed_decision_response_count: int = Field(ge=0)
    replayed_criterion_response_count: int = Field(ge=0)
    replayed_request_count: int = Field(ge=0)
    decision_response_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_response_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_prediction_count: int = Field(ge=0)
    condition_source_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_replay_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_source_parquet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition_replay_parquet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_record_count: int = Field(ge=0)
    criterion_source_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_replay_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_source_parquet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_replay_parquet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scored_dataset_status: Literal["not_yet_created"] = "not_yet_created"
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    gate_passed: Literal[True] = True
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> ExternalReplayReport:
        _require_utc(self.completed_at_utc)
        if self.replayed_request_count != (
            self.replayed_decision_response_count + self.replayed_criterion_response_count
        ):
            raise ValueError("External replay request counts do not reconcile")
        if (
            self.condition_source_publication_hash != self.condition_replay_publication_hash
            or self.condition_source_parquet_hash != self.condition_replay_parquet_hash
        ):
            raise ValueError("Condition prediction replay differs from its source")
        if (
            self.criterion_source_publication_hash != self.criterion_replay_publication_hash
            or self.criterion_source_parquet_hash != self.criterion_replay_parquet_hash
        ):
            raise ValueError("Criterion record replay differs from its source")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("External replay report hash does not match its content")
        return self


def run_external_preanalysis_replay(
    *,
    integrity_report_path: Path,
    decision_responses_path: Path,
    seal_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    replay_code_revision: str,
    created_at_utc: datetime,
) -> ExternalReplayReport:
    """Replay all recordings and rebuild both published datasets without external access."""

    _require_utc(created_at_utc)
    integrity = _load_json(integrity_report_path, ExternalCriterionIntegrityReport)
    decision_responses = _load_jsonl(decision_responses_path)
    criterion_responses = _load_jsonl(
        seal_root.resolve() / "criterion" / "recorded_responses.jsonl"
    )
    try:
        pixi_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ExternalReplayError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    plan = _create_plan(
        integrity=integrity,
        decision_responses=decision_responses,
        criterion_responses=criterion_responses,
        replay_code_revision=replay_code_revision,
        pixi_lock_hash=pixi_hash,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "external_replay_report.json"
    if report_path.exists():
        report = _load_json(report_path, ExternalReplayReport)
        if report.replay_plan_hash != plan.plan_hash:
            raise ExternalReplayError("Existing replay report belongs to another plan")
        return report
    write_immutable_json(root / "external_replay_plan.json", plan)

    decision_count = asyncio.run(
        replay_recorded_collection(
            decision_responses,
            allowed_channels=frozenset({GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE}),
        )
    )
    criterion_count = asyncio.run(
        replay_recorded_collection(
            criterion_responses,
            allowed_channels=frozenset({GenerationChannel.CRITERION}),
        )
    )
    comparison = replay_published_datasets(
        seal_root=seal_root,
        output_dataset_root=root / "datasets",
    )
    if (
        decision_count != integrity.planned_case_count * 2
        or criterion_count != integrity.recorded_response_count
    ):
        raise ExternalReplayError("Replayed response counts differ from the audited run")
    if (
        comparison["condition_source_publication_hash"] != integrity.decision_publication_hash
        or comparison["criterion_source_publication_hash"] != integrity.criterion_publication_hash
    ):
        raise ExternalReplayError("Replay sources differ from the post-criterion audit")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_replay_report.v1",
        "run_id": integrity.run_id,
        "replay_plan_hash": plan.plan_hash,
        "postcriterion_integrity_report_hash": integrity.report_hash,
        "replayed_decision_response_count": decision_count,
        "replayed_criterion_response_count": criterion_count,
        "replayed_request_count": decision_count + criterion_count,
        "decision_response_root_hash": plan.decision_response_root_hash,
        "criterion_response_root_hash": plan.criterion_response_root_hash,
        **comparison,
        "scored_dataset_status": "not_yet_created",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
        "gate_passed": True,
    }
    draft = ExternalReplayReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalReplayReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


async def replay_recorded_collection(
    records: tuple[RecordedGenerationResponse, ...],
    *,
    allowed_channels: frozenset[GenerationChannel],
) -> int:
    """Replay exact requests through the network-free in-memory gateway."""

    gateway = RecordedResponseGateway(records, allowed_channels=allowed_channels)
    for record in records:
        replay = await gateway.generate(record.request)
        if (
            replay.response_hash != record.response_hash
            or replay.channel is not record.request.channel
        ):
            raise ExternalReplayError("Recorded response replay differs from its source")
    return len(records)


def replay_published_datasets(*, seal_root: Path, output_dataset_root: Path) -> dict[str, object]:
    """Rebuild prediction and criterion Parquet files from immutable JSON records."""

    source_root = seal_root.resolve()
    condition_store = FilesystemConditionCommitStore(source_root / "decision")
    seal = FilesystemGlobalDecisionSealStore(source_root / "decision", condition_store).load()
    if seal is None:
        raise ExternalReplayError("Global decision seal is missing")
    complete_statuses = tuple(
        item for item in seal.sample_statuses if item.status is SampleDecisionStatus.COMPLETE
    )
    if len(complete_statuses) != len(seal.sample_statuses):
        raise ExternalReplayError("Pre-analysis replay requires complete decision cases")
    predictions = tuple(
        record for status in complete_statuses for record in condition_store.get_records(status.key)
    )
    record_files = tuple(sorted((source_root / "criterion" / "records").glob("*.json")))
    records_by_case = {
        record.key.case_id: record
        for record in (_load_json(path, CriterionRecord) for path in record_files)
    }
    criterion_records = tuple(records_by_case[status.key.case_id] for status in complete_statuses)
    if len(records_by_case) != len(complete_statuses):
        raise ExternalReplayError("Criterion record set differs from complete decision cases")

    source_store = AtomicParquetDatasetStore(source_root / "datasets")
    source_predictions = read_condition_predictions(source_store).manifest
    source_criteria = read_criterion_records(source_store).manifest
    replay_store = AtomicParquetDatasetStore(output_dataset_root)
    replay_predictions = publish_condition_predictions(
        replay_store,
        predictions,
        global_seal=seal,
        published_at_utc=source_predictions.published_at_utc,
    )
    replay_criteria = publish_criterion_records(
        replay_store,
        criterion_records,
        global_seal=seal,
        published_at_utc=source_criteria.published_at_utc,
    )
    return {
        "condition_prediction_count": len(predictions),
        "condition_source_publication_hash": source_predictions.publication_hash,
        "condition_replay_publication_hash": replay_predictions.publication_hash,
        "condition_source_parquet_hash": source_predictions.parquet_sha256,
        "condition_replay_parquet_hash": replay_predictions.parquet_sha256,
        "criterion_record_count": len(criterion_records),
        "criterion_source_publication_hash": source_criteria.publication_hash,
        "criterion_replay_publication_hash": replay_criteria.publication_hash,
        "criterion_source_parquet_hash": source_criteria.parquet_sha256,
        "criterion_replay_parquet_hash": replay_criteria.parquet_sha256,
    }


def _create_plan(
    *,
    integrity: ExternalCriterionIntegrityReport,
    decision_responses: tuple[RecordedGenerationResponse, ...],
    criterion_responses: tuple[RecordedGenerationResponse, ...],
    replay_code_revision: str,
    pixi_lock_hash: str,
    created_at_utc: datetime,
) -> ExternalReplayPlan:
    if not integrity.gate_passed:
        raise ExternalReplayError("Replay requires a passing post-criterion audit")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_replay_plan.v1",
        "run_id": integrity.run_id,
        "postcriterion_integrity_report_hash": integrity.report_hash,
        "global_seal_hash": integrity.global_seal_hash,
        "decision_response_root_hash": _response_root(decision_responses),
        "criterion_response_root_hash": _response_root(criterion_responses),
        "replay_code_revision": replay_code_revision,
        "replay_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ExternalReplayPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ExternalReplayPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _response_root(records: tuple[RecordedGenerationResponse, ...]) -> str:
    return canonical_sha256(tuple(item.response_hash for item in records))


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ExternalReplayError(f"Could not verify replay source: {path}") from error


def _load_jsonl(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExternalReplayError(f"Could not read replay responses: {path}") from error
    try:
        return tuple(
            RecordedGenerationResponse.model_validate_json(line) for line in lines if line.strip()
        )
    except ValidationError as error:
        raise ExternalReplayError("Recorded replay response is invalid") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("External replay timestamp must be UTC")
