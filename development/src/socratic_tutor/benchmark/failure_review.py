"""Prepare an observable, matched review packet for primary-analysis failures."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.datasets import (
    read_criterion_records,
    read_repeat_metrics,
)
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import authored_manifest_content_hash
from socratic_tutor.benchmark.evaluator.scoring import RepeatEstimator
from socratic_tutor.benchmark.failure_taxonomy import (
    FailureTaxonomy,
    failure_taxonomy_hash,
    load_failure_taxonomy,
)
from socratic_tutor.benchmark.generation import (
    EvidenceTaskPayload,
    GenerationChannel,
    PublicTaskPayload,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport, PrimaryCaseResult
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel

_SHEET_FIELDS = (
    "case_id",
    "selection_role",
    "matched_case_id",
    "target_concept",
    "misconception_id",
    "evidence_pattern",
    "criterion_outcome",
    "dialogue_score",
    "probe_score",
    "dialogue_decision",
    "probe_decision",
    "public_response",
    "evidence_response",
    "criterion_response",
    "category",
    "rationale",
    "evidence_references",
)
type SelectionRole = Literal["primary_failure", "matched_success"]


class FailureReviewError(ValueError):
    """Review sources cannot produce one complete and reproducible packet."""


class ObservableResponse(ContractModel):
    """One visible prompt and response, without hidden model reasoning."""

    channel: Literal["public", "evidence", "criterion"]
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)
    request_hash: Sha256
    response_hash: Sha256


class FailureReviewItem(ContractModel):
    """Observable evidence for one primary failure or matched success."""

    case_id: str = Field(min_length=1)
    selection_role: SelectionRole
    matched_case_id: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    misconception_id: str = Field(min_length=1)
    evidence_pattern: str = Field(min_length=1)
    criterion_outcome: bool
    criterion_execution_status: Literal["completed"]
    criterion_tests_passed: int = Field(ge=0)
    criterion_tests_failed: int = Field(ge=0)
    dialogue_score: float = Field(ge=0.0, le=1.0)
    probe_score: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    dialogue_decision: bool
    probe_decision: bool
    dialogue_correct: bool
    probe_correct: bool
    public_exchange: ObservableResponse
    evidence_exchange: ObservableResponse
    criterion_exchange: ObservableResponse
    case_content_hash: Sha256
    criterion_record_hash: Sha256
    dialogue_repeat_hash: Sha256
    probe_repeat_hash: Sha256
    aggregate_record_hash: Sha256

    @model_validator(mode="after")
    def validate_role(self) -> FailureReviewItem:
        if self.selection_role == "primary_failure" and self.probe_correct:
            raise ValueError("Primary failure must be incorrect under the probe-informed estimator")
        if self.selection_role == "matched_success" and not self.probe_correct:
            raise ValueError("Matched success must be correct under the probe-informed estimator")
        if self.dialogue_decision is not (self.dialogue_score >= self.policy_threshold):
            raise ValueError("Dialogue decision does not match the policy threshold")
        if self.probe_decision is not (self.probe_score >= self.policy_threshold):
            raise ValueError("Probe decision does not match the policy threshold")
        return self


class FailureReviewPlan(ContractModel):
    """Frozen sources and selection rule for one review packet."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_review_plan.v1"] = "benchmark.failure_review_plan.v1"
    run_id: str = Field(min_length=1)
    taxonomy_hash: Sha256
    analysis_specification_hash: Sha256
    source_manifest_hash: Sha256
    primary_report_hash: Sha256
    criterion_publication_hash: Sha256
    repeat_publication_hash: Sha256
    generation_responses_sha256: Sha256
    criterion_responses_sha256: Sha256
    matching_rule: Literal[
        "same_criterion_outcome_then_evidence_pattern_concept_misconception_case_id"
    ] = "same_criterion_outcome_then_evidence_pattern_concept_misconception_case_id"
    reuse_success_cases: Literal[False] = False
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> FailureReviewPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Failure-review plan hash does not match its content")
        return self


class FailureReviewPacket(ContractModel):
    """Equal-sized failure and success records ready for independent review."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_review_packet.v1"] = "benchmark.failure_review_packet.v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    primary_failure_count: int = Field(ge=1)
    matched_success_count: int = Field(ge=1)
    items: tuple[FailureReviewItem, ...] = Field(min_length=2)
    hidden_reasoning_included: Literal[False] = False
    packet_hash: Sha256

    @model_validator(mode="after")
    def validate_packet(self) -> FailureReviewPacket:
        failures = tuple(item for item in self.items if item.selection_role == "primary_failure")
        successes = tuple(item for item in self.items if item.selection_role == "matched_success")
        if (
            len(failures) != self.primary_failure_count
            or len(successes) != self.matched_success_count
        ):
            raise ValueError("Failure-review role counts do not reconcile")
        if self.primary_failure_count != self.matched_success_count:
            raise ValueError("Failure review requires an equal-sized matched success set")
        case_ids = {item.case_id for item in self.items}
        if len(case_ids) != len(self.items):
            raise ValueError("Failure-review cases must be unique")
        if any(item.matched_case_id not in case_ids for item in self.items):
            raise ValueError("Failure-review match points outside the packet")
        if self.packet_hash != model_content_hash(self, exclude={"packet_hash"}):
            raise ValueError("Failure-review packet hash does not match its content")
        return self


class FailureReviewPreparationReport(ContractModel):
    """Hashes for the immutable packet, instructions, and blank review sheet."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_review_preparation_report.v1"] = (
        "benchmark.failure_review_preparation_report.v1"
    )
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    packet_hash: Sha256
    primary_failure_count: int = Field(ge=1)
    matched_success_count: int = Field(ge=1)
    review_sheet_sha256: Sha256
    reviewer_instructions_sha256: Sha256
    completed_at_utc: datetime
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> FailureReviewPreparationReport:
        _require_utc(self.completed_at_utc)
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Failure-review preparation hash does not match its content")
        return self


def prepare_failure_review(
    *,
    taxonomy_path: Path,
    analysis_specification_path: Path,
    manifest_path: Path,
    primary_report_path: Path,
    dataset_root: Path,
    generation_responses_path: Path,
    criterion_responses_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> FailureReviewPreparationReport:
    """Select failures and controls, then emit a self-contained human-review workbook."""

    _require_utc(created_at_utc)
    taxonomy = load_failure_taxonomy(taxonomy_path)
    analysis_specification = load_analysis_specification(analysis_specification_path)
    manifest = load_and_verify_manifest(manifest_path)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    if manifest.manifest_hash is None:
        raise FailureReviewError("Failure review requires a frozen benchmark manifest")
    if primary.benchmark_version != manifest.benchmark_version:
        raise FailureReviewError("Primary report and benchmark manifest versions differ")
    if created_at_utc <= primary.completed_at_utc:
        raise FailureReviewError("Failure review must follow the primary analysis")

    store = AtomicParquetDatasetStore(dataset_root)
    criteria = read_criterion_records(store)
    repeats = read_repeat_metrics(store)
    criterion_rows = criteria.table.to_pylist()
    repeat_rows = repeats.table.to_pylist()
    _validate_dataset_identity(primary, criterion_rows, repeat_rows)

    generation_records = _load_recorded_responses(generation_responses_path)
    criterion_records = _load_recorded_responses(criterion_responses_path)
    response_index = _response_index((*generation_records, *criterion_records), primary.run_id)
    metadata = {case.public.case_id: case for case in manifest.cases}
    failures = sorted(
        (record for record in primary.case_results if not record.valid_evidence_correct),
        key=lambda record: record.case_id,
    )
    matches = _match_successes(failures, primary.case_results, metadata)
    match_by_failure = {failure.case_id: success.case_id for failure, success in matches}
    match_by_success = {success.case_id: failure.case_id for failure, success in matches}

    criterion_by_case = {str(row["case_id"]): row for row in criterion_rows}
    repeat_by_key = {(str(row["case_id"]), str(row["estimator"])): row for row in repeat_rows}
    selected: list[FailureReviewItem] = []
    for record in failures:
        selected.append(
            _review_item(
                record,
                role="primary_failure",
                matched_case_id=match_by_failure[record.case_id],
                metadata=metadata,
                criterion_by_case=criterion_by_case,
                repeat_by_key=repeat_by_key,
                response_index=response_index,
                policy_threshold=analysis_specification.policy_threshold,
            )
        )
    for record in sorted((success for _, success in matches), key=lambda item: item.case_id):
        selected.append(
            _review_item(
                record,
                role="matched_success",
                matched_case_id=match_by_success[record.case_id],
                metadata=metadata,
                criterion_by_case=criterion_by_case,
                repeat_by_key=repeat_by_key,
                response_index=response_index,
                policy_threshold=analysis_specification.policy_threshold,
            )
        )

    lock_hash = _read_hash(pixi_lock_path, "Pixi lock")
    plan = _make_plan(
        taxonomy=taxonomy,
        analysis_specification_hash_value=analysis_specification_hash(analysis_specification),
        source_manifest_hash=authored_manifest_content_hash(manifest),
        primary=primary,
        criterion_publication_hash=criteria.manifest.publication_hash,
        repeat_publication_hash=repeats.manifest.publication_hash,
        generation_responses_sha256=_read_hash(generation_responses_path, "generation responses"),
        criterion_responses_sha256=_read_hash(criterion_responses_path, "criterion responses"),
        analysis_code_revision=analysis_code_revision,
        lock_hash=lock_hash,
        created_at_utc=created_at_utc,
    )
    packet = _make_packet(primary.run_id, plan.plan_hash, tuple(selected))
    sheet = _review_sheet(packet)
    instructions = _reviewer_instructions(taxonomy, packet)
    report = _make_report(plan, packet, sheet, instructions, created_at_utc)

    root = output_root.resolve()
    write_immutable_json(root / "failure_review_plan.json", plan)
    write_immutable_json(root / "failure_review_packet.json", packet)
    write_immutable_bytes(root / "failure_review_sheet.csv", sheet)
    write_immutable_bytes(root / "reviewer_instructions.md", instructions)
    write_immutable_json(root / "failure_review_preparation_report.json", report)
    return report


def _match_successes(
    failures: list[PrimaryCaseResult],
    all_results: tuple[PrimaryCaseResult, ...],
    metadata: dict[str, Any],
) -> tuple[tuple[PrimaryCaseResult, PrimaryCaseResult], ...]:
    available = [record for record in all_results if record.valid_evidence_correct]
    matches: list[tuple[PrimaryCaseResult, PrimaryCaseResult]] = []
    for failure in failures:
        same_outcome = [
            candidate
            for candidate in available
            if candidate.criterion_outcome is failure.criterion_outcome
        ]
        if not same_outcome:
            raise FailureReviewError(f"No unused matched success for {failure.case_id}")
        failure_case = metadata[failure.case_id]
        selected = min(
            same_outcome,
            key=lambda candidate: (
                metadata[candidate.case_id].criterion.evidence_pattern
                != failure_case.criterion.evidence_pattern,
                metadata[candidate.case_id].public.target_concept
                != failure_case.public.target_concept,
                metadata[candidate.case_id].criterion.misconception_id
                != failure_case.criterion.misconception_id,
                candidate.case_id,
            ),
        )
        available.remove(selected)
        matches.append((failure, selected))
    return tuple(matches)


def _review_item(
    record: PrimaryCaseResult,
    *,
    role: SelectionRole,
    matched_case_id: str,
    metadata: dict[str, Any],
    criterion_by_case: dict[str, dict[str, Any]],
    repeat_by_key: dict[tuple[str, str], dict[str, Any]],
    response_index: dict[tuple[str, GenerationChannel], RecordedGenerationResponse],
    policy_threshold: float,
) -> FailureReviewItem:
    case = metadata[record.case_id]
    criterion = criterion_by_case[record.case_id]
    dialogue = repeat_by_key[(record.case_id, RepeatEstimator.DIALOGUE_ONLY.value)]
    probe = repeat_by_key[(record.case_id, RepeatEstimator.PROBE_INFORMED.value)]
    return FailureReviewItem(
        case_id=record.case_id,
        selection_role=role,
        matched_case_id=matched_case_id,
        target_concept=case.public.target_concept,
        misconception_id=case.criterion.misconception_id,
        evidence_pattern=case.criterion.evidence_pattern.value,
        criterion_outcome=record.criterion_outcome,
        criterion_execution_status=criterion["execution_status"],
        criterion_tests_passed=criterion["passed"],
        criterion_tests_failed=criterion["failed"],
        dialogue_score=dialogue["score"],
        probe_score=probe["score"],
        policy_threshold=policy_threshold,
        dialogue_decision=record.dialogue_binary_decision,
        probe_decision=record.valid_evidence_binary_decision,
        dialogue_correct=record.dialogue_correct,
        probe_correct=record.valid_evidence_correct,
        public_exchange=_observable(response_index[(record.case_id, GenerationChannel.PUBLIC)]),
        evidence_exchange=_observable(response_index[(record.case_id, GenerationChannel.EVIDENCE)]),
        criterion_exchange=_observable(
            response_index[(record.case_id, GenerationChannel.CRITERION)]
        ),
        case_content_hash=case.public.case_content_hash,
        criterion_record_hash=record.criterion_record_hash,
        dialogue_repeat_hash=record.dialogue_repeat_hash,
        probe_repeat_hash=record.valid_evidence_repeat_hash,
        aggregate_record_hash=record.aggregate_record_hash,
    )


def _observable(record: RecordedGenerationResponse) -> ObservableResponse:
    payload = record.request.task_payload
    if isinstance(payload, PublicTaskPayload):
        prompt = payload.public_interaction
    elif isinstance(payload, EvidenceTaskPayload):
        prompt = payload.evidence_probe
    else:
        prompt = payload.criterion_probe
    return ObservableResponse(
        channel=record.request.channel.value,
        prompt=prompt,
        response=record.final_response,
        request_hash=record.request.request_hash,
        response_hash=record.response_hash,
    )


def _response_index(
    records: tuple[RecordedGenerationResponse, ...], run_id: str
) -> dict[tuple[str, GenerationChannel], RecordedGenerationResponse]:
    index: dict[tuple[str, GenerationChannel], RecordedGenerationResponse] = {}
    for record in records:
        if record.request.run_id != run_id:
            raise FailureReviewError("Recorded response belongs to another run")
        key = (record.request.case_id, record.request.channel)
        if key in index:
            raise FailureReviewError(f"Duplicate recorded response for {key[0]} {key[1].value}")
        index[key] = record
    return index


def _validate_dataset_identity(
    primary: PrimaryAnalysisReport,
    criterion_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
) -> None:
    primary_ids = {record.case_id for record in primary.case_results}
    criterion_ids = {
        str(row["case_id"]) for row in criterion_rows if row["demonstrated_performance"] is not None
    }
    if primary_ids != criterion_ids:
        raise FailureReviewError("Primary and criterion eligible cases differ")
    for record in primary.case_results:
        rows = [
            row
            for row in repeat_rows
            if row["case_id"] == record.case_id
            and row["estimator"]
            in {RepeatEstimator.DIALOGUE_ONLY.value, RepeatEstimator.PROBE_INFORMED.value}
        ]
        if len(rows) != 2:
            raise FailureReviewError(f"Primary review lacks paired rows for {record.case_id}")


def _make_plan(
    *,
    taxonomy: FailureTaxonomy,
    analysis_specification_hash_value: Sha256,
    source_manifest_hash: Sha256,
    primary: PrimaryAnalysisReport,
    criterion_publication_hash: Sha256,
    repeat_publication_hash: Sha256,
    generation_responses_sha256: Sha256,
    criterion_responses_sha256: Sha256,
    analysis_code_revision: str,
    lock_hash: Sha256,
    created_at_utc: datetime,
) -> FailureReviewPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_plan.v1",
        "run_id": primary.run_id,
        "taxonomy_hash": failure_taxonomy_hash(taxonomy),
        "analysis_specification_hash": analysis_specification_hash_value,
        "source_manifest_hash": source_manifest_hash,
        "primary_report_hash": primary.report_hash,
        "criterion_publication_hash": criterion_publication_hash,
        "repeat_publication_hash": repeat_publication_hash,
        "generation_responses_sha256": generation_responses_sha256,
        "criterion_responses_sha256": criterion_responses_sha256,
        "matching_rule": (
            "same_criterion_outcome_then_evidence_pattern_concept_misconception_case_id"
        ),
        "reuse_success_cases": False,
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = FailureReviewPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return FailureReviewPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _make_packet(
    run_id: str, plan_hash: Sha256, items: tuple[FailureReviewItem, ...]
) -> FailureReviewPacket:
    failure_count = sum(item.selection_role == "primary_failure" for item in items)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_packet.v1",
        "run_id": run_id,
        "analysis_plan_hash": plan_hash,
        "primary_failure_count": failure_count,
        "matched_success_count": len(items) - failure_count,
        "items": items,
        "hidden_reasoning_included": False,
    }
    draft = FailureReviewPacket.model_construct(
        _fields_set=set(content), **content, packet_hash="0" * 64
    )
    return FailureReviewPacket.model_validate(
        {**content, "packet_hash": model_content_hash(draft, exclude={"packet_hash"})}
    )


def _review_sheet(packet: FailureReviewPacket) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_SHEET_FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in packet.items:
        writer.writerow(
            {
                "case_id": item.case_id,
                "selection_role": item.selection_role,
                "matched_case_id": item.matched_case_id,
                "target_concept": item.target_concept,
                "misconception_id": item.misconception_id,
                "evidence_pattern": item.evidence_pattern,
                "criterion_outcome": item.criterion_outcome,
                "dialogue_score": item.dialogue_score,
                "probe_score": item.probe_score,
                "dialogue_decision": item.dialogue_decision,
                "probe_decision": item.probe_decision,
                "public_response": item.public_exchange.response,
                "evidence_response": item.evidence_exchange.response,
                "criterion_response": item.criterion_exchange.response,
                "category": "",
                "rationale": "",
                "evidence_references": ";".join(
                    (
                        item.public_exchange.response_hash,
                        item.evidence_exchange.response_hash,
                        item.criterion_exchange.response_hash,
                        item.criterion_record_hash,
                        item.probe_repeat_hash,
                    )
                ),
            }
        )
    return output.getvalue().encode("utf-8")


def _reviewer_instructions(taxonomy: FailureTaxonomy, packet: FailureReviewPacket) -> bytes:
    categories = "\n".join(
        f"- `{spec.category.value}`: {spec.meaning}" for spec in taxonomy.categories
    )
    content = f"""# Failure Review Instructions

Review all {len(packet.items)} records independently. The packet contains every primary
probe-informed classification failure and an equal-sized matched success set.

Fill only `category` and `rationale` in `failure_review_sheet.csv`. Use a taxonomy category
when an observable failure is present. For a matched success with no observable failure, enter
`no_failure_observed`. Cite the visible response, score, decision, test result, or source hash
that supports the label. Do not infer hidden reasoning or a human mental state.

## Categories

{categories}
- `no_failure_observed`: The matched success contains no observable failure under this review.

`synthetic_student_inconsistency` is retained because it is the frozen taxonomy name. In this
external run, it can describe only inconsistent evaluation-model outputs; it is not evidence
about a human learner. Some categories may not apply to this route. Do not force a label.

The matching rule first requires the same criterion outcome, then prefers the same evidence
pattern, target concept, misconception, and finally case ID. Success cases are never reused.
Matching improves comparison; it does not make the records statistically representative.

Packet hash: `{packet.packet_hash}`
"""
    return content.encode("utf-8")


def _make_report(
    plan: FailureReviewPlan,
    packet: FailureReviewPacket,
    sheet: bytes,
    instructions: bytes,
    created_at_utc: datetime,
) -> FailureReviewPreparationReport:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_preparation_report.v1",
        "run_id": plan.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "packet_hash": packet.packet_hash,
        "primary_failure_count": packet.primary_failure_count,
        "matched_success_count": packet.matched_success_count,
        "review_sheet_sha256": file_sha256(sheet),
        "reviewer_instructions_sha256": file_sha256(instructions),
        "completed_at_utc": created_at_utc,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
    }
    draft = FailureReviewPreparationReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    return FailureReviewPreparationReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )


def _load_recorded_responses(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return tuple(RecordedGenerationResponse.model_validate_json(line) for line in lines if line)
    except (OSError, ValidationError) as error:
        raise FailureReviewError(f"Could not verify recorded responses: {path}") from error


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, json.JSONDecodeError) as error:
        raise FailureReviewError(f"Could not verify failure-review source: {path}") from error


def _read_hash(path: Path, label: str) -> Sha256:
    try:
        return file_sha256(path.read_bytes())
    except OSError as error:
        raise FailureReviewError(f"Could not read {label}: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Failure-review timestamp must be UTC")
