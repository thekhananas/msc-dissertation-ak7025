"""Blinded public-answer rating contracts for the external benchmark."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    write_immutable_bytes,
    write_immutable_json,
    write_immutable_jsonl,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.external_protocol import ExternalModelExecutionProtocol
from socratic_tutor.benchmark.generation import GenerationChannel, PublicTaskPayload
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel, EvidenceCategory

_EXPECTED_CATEGORIES = tuple(EvidenceCategory)
_PACKET_ITEM_FIELDS = ("case_id", "public_prompt", "visible_response", "response_hash")
_FORBIDDEN_CONTEXT = (
    "evidence_probe",
    "evidence_result",
    "criterion_probe",
    "criterion_result",
    "authored_evidence_pattern",
    "condition_prediction",
)
_RATING_SHEET_FIELDS = (
    "case_id",
    "response_hash",
    "public_prompt",
    "visible_response",
    "category",
    "rationale",
)


class RatingCategoryRule(ContractModel):
    """One mutually exclusive public-answer category and its fixed confidence."""

    category: EvidenceCategory
    fixed_confidence: float = Field(ge=0.0, le=1.0)
    rule: str = Field(min_length=1)


class PublicAnswerRatingGuidePlan(ContractModel):
    """Reviewed source used to freeze one rating guide and amendment."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_answer_rating_guide_plan.v1"] = (
        "benchmark.public_answer_rating_guide_plan.v1"
    )
    guide_version: str = Field(min_length=1)
    base_protocol_hash: Sha256
    required_rater_count: Literal[2] = 2
    category_precedence: tuple[EvidenceCategory, ...] = Field(min_length=6, max_length=6)
    category_rules: tuple[RatingCategoryRule, ...] = Field(min_length=6, max_length=6)
    allowed_packet_fields: tuple[str, ...]
    forbidden_context: tuple[str, ...]
    agreement_method: Literal["raw_agreement_and_cohen_kappa"]
    adjudication_rule: Literal["resolve_disagreements_before_decision_seal"]

    @model_validator(mode="after")
    def validate_complete_guide(self) -> PublicAnswerRatingGuidePlan:
        categories = tuple(rule.category for rule in self.category_rules)
        if len(set(categories)) != len(_EXPECTED_CATEGORIES) or set(categories) != set(
            _EXPECTED_CATEGORIES
        ):
            raise ValueError("Rating guide must define every evidence category exactly once")
        if len(set(self.category_precedence)) != len(_EXPECTED_CATEGORIES) or set(
            self.category_precedence
        ) != set(_EXPECTED_CATEGORIES):
            raise ValueError("Rating precedence must contain every evidence category exactly once")
        if self.allowed_packet_fields != _PACKET_ITEM_FIELDS:
            raise ValueError("Rating packet fields must match the blinded public schema")
        if self.forbidden_context != _FORBIDDEN_CONTEXT:
            raise ValueError("Rating guide must list the complete forbidden context")
        return self


class PublicAnswerRatingGuide(ContractModel):
    """Content-addressed instructions fixed before held-out generation."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_answer_rating_guide.v1"] = (
        "benchmark.public_answer_rating_guide.v1"
    )
    guide_version: str = Field(min_length=1)
    required_rater_count: Literal[2] = 2
    category_precedence: tuple[EvidenceCategory, ...]
    category_rules: tuple[RatingCategoryRule, ...]
    allowed_packet_fields: tuple[str, ...]
    forbidden_context: tuple[str, ...]
    agreement_method: Literal["raw_agreement_and_cohen_kappa"]
    adjudication_rule: Literal["resolve_disagreements_before_decision_seal"]
    guide_hash: Sha256

    @model_validator(mode="after")
    def validate_guide_hash(self) -> PublicAnswerRatingGuide:
        categories = tuple(rule.category for rule in self.category_rules)
        if len(set(categories)) != len(_EXPECTED_CATEGORIES) or set(categories) != set(
            _EXPECTED_CATEGORIES
        ):
            raise ValueError("Rating guide must define every evidence category exactly once")
        if len(set(self.category_precedence)) != len(_EXPECTED_CATEGORIES) or set(
            self.category_precedence
        ) != set(_EXPECTED_CATEGORIES):
            raise ValueError("Rating precedence must contain every evidence category exactly once")
        if self.allowed_packet_fields != _PACKET_ITEM_FIELDS:
            raise ValueError("Rating packet fields must match the blinded public schema")
        if self.forbidden_context != _FORBIDDEN_CONTEXT:
            raise ValueError("Rating guide must list the complete forbidden context")
        if self.guide_hash != model_content_hash(self, exclude={"guide_hash"}):
            raise ValueError("Public-answer rating guide hash does not match its content")
        return self

    def confidence_for(self, category: EvidenceCategory) -> float:
        """Return the prespecified confidence for one final category."""

        for rule in self.category_rules:
            if rule.category is category:
                return rule.fixed_confidence
        raise ValueError(f"Rating guide has no confidence for {category.value}")


class PublicRatingProtocolAmendment(ContractModel):
    """Immutable link from the base external protocol to the rating boundary."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_rating_protocol_amendment.v1"] = (
        "benchmark.public_rating_protocol_amendment.v1"
    )
    base_protocol_hash: Sha256
    rating_guide_hash: Sha256
    required_rater_count: Literal[2] = 2
    allowed_packet_fields: tuple[str, ...]
    agreement_method: Literal["raw_agreement_and_cohen_kappa"]
    adjudication_rule: Literal["resolve_disagreements_before_decision_seal"]
    amendment_hash: Sha256

    @model_validator(mode="after")
    def validate_amendment(self) -> PublicRatingProtocolAmendment:
        if self.allowed_packet_fields != _PACKET_ITEM_FIELDS:
            raise ValueError("Protocol amendment exposes fields outside the blinded packet")
        if self.amendment_hash != model_content_hash(self, exclude={"amendment_hash"}):
            raise ValueError("Public-rating amendment hash does not match its content")
        return self


class PublicRatingFreezeResult(ContractModel):
    """Machine-readable result of freezing both rating artifacts."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_rating_freeze_result.v1"] = (
        "benchmark.public_rating_freeze_result.v1"
    )
    base_protocol_hash: Sha256
    rating_guide_hash: Sha256
    amendment_hash: Sha256


class PublicRatingPacketItem(ContractModel):
    """The only case-level values visible to a public-answer rater."""

    case_id: str = Field(min_length=1)
    public_prompt: str = Field(min_length=1)
    visible_response: str = Field(min_length=1)
    response_hash: Sha256


class PublicAnswerRatingPacket(ContractModel):
    """Blinded public-response packet with no probe, prediction, or criterion data."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_answer_rating_packet.v1"] = (
        "benchmark.public_answer_rating_packet.v1"
    )
    rating_guide_hash: Sha256
    protocol_amendment_hash: Sha256
    items: tuple[PublicRatingPacketItem, ...] = Field(min_length=1)
    packet_hash: Sha256

    @model_validator(mode="after")
    def validate_packet(self) -> PublicAnswerRatingPacket:
        case_ids = tuple(item.case_id for item in self.items)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Rating packet contains more than one public response per case")
        for item in self.items:
            if set(item.model_dump(mode="json")) != set(_PACKET_ITEM_FIELDS):
                raise ValueError("Rating packet item contains fields outside the blinded schema")
        if self.packet_hash != model_content_hash(self, exclude={"packet_hash"}):
            raise ValueError("Public-answer rating packet hash does not match its content")
        return self


class PublicAnswerRating(ContractModel):
    """One immutable independent rating authored against a blinded packet."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_answer_rating.v1"] = "benchmark.public_answer_rating.v1"
    packet_hash: Sha256
    rating_guide_hash: Sha256
    case_id: str = Field(min_length=1)
    response_hash: Sha256
    rater_id: str = Field(min_length=1)
    category: EvidenceCategory
    rationale: str = Field(min_length=1)
    rating_hash: Sha256

    @model_validator(mode="after")
    def validate_rating_hash(self) -> PublicAnswerRating:
        if self.rating_hash != model_content_hash(self, exclude={"rating_hash"}):
            raise ValueError("Public-answer rating hash does not match its content")
        return self


class PublicAnswerAdjudication(ContractModel):
    """Resolution supplied only when the two blinded ratings disagree."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_answer_adjudication.v1"] = (
        "benchmark.public_answer_adjudication.v1"
    )
    packet_hash: Sha256
    rating_guide_hash: Sha256
    case_id: str = Field(min_length=1)
    response_hash: Sha256
    rating_hashes: tuple[Sha256, Sha256]
    final_category: EvidenceCategory
    rationale: str = Field(min_length=1)
    adjudication_hash: Sha256

    @model_validator(mode="after")
    def validate_adjudication_hash(self) -> PublicAnswerAdjudication:
        if len(set(self.rating_hashes)) != 2:
            raise ValueError("Adjudication must reference two distinct rating hashes")
        if self.adjudication_hash != model_content_hash(self, exclude={"adjudication_hash"}):
            raise ValueError("Public-answer adjudication hash does not match its content")
        return self


class FinalPublicAnswerRating(ContractModel):
    """One resolved category suitable for the public tracker update."""

    case_id: str = Field(min_length=1)
    response_hash: Sha256
    category: EvidenceCategory
    confidence: float = Field(ge=0.0, le=1.0)
    resolution: Literal["agreement", "adjudicated"]
    source_hashes: tuple[Sha256, ...] = Field(min_length=2, max_length=3)


class PublicAnswerRatingReport(ContractModel):
    """Agreement diagnostics and final public labels committed before reveal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_answer_rating_report.v1"] = (
        "benchmark.public_answer_rating_report.v1"
    )
    packet_hash: Sha256
    rating_guide_hash: Sha256
    protocol_amendment_hash: Sha256
    rater_ids: tuple[str, str]
    item_count: int = Field(ge=1)
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    raw_agreement: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    kappa_status: Literal["defined", "undefined_degenerate_marginals"]
    final_ratings: tuple[FinalPublicAnswerRating, ...] = Field(min_length=1)
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> PublicAnswerRatingReport:
        if self.item_count != len(self.final_ratings):
            raise ValueError("Rating report item count does not match final ratings")
        if self.item_count != self.agreement_count + self.disagreement_count:
            raise ValueError("Rating agreement counts do not match item count")
        if self.raw_agreement != self.agreement_count / self.item_count:
            raise ValueError("Raw agreement does not match rating counts")
        if (self.cohen_kappa is None) is not (
            self.kappa_status == "undefined_degenerate_marginals"
        ):
            raise ValueError("Kappa value and status are inconsistent")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Public-answer rating report hash does not match its content")
        return self


class PublicRatingWorkbookResult(ContractModel):
    """Identity and row count for one shareable blinded rating workbook."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_rating_workbook_result.v1"] = (
        "benchmark.public_rating_workbook_result.v1"
    )
    packet_hash: Sha256
    rating_guide_hash: Sha256
    item_count: int = Field(ge=1)
    instructions_sha256: Sha256
    rating_sheet_sha256: Sha256


class PublicRatingSubmissionResult(ContractModel):
    """Identity of one complete independently authored rating set."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_rating_submission_result.v1"] = (
        "benchmark.public_rating_submission_result.v1"
    )
    packet_hash: Sha256
    rating_guide_hash: Sha256
    rater_id: str = Field(min_length=1)
    rating_count: int = Field(ge=1)
    rating_set_hash: Sha256


def load_public_rating_guide_plan(path: Path) -> PublicAnswerRatingGuidePlan:
    """Load the reviewed rating-guide plan."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read public-rating guide plan: {path}") from error
    return PublicAnswerRatingGuidePlan.model_validate(raw)


def load_public_answer_rating_guide(path: Path) -> PublicAnswerRatingGuide:
    return _load_json_model(path, PublicAnswerRatingGuide)


def load_public_rating_protocol_amendment(path: Path) -> PublicRatingProtocolAmendment:
    return _load_json_model(path, PublicRatingProtocolAmendment)


def load_public_answer_rating_packet(path: Path) -> PublicAnswerRatingPacket:
    return _load_json_model(path, PublicAnswerRatingPacket)


def freeze_public_rating_boundary(
    *,
    plan: PublicAnswerRatingGuidePlan,
    protocol: ExternalModelExecutionProtocol,
    guide_output_path: Path,
    amendment_output_path: Path,
) -> PublicRatingFreezeResult:
    """Freeze the guide and bind it to the existing immutable base protocol."""

    if plan.base_protocol_hash != protocol.protocol_hash:
        raise ValueError("Rating guide plan belongs to a different external protocol")
    if protocol.repeats_per_case != 1:
        raise ValueError("The blinded packet v1 requires exactly one public response per case")
    guide_content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_answer_rating_guide.v1",
        "guide_version": plan.guide_version,
        "required_rater_count": plan.required_rater_count,
        "category_precedence": plan.category_precedence,
        "category_rules": plan.category_rules,
        "allowed_packet_fields": plan.allowed_packet_fields,
        "forbidden_context": plan.forbidden_context,
        "agreement_method": plan.agreement_method,
        "adjudication_rule": plan.adjudication_rule,
    }
    guide = PublicAnswerRatingGuide.model_validate(
        {**guide_content, "guide_hash": canonical_sha256(guide_content)}
    )
    amendment_content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_rating_protocol_amendment.v1",
        "base_protocol_hash": protocol.protocol_hash,
        "rating_guide_hash": guide.guide_hash,
        "required_rater_count": guide.required_rater_count,
        "allowed_packet_fields": guide.allowed_packet_fields,
        "agreement_method": guide.agreement_method,
        "adjudication_rule": guide.adjudication_rule,
    }
    amendment = PublicRatingProtocolAmendment.model_validate(
        {**amendment_content, "amendment_hash": canonical_sha256(amendment_content)}
    )
    write_immutable_json(guide_output_path, guide)
    write_immutable_json(amendment_output_path, amendment)
    return PublicRatingFreezeResult(
        base_protocol_hash=protocol.protocol_hash,
        rating_guide_hash=guide.guide_hash,
        amendment_hash=amendment.amendment_hash,
    )


def build_public_answer_rating_packet(
    *,
    protocol: ExternalModelExecutionProtocol,
    amendment: PublicRatingProtocolAmendment,
    recorded_responses_path: Path,
    output_path: Path,
) -> PublicAnswerRatingPacket:
    """Project public responses into the only fields visible to raters."""

    _validate_protocol_amendment(protocol, amendment)
    records = _load_recorded_responses(recorded_responses_path)
    public_records = tuple(
        record for record in records if record.request.channel is GenerationChannel.PUBLIC
    )
    if not public_records:
        raise ValueError("Rating packet requires at least one public response")
    items: list[PublicRatingPacketItem] = []
    for record in public_records:
        _validate_record_against_protocol(record, protocol)
        payload = cast(PublicTaskPayload, record.request.task_payload)
        items.append(
            PublicRatingPacketItem(
                case_id=record.request.case_id,
                public_prompt=payload.public_interaction,
                visible_response=record.final_response,
                response_hash=record.response_hash,
            )
        )
    ordered_items = tuple(sorted(items, key=lambda item: item.case_id))
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_answer_rating_packet.v1",
        "rating_guide_hash": amendment.rating_guide_hash,
        "protocol_amendment_hash": amendment.amendment_hash,
        "items": ordered_items,
    }
    packet = PublicAnswerRatingPacket.model_validate(
        {**content, "packet_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_path, packet)
    return packet


def export_public_rating_workbook(
    *,
    guide: PublicAnswerRatingGuide,
    packet: PublicAnswerRatingPacket,
    output_root: Path,
) -> PublicRatingWorkbookResult:
    """Create a human-readable guide and spreadsheet without hidden benchmark context."""

    if packet.rating_guide_hash != guide.guide_hash:
        raise ValueError("Public-rating packet belongs to a different guide")
    instructions = _rating_instructions(guide=guide, packet=packet)
    sheet = _rating_sheet(packet)
    root = output_root.resolve()
    write_immutable_bytes(root / "instructions.md", instructions)
    write_immutable_bytes(root / "rating_sheet.csv", sheet)
    return PublicRatingWorkbookResult(
        packet_hash=packet.packet_hash,
        rating_guide_hash=guide.guide_hash,
        item_count=len(packet.items),
        instructions_sha256=file_sha256(instructions),
        rating_sheet_sha256=file_sha256(sheet),
    )


def record_public_answer_ratings(
    *,
    guide: PublicAnswerRatingGuide,
    packet: PublicAnswerRatingPacket,
    completed_sheet_path: Path,
    rater_id: str,
    output_path: Path,
) -> PublicRatingSubmissionResult:
    """Validate one completed CSV and convert it to content-addressed rating JSONL."""

    if packet.rating_guide_hash != guide.guide_hash:
        raise ValueError("Public-rating packet belongs to a different guide")
    if not rater_id.strip():
        raise ValueError("Public-rating submission requires a non-empty rater ID")
    rows = _load_completed_rating_sheet(completed_sheet_path)
    items = {item.case_id: item for item in packet.items}
    if len(rows) != len(items):
        raise ValueError("Completed rating sheet must contain every packet case exactly once")
    ratings: list[PublicAnswerRating] = []
    seen: set[str] = set()
    for row in rows:
        case_id = row["case_id"]
        if case_id in seen:
            raise ValueError(f"Completed rating sheet contains a duplicate case: {case_id}")
        seen.add(case_id)
        item = items.get(case_id)
        if item is None:
            raise ValueError(f"Completed rating sheet contains an unknown case: {case_id}")
        if (
            row["response_hash"] != item.response_hash
            or row["public_prompt"] != item.public_prompt
            or row["visible_response"] != item.visible_response
        ):
            raise ValueError(f"Completed rating sheet changed blinded content for {case_id}")
        try:
            category = EvidenceCategory(row["category"].strip())
        except ValueError as error:
            raise ValueError(
                f"Completed rating sheet has an invalid category for {case_id}"
            ) from error
        rationale = row["rationale"].strip()
        if not rationale:
            raise ValueError(f"Completed rating sheet requires a rationale for {case_id}")
        ratings.append(
            create_public_answer_rating(
                packet=packet,
                guide=guide,
                case_id=case_id,
                rater_id=rater_id.strip(),
                category=category,
                rationale=rationale,
            )
        )
    if seen != set(items):
        raise ValueError("Completed rating sheet does not cover every packet case")
    ordered = tuple(sorted(ratings, key=lambda rating: rating.case_id))
    write_immutable_jsonl(output_path, ordered)
    return PublicRatingSubmissionResult(
        packet_hash=packet.packet_hash,
        rating_guide_hash=guide.guide_hash,
        rater_id=rater_id.strip(),
        rating_count=len(ordered),
        rating_set_hash=canonical_sha256([rating.model_dump(mode="json") for rating in ordered]),
    )


def finalize_public_answer_ratings(
    *,
    guide: PublicAnswerRatingGuide,
    amendment: PublicRatingProtocolAmendment,
    packet: PublicAnswerRatingPacket,
    ratings_a_path: Path,
    ratings_b_path: Path,
    adjudications_path: Path | None,
    output_path: Path,
) -> PublicAnswerRatingReport:
    """Validate two complete independent rating sets and resolve disagreements."""

    _validate_guide_amendment_packet(guide, amendment, packet)
    ratings_a = _load_jsonl_models(ratings_a_path, PublicAnswerRating)
    ratings_b = _load_jsonl_models(ratings_b_path, PublicAnswerRating)
    items = {item.case_id: item for item in packet.items}
    first_id, first = _index_ratings(ratings_a, items, guide, packet)
    second_id, second = _index_ratings(ratings_b, items, guide, packet)
    if first_id == second_id:
        raise ValueError("Independent rating files must use different rater IDs")
    pairs = tuple((first[case_id], second[case_id]) for case_id in sorted(items))
    disagreements = tuple(pair for pair in pairs if pair[0].category is not pair[1].category)
    adjudications = (
        ()
        if adjudications_path is None
        else _load_jsonl_models(adjudications_path, PublicAnswerAdjudication)
    )
    adjudication_by_case = _validate_adjudications(
        adjudications,
        disagreements,
        packet,
        guide,
    )
    final: list[FinalPublicAnswerRating] = []
    for left, right in pairs:
        if left.category is right.category:
            category = left.category
            resolution: Literal["agreement", "adjudicated"] = "agreement"
            source_hashes: tuple[Sha256, ...] = (left.rating_hash, right.rating_hash)
        else:
            adjudication = adjudication_by_case[left.case_id]
            category = adjudication.final_category
            resolution = "adjudicated"
            source_hashes = (
                left.rating_hash,
                right.rating_hash,
                adjudication.adjudication_hash,
            )
        final.append(
            FinalPublicAnswerRating(
                case_id=left.case_id,
                response_hash=left.response_hash,
                category=category,
                confidence=guide.confidence_for(category),
                resolution=resolution,
                source_hashes=source_hashes,
            )
        )
    agreement_count = len(pairs) - len(disagreements)
    kappa, kappa_status = _cohen_kappa(pairs)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_answer_rating_report.v1",
        "packet_hash": packet.packet_hash,
        "rating_guide_hash": guide.guide_hash,
        "protocol_amendment_hash": amendment.amendment_hash,
        "rater_ids": (first_id, second_id),
        "item_count": len(pairs),
        "agreement_count": agreement_count,
        "disagreement_count": len(disagreements),
        "raw_agreement": agreement_count / len(pairs),
        "cohen_kappa": kappa,
        "kappa_status": kappa_status,
        "final_ratings": tuple(final),
    }
    report = PublicAnswerRatingReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_path, report)
    return report


def create_public_answer_rating(
    *,
    packet: PublicAnswerRatingPacket,
    guide: PublicAnswerRatingGuide,
    case_id: str,
    rater_id: str,
    category: EvidenceCategory,
    rationale: str,
) -> PublicAnswerRating:
    """Construct one content-addressed rating for authored fixtures or rating tools."""

    item = next((candidate for candidate in packet.items if candidate.case_id == case_id), None)
    if item is None:
        raise ValueError(f"Rating case is not in the packet: {case_id}")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_answer_rating.v1",
        "packet_hash": packet.packet_hash,
        "rating_guide_hash": guide.guide_hash,
        "case_id": case_id,
        "response_hash": item.response_hash,
        "rater_id": rater_id,
        "category": category,
        "rationale": rationale,
    }
    return PublicAnswerRating.model_validate({**content, "rating_hash": canonical_sha256(content)})


def create_public_answer_adjudication(
    *,
    packet: PublicAnswerRatingPacket,
    guide: PublicAnswerRatingGuide,
    left: PublicAnswerRating,
    right: PublicAnswerRating,
    final_category: EvidenceCategory,
    rationale: str,
) -> PublicAnswerAdjudication:
    """Construct one content-addressed resolution for a genuine disagreement."""

    if left.case_id != right.case_id or left.response_hash != right.response_hash:
        raise ValueError("Adjudication ratings must describe the same public response")
    if left.category is right.category:
        raise ValueError("Adjudication is not allowed when independent ratings agree")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_answer_adjudication.v1",
        "packet_hash": packet.packet_hash,
        "rating_guide_hash": guide.guide_hash,
        "case_id": left.case_id,
        "response_hash": left.response_hash,
        "rating_hashes": (left.rating_hash, right.rating_hash),
        "final_category": final_category,
        "rationale": rationale,
    }
    return PublicAnswerAdjudication.model_validate(
        {**content, "adjudication_hash": canonical_sha256(content)}
    )


def _rating_instructions(
    *, guide: PublicAnswerRatingGuide, packet: PublicAnswerRatingPacket
) -> bytes:
    rules = "\n".join(f"- `{rule.category.value}`: {rule.rule}" for rule in guide.category_rules)
    precedence = " > ".join(category.value for category in guide.category_precedence)
    content = f"""# Public Answer Rating

Rate all {len(packet.items)} answers independently. Use only the prompt and visible response
in the supplied CSV. Do not discuss ratings with the other rater until both completed files
have been returned.

## Categories

{rules}

If more than one category seems possible, use this precedence order:

`{precedence}`

## What To Enter

For every row, fill in `category` and a short `rationale`. Do not change the other columns.
The rationale should point to the part of the visible answer that determined the category.

Do not look for, request, or use evidence probes, executable results, final test questions,
expected labels, or another rater's decisions.

Packet hash: `{packet.packet_hash}`

Guide hash: `{guide.guide_hash}`
"""
    return content.encode("utf-8")


def _rating_sheet(packet: PublicAnswerRatingPacket) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_RATING_SHEET_FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in packet.items:
        writer.writerow(
            {
                "case_id": item.case_id,
                "response_hash": item.response_hash,
                "public_prompt": item.public_prompt,
                "visible_response": item.visible_response,
                "category": "",
                "rationale": "",
            }
        )
    return output.getvalue().encode("utf-8")


def _load_completed_rating_sheet(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _RATING_SHEET_FIELDS:
                raise ValueError("Completed rating sheet columns do not match the template")
            raw_rows = tuple(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"Could not read completed public-rating sheet: {path}") from error
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("Completed rating sheet contains malformed columns")
        rows.append({key: cast(str, value) for key, value in raw.items()})
    return tuple(rows)


def _validate_protocol_amendment(
    protocol: ExternalModelExecutionProtocol,
    amendment: PublicRatingProtocolAmendment,
) -> None:
    if amendment.base_protocol_hash != protocol.protocol_hash:
        raise ValueError("Public-rating amendment belongs to a different external protocol")


def _validate_guide_amendment_packet(
    guide: PublicAnswerRatingGuide,
    amendment: PublicRatingProtocolAmendment,
    packet: PublicAnswerRatingPacket,
) -> None:
    if amendment.rating_guide_hash != guide.guide_hash:
        raise ValueError("Public-rating amendment belongs to a different guide")
    if packet.rating_guide_hash != guide.guide_hash:
        raise ValueError("Public-rating packet belongs to a different guide")
    if packet.protocol_amendment_hash != amendment.amendment_hash:
        raise ValueError("Public-rating packet belongs to a different amendment")


def _validate_record_against_protocol(
    record: RecordedGenerationResponse,
    protocol: ExternalModelExecutionProtocol,
) -> None:
    request = record.request
    if request.channel is not GenerationChannel.PUBLIC:
        raise ValueError("Only public responses can enter the rating packet")
    if request.model_route.provider != protocol.provider_id:
        raise ValueError("Public response provider differs from the frozen protocol")
    if request.model_route.model != protocol.requested_model_id:
        raise ValueError("Public response model differs from the frozen protocol")
    if request.system_prompt_version != protocol.system_prompt_version:
        raise ValueError("Public response prompt version differs from the frozen protocol")
    if request.system_prompt_sha256 != protocol.system_prompt_sha256:
        raise ValueError("Public response prompt hash differs from the frozen protocol")
    if request.sampling.temperature != protocol.temperature:
        raise ValueError("Public response temperature differs from the frozen protocol")
    if request.sampling.max_output_tokens != protocol.max_output_tokens:
        raise ValueError("Public response token limit differs from the frozen protocol")
    seed_offset = (request.sampling.seed - protocol.run_seed) % (2**32)
    if seed_offset >= protocol.request_budget:
        raise ValueError("Public response seed falls outside the frozen protocol budget")


def _index_ratings(
    ratings: tuple[PublicAnswerRating, ...],
    items: dict[str, PublicRatingPacketItem],
    guide: PublicAnswerRatingGuide,
    packet: PublicAnswerRatingPacket,
) -> tuple[str, dict[str, PublicAnswerRating]]:
    if not ratings:
        raise ValueError("Each independent rating file must contain every packet item")
    rater_ids = {rating.rater_id for rating in ratings}
    if len(rater_ids) != 1:
        raise ValueError("One independent rating file must contain exactly one rater ID")
    indexed: dict[str, PublicAnswerRating] = {}
    for rating in ratings:
        item = items.get(rating.case_id)
        if item is None:
            raise ValueError(f"Rating refers to a case outside the packet: {rating.case_id}")
        if rating.case_id in indexed:
            raise ValueError(f"Duplicate rating for case: {rating.case_id}")
        if rating.packet_hash != packet.packet_hash:
            raise ValueError("Rating belongs to a different packet")
        if rating.rating_guide_hash != guide.guide_hash:
            raise ValueError("Rating belongs to a different guide")
        if rating.response_hash != item.response_hash:
            raise ValueError("Rating response hash does not match the packet")
        indexed[rating.case_id] = rating
    if set(indexed) != set(items):
        raise ValueError("Independent rating file does not cover every packet case")
    return next(iter(rater_ids)), indexed


def _validate_adjudications(
    adjudications: tuple[PublicAnswerAdjudication, ...],
    disagreements: tuple[tuple[PublicAnswerRating, PublicAnswerRating], ...],
    packet: PublicAnswerRatingPacket,
    guide: PublicAnswerRatingGuide,
) -> dict[str, PublicAnswerAdjudication]:
    expected = {left.case_id: (left, right) for left, right in disagreements}
    indexed: dict[str, PublicAnswerAdjudication] = {}
    for adjudication in adjudications:
        pair = expected.get(adjudication.case_id)
        if pair is None:
            raise ValueError("Adjudication exists for a case without a rating disagreement")
        if adjudication.case_id in indexed:
            raise ValueError(f"Duplicate adjudication for case: {adjudication.case_id}")
        if adjudication.packet_hash != packet.packet_hash:
            raise ValueError("Adjudication belongs to a different packet")
        if adjudication.rating_guide_hash != guide.guide_hash:
            raise ValueError("Adjudication belongs to a different guide")
        if adjudication.response_hash != pair[0].response_hash:
            raise ValueError("Adjudication response hash does not match the ratings")
        if set(adjudication.rating_hashes) != {pair[0].rating_hash, pair[1].rating_hash}:
            raise ValueError("Adjudication does not reference the two independent ratings")
        indexed[adjudication.case_id] = adjudication
    if set(indexed) != set(expected):
        raise ValueError("Every rating disagreement requires one adjudication")
    return indexed


def _cohen_kappa(
    pairs: tuple[tuple[PublicAnswerRating, PublicAnswerRating], ...],
) -> tuple[float | None, Literal["defined", "undefined_degenerate_marginals"]]:
    count = len(pairs)
    observed = sum(left.category is right.category for left, right in pairs) / count
    left_counts = Counter(left.category for left, _ in pairs)
    right_counts = Counter(right.category for _, right in pairs)
    expected = sum(
        (left_counts[category] / count) * (right_counts[category] / count)
        for category in _EXPECTED_CATEGORIES
    )
    if abs(1.0 - expected) < 1e-12:
        return None, "undefined_degenerate_marginals"
    return (observed - expected) / (1.0 - expected), "defined"


def _load_recorded_responses(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    return _load_jsonl_models(path, RecordedGenerationResponse)


def _load_json_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON artifact: {path}") from error
    return model.model_validate(raw)


def _load_jsonl_models[ModelT: ContractModel](
    path: Path,
    model: type[ModelT],
) -> tuple[ModelT, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read JSONL artifact: {path}") from error
    values: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        values.append(model.model_validate(raw))
    return tuple(values)
