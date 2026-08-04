"""Immutable record of how the public-answer ratings were completed."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public_rating import PublicAnswerRatingReport
from socratic_tutor.contracts import ContractModel


class PublicRaterProcedure(ContractModel):
    """Declared role, access, and completion conditions for one rater."""

    rater_id: str = Field(min_length=1)
    stable_pseudonym: str = Field(min_length=1)
    relevant_experience: str = Field(min_length=1)
    completion_order: int = Field(ge=1, le=2)
    completed_independently: Literal[True] = True
    saw_other_rater_decisions: Literal[False] = False
    saw_evidence_probes: Literal[False] = False
    saw_executable_results: Literal[False] = False
    saw_criterion_material: Literal[False] = False
    additional_tools_used: tuple[str, ...] = ()


class PublicRatingProcedurePlan(ContractModel):
    """Researcher declaration written after rating and before criterion reveal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_rating_procedure_plan.v1"] = (
        "benchmark.public_rating_procedure_plan.v1"
    )
    procedure_version: Literal["public-rating-procedure-v1"]
    recorded_on: date
    recorded_by: Literal["dissertation_author"]
    public_rating_report_hash: Sha256
    methodology_clarification_hash: Sha256
    raters: tuple[PublicRaterProcedure, PublicRaterProcedure]
    supplied_materials: tuple[
        Literal["blinded_rating_csv"],
        Literal["public_category_guide"],
    ]
    public_category_guide_present: Literal[True] = True
    case_specific_scoring_key_present: Literal[False] = False
    planned_adjudicator: Literal["dissertation_author"]
    adjudication_rule: Literal["resolve_only_genuine_disagreements_before_decision_seal"]
    actual_adjudication_required: Literal[False] = False
    ratings_rerun_or_reinterpreted: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> PublicRatingProcedurePlan:
        rater_ids = tuple(item.rater_id for item in self.raters)
        pseudonyms = tuple(item.stable_pseudonym for item in self.raters)
        orders = tuple(item.completion_order for item in self.raters)
        if len(set(rater_ids)) != 2:
            raise ValueError("Public rating procedure requires two distinct rater IDs")
        if len(set(pseudonyms)) != 2:
            raise ValueError("Public rating procedure requires two distinct pseudonyms")
        if set(orders) != {1, 2}:
            raise ValueError("Public rating completion order must contain first and second")
        return self


class PublicRatingProcedureRecord(ContractModel):
    """Content-addressed procedural evidence bound to the completed ratings."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_rating_procedure_record.v1"] = (
        "benchmark.public_rating_procedure_record.v1"
    )
    procedure_version: Literal["public-rating-procedure-v1"]
    recorded_on: date
    recorded_by: Literal["dissertation_author"]
    public_rating_report_hash: Sha256
    methodology_clarification_hash: Sha256
    packet_hash: Sha256
    rating_guide_hash: Sha256
    protocol_amendment_hash: Sha256
    item_count: Literal[24]
    rater_ids: tuple[str, str]
    raters: tuple[PublicRaterProcedure, PublicRaterProcedure]
    supplied_materials: tuple[
        Literal["blinded_rating_csv"],
        Literal["public_category_guide"],
    ]
    public_category_guide_present: Literal[True] = True
    case_specific_scoring_key_present: Literal[False] = False
    planned_adjudicator: Literal["dissertation_author"]
    adjudication_rule: Literal["resolve_only_genuine_disagreements_before_decision_seal"]
    actual_disagreement_count: Literal[0]
    actual_adjudication_required: Literal[False] = False
    ratings_rerun_or_reinterpreted: Literal[False] = False
    criterion_material_remained_restricted: Literal[True] = True
    required_before_criterion_access: Literal[True] = True
    sufficient_to_unlock_criterion_access: Literal[False] = False
    record_hash: Sha256

    @model_validator(mode="after")
    def validate_record(self) -> PublicRatingProcedureRecord:
        if tuple(item.rater_id for item in self.raters) != self.rater_ids:
            raise ValueError("Procedure rater order must match the rating report")
        if self.record_hash != model_content_hash(self, exclude={"record_hash"}):
            raise ValueError("Public rating procedure hash does not match its content")
        return self


def load_public_rating_procedure_plan(path: Path) -> PublicRatingProcedurePlan:
    """Load the researcher-authored procedure declaration from YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read public rating procedure plan: {path}") from error
    return PublicRatingProcedurePlan.model_validate(raw)


def load_public_answer_rating_report(path: Path) -> PublicAnswerRatingReport:
    """Load and validate the completed public rating report."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read public rating report: {path}") from error
    return PublicAnswerRatingReport.model_validate(raw)


def freeze_public_rating_procedure(
    *,
    plan: PublicRatingProcedurePlan,
    rating_report: PublicAnswerRatingReport,
    output_path: Path,
) -> PublicRatingProcedureRecord:
    """Bind the declared procedure to the completed pre-criterion rating report."""

    if plan.public_rating_report_hash != rating_report.report_hash:
        raise ValueError("Procedure plan belongs to a different public rating report")
    if rating_report.item_count != 24:
        raise ValueError("Procedure record requires the complete 24-case rating report")
    if rating_report.disagreement_count != 0:
        raise ValueError("A disagreement requires a separate adjudication procedure record")
    plan_rater_ids = tuple(item.rater_id for item in plan.raters)
    if plan_rater_ids != rating_report.rater_ids:
        raise ValueError("Procedure raters do not match the public rating report")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.public_rating_procedure_record.v1",
        "procedure_version": plan.procedure_version,
        "recorded_on": plan.recorded_on,
        "recorded_by": plan.recorded_by,
        "public_rating_report_hash": rating_report.report_hash,
        "methodology_clarification_hash": plan.methodology_clarification_hash,
        "packet_hash": rating_report.packet_hash,
        "rating_guide_hash": rating_report.rating_guide_hash,
        "protocol_amendment_hash": rating_report.protocol_amendment_hash,
        "item_count": rating_report.item_count,
        "rater_ids": rating_report.rater_ids,
        "raters": plan.raters,
        "supplied_materials": plan.supplied_materials,
        "public_category_guide_present": plan.public_category_guide_present,
        "case_specific_scoring_key_present": plan.case_specific_scoring_key_present,
        "planned_adjudicator": plan.planned_adjudicator,
        "adjudication_rule": plan.adjudication_rule,
        "actual_disagreement_count": rating_report.disagreement_count,
        "actual_adjudication_required": plan.actual_adjudication_required,
        "ratings_rerun_or_reinterpreted": plan.ratings_rerun_or_reinterpreted,
        "criterion_material_remained_restricted": True,
        "required_before_criterion_access": True,
        "sufficient_to_unlock_criterion_access": False,
    }
    draft = PublicRatingProcedureRecord.model_construct(
        _fields_set=set(content), **content, record_hash="0" * 64
    )
    record = PublicRatingProcedureRecord.model_validate(
        {**content, "record_hash": model_content_hash(draft, exclude={"record_hash"})}
    )
    write_immutable_json(output_path, record)
    return record
