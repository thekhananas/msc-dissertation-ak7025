"""Reproducible agreement audit for the two blinded public-answer raters."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public_rating import PublicAnswerRating, PublicAnswerRatingReport
from socratic_tutor.benchmark.public_rating_procedure import PublicRatingProcedureRecord
from socratic_tutor.contracts import ContractModel, EvidenceCategory

_CATEGORIES = tuple(EvidenceCategory)


class RatingReliabilityError(ValueError):
    """Rating files or procedure records do not describe the same blinded review."""


class RatingCategoryCount(ContractModel):
    """One category's marginal count for both raters and the resolved labels."""

    category: EvidenceCategory
    rater_a_count: int = Field(ge=0)
    rater_b_count: int = Field(ge=0)
    final_count: int = Field(ge=0)


class RatingConfusionCell(ContractModel):
    """One cell in the complete rater-A by rater-B confusion matrix."""

    rater_a_category: EvidenceCategory
    rater_b_category: EvidenceCategory
    count: int = Field(ge=0)


class RatingDisagreement(ContractModel):
    """One case for which the blinded raters selected different categories."""

    case_id: str = Field(min_length=1)
    rater_a_category: EvidenceCategory
    rater_b_category: EvidenceCategory


class RatingReliabilityPlan(ContractModel):
    """Immutable inputs and software identity for the agreement audit."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.rating_reliability_plan.v1"] = (
        "benchmark.rating_reliability_plan.v1"
    )
    public_rating_report_hash: Sha256
    public_rating_procedure_hash: Sha256
    rater_a_file_hash: Sha256
    rater_b_file_hash: Sha256
    method: Literal["raw_agreement_unweighted_kappa_full_confusion_v1"] = (
        "raw_agreement_unweighted_kappa_full_confusion_v1"
    )
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> RatingReliabilityPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Rating-reliability plan hash does not match its content")
        return self


class RatingReliabilityReport(ContractModel):
    """Agreement, marginals, confusion matrix, and interpretation limits."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.rating_reliability_report.v1"] = (
        "benchmark.rating_reliability_report.v1"
    )
    analysis_plan_hash: Sha256
    rater_ids: tuple[str, str]
    item_count: Literal[24]
    agreement_count: int = Field(ge=0, le=24)
    disagreement_count: int = Field(ge=0, le=24)
    raw_agreement: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    kappa_status: Literal["defined", "undefined_degenerate_marginals"]
    category_counts: tuple[RatingCategoryCount, ...] = Field(min_length=6, max_length=6)
    confusion_matrix: tuple[RatingConfusionCell, ...] = Field(min_length=36, max_length=36)
    disagreements: tuple[RatingDisagreement, ...]
    adjudication_count: int = Field(ge=0, le=24)
    adjudicated_case_ids: tuple[str, ...]
    interpretation_limits: tuple[
        Literal["agreement_does_not_establish_label_correctness"],
        Literal["two_raters_are_not_a_population_sample"],
        Literal["sparse_categories_limit_kappa_interpretation"],
        Literal["ratings_cover_public_answers_not_criterion_performance"],
    ]
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> RatingReliabilityReport:
        _require_utc(self.completed_at_utc)
        if self.item_count != self.agreement_count + self.disagreement_count:
            raise ValueError("Rating agreement counts do not reconcile")
        if abs(self.raw_agreement - self.agreement_count / self.item_count) > 1e-12:
            raise ValueError("Raw agreement does not match its counts")
        if (self.cohen_kappa is None) is not (
            self.kappa_status == "undefined_degenerate_marginals"
        ):
            raise ValueError("Kappa value and status are inconsistent")
        if tuple(record.category for record in self.category_counts) != _CATEGORIES:
            raise ValueError("Rating marginals do not contain every category in canonical order")
        expected_cells = tuple((left, right) for left in _CATEGORIES for right in _CATEGORIES)
        actual_cells = tuple(
            (record.rater_a_category, record.rater_b_category) for record in self.confusion_matrix
        )
        if actual_cells != expected_cells:
            raise ValueError("Rating confusion matrix is incomplete or out of order")
        if sum(record.count for record in self.confusion_matrix) != self.item_count:
            raise ValueError("Rating confusion matrix does not sum to the item count")
        if any(
            sum(getattr(record, field) for record in self.category_counts) != self.item_count
            for field in ("rater_a_count", "rater_b_count", "final_count")
        ):
            raise ValueError("Rating category marginals do not sum to the item count")
        counts = {record.category: record for record in self.category_counts}
        for category in _CATEGORIES:
            row_total = sum(
                record.count
                for record in self.confusion_matrix
                if record.rater_a_category is category
            )
            column_total = sum(
                record.count
                for record in self.confusion_matrix
                if record.rater_b_category is category
            )
            if row_total != counts[category].rater_a_count:
                raise ValueError("Rating confusion row differs from its marginal count")
            if column_total != counts[category].rater_b_count:
                raise ValueError("Rating confusion column differs from its marginal count")
        diagonal = sum(
            record.count
            for record in self.confusion_matrix
            if record.rater_a_category is record.rater_b_category
        )
        if diagonal != self.agreement_count:
            raise ValueError("Rating confusion diagonal differs from agreement count")
        if len(self.disagreements) != self.disagreement_count:
            raise ValueError("Rating disagreement records do not reconcile")
        if self.adjudication_count != len(self.adjudicated_case_ids):
            raise ValueError("Rating adjudication records do not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Rating-reliability report hash does not match its content")
        return self


def run_rating_reliability(
    *,
    rating_report_path: Path,
    procedure_record_path: Path,
    ratings_a_path: Path,
    ratings_b_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> RatingReliabilityReport:
    """Reconstruct the complete agreement table from two immutable rating files."""

    _require_utc(created_at_utc)
    rating_report = _load_json(rating_report_path, PublicAnswerRatingReport)
    procedure = _load_json(procedure_record_path, PublicRatingProcedureRecord)
    ratings_a = _load_jsonl(ratings_a_path)
    ratings_b = _load_jsonl(ratings_b_path)
    if procedure.public_rating_report_hash != rating_report.report_hash:
        raise RatingReliabilityError("Rating procedure belongs to another rating report")
    if procedure.rater_ids != rating_report.rater_ids:
        raise RatingReliabilityError("Rating procedure and report use different raters")
    if procedure.actual_disagreement_count != rating_report.disagreement_count:
        raise RatingReliabilityError("Rating procedure and report disagree about adjudication")
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
        rater_a_hash = file_sha256(ratings_a_path.read_bytes())
        rater_b_hash = file_sha256(ratings_b_path.read_bytes())
    except OSError as error:
        raise RatingReliabilityError("Could not hash rating-reliability input") from error

    first = _index_ratings(ratings_a, rating_report, expected_rater=rating_report.rater_ids[0])
    second = _index_ratings(ratings_b, rating_report, expected_rater=rating_report.rater_ids[1])
    pairs = tuple((first[case_id], second[case_id]) for case_id in sorted(first))
    _validate_report_sources(rating_report, pairs)
    plan = _create_plan(
        rating_report=rating_report,
        procedure=procedure,
        rater_a_file_hash=rater_a_hash,
        rater_b_file_hash=rater_b_hash,
        lock_hash=lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "rating_reliability_report.json"
    if report_path.exists():
        report = _load_json(report_path, RatingReliabilityReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise RatingReliabilityError("Existing reliability report belongs to another plan")
        return report
    write_immutable_json(root / "rating_reliability_plan.json", plan)

    counts_a = Counter(record.category for record, _ in pairs)
    counts_b = Counter(record.category for _, record in pairs)
    final_counts = Counter(record.category for record in rating_report.final_ratings)
    confusion = Counter((left.category, right.category) for left, right in pairs)
    disagreements = tuple(
        RatingDisagreement(
            case_id=left.case_id,
            rater_a_category=left.category,
            rater_b_category=right.category,
        )
        for left, right in pairs
        if left.category is not right.category
    )
    category_counts = tuple(
        RatingCategoryCount(
            category=category,
            rater_a_count=counts_a[category],
            rater_b_count=counts_b[category],
            final_count=final_counts[category],
        )
        for category in _CATEGORIES
    )
    matrix = tuple(
        RatingConfusionCell(
            rater_a_category=left,
            rater_b_category=right,
            count=confusion[(left, right)],
        )
        for left in _CATEGORIES
        for right in _CATEGORIES
    )
    kappa, kappa_status = _cohen_kappa(pairs)
    adjudicated = tuple(
        record.case_id
        for record in rating_report.final_ratings
        if record.resolution == "adjudicated"
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.rating_reliability_report.v1",
        "analysis_plan_hash": plan.plan_hash,
        "rater_ids": rating_report.rater_ids,
        "item_count": rating_report.item_count,
        "agreement_count": rating_report.agreement_count,
        "disagreement_count": rating_report.disagreement_count,
        "raw_agreement": rating_report.raw_agreement,
        "cohen_kappa": kappa,
        "kappa_status": kappa_status,
        "category_counts": category_counts,
        "confusion_matrix": matrix,
        "disagreements": disagreements,
        "adjudication_count": len(adjudicated),
        "adjudicated_case_ids": adjudicated,
        "interpretation_limits": (
            "agreement_does_not_establish_label_correctness",
            "two_raters_are_not_a_population_sample",
            "sparse_categories_limit_kappa_interpretation",
            "ratings_cover_public_answers_not_criterion_performance",
        ),
        "completed_at_utc": created_at_utc,
    }
    draft = RatingReliabilityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = RatingReliabilityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    if (
        report.cohen_kappa != rating_report.cohen_kappa
        or report.kappa_status != rating_report.kappa_status
    ):
        raise RatingReliabilityError("Reconstructed kappa differs from the frozen rating report")
    write_immutable_json(report_path, report)
    return report


def _create_plan(
    *,
    rating_report: PublicAnswerRatingReport,
    procedure: PublicRatingProcedureRecord,
    rater_a_file_hash: Sha256,
    rater_b_file_hash: Sha256,
    lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> RatingReliabilityPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.rating_reliability_plan.v1",
        "public_rating_report_hash": rating_report.report_hash,
        "public_rating_procedure_hash": procedure.record_hash,
        "rater_a_file_hash": rater_a_file_hash,
        "rater_b_file_hash": rater_b_file_hash,
        "method": "raw_agreement_unweighted_kappa_full_confusion_v1",
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = RatingReliabilityPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return RatingReliabilityPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _index_ratings(
    ratings: tuple[PublicAnswerRating, ...],
    report: PublicAnswerRatingReport,
    *,
    expected_rater: str,
) -> dict[str, PublicAnswerRating]:
    indexed: dict[str, PublicAnswerRating] = {}
    for rating in ratings:
        if rating.rater_id != expected_rater:
            raise RatingReliabilityError("Rating file contains an unexpected rater ID")
        if (
            rating.packet_hash != report.packet_hash
            or rating.rating_guide_hash != report.rating_guide_hash
        ):
            raise RatingReliabilityError("Rating belongs to another packet or guide")
        if rating.case_id in indexed:
            raise RatingReliabilityError("Rating file contains a duplicate case")
        indexed[rating.case_id] = rating
    if len(indexed) != report.item_count:
        raise RatingReliabilityError("Rating file does not contain all reviewed cases")
    return indexed


def _validate_report_sources(
    report: PublicAnswerRatingReport,
    pairs: tuple[tuple[PublicAnswerRating, PublicAnswerRating], ...],
) -> None:
    final = {record.case_id: record for record in report.final_ratings}
    if set(final) != {left.case_id for left, _ in pairs}:
        raise RatingReliabilityError("Rating report and source files contain different cases")
    for left, right in pairs:
        if left.case_id != right.case_id or left.response_hash != right.response_hash:
            raise RatingReliabilityError("Rater files do not describe the same case response")
        resolved = final[left.case_id]
        if resolved.response_hash != left.response_hash or not {
            left.rating_hash,
            right.rating_hash,
        }.issubset(set(resolved.source_hashes)):
            raise RatingReliabilityError("Final rating does not bind both source ratings")


def _cohen_kappa(
    pairs: tuple[tuple[PublicAnswerRating, PublicAnswerRating], ...],
) -> tuple[float | None, Literal["defined", "undefined_degenerate_marginals"]]:
    count = len(pairs)
    observed = sum(left.category is right.category for left, right in pairs) / count
    left_counts = Counter(left.category for left, _ in pairs)
    right_counts = Counter(right.category for _, right in pairs)
    expected = sum(
        (left_counts[category] / count) * (right_counts[category] / count)
        for category in _CATEGORIES
    )
    if abs(1.0 - expected) < 1e-12:
        return None, "undefined_degenerate_marginals"
    return (observed - expected) / (1.0 - expected), "defined"


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise RatingReliabilityError(f"Could not verify rating source: {path}") from error


def _load_jsonl(path: Path) -> tuple[PublicAnswerRating, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RatingReliabilityError(f"Could not read rating file: {path}") from error
    ratings: list[PublicAnswerRating] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            ratings.append(PublicAnswerRating.model_validate_json(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise RatingReliabilityError(f"Invalid rating at {path}:{line_number}") from error
    return tuple(ratings)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Rating-reliability timestamp must be UTC")
