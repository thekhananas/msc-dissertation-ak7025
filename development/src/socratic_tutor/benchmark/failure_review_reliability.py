"""Reproducible agreement audit for two failure-review reports."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.failure_review import (
    FailureReviewLabel,
    FailureReviewRecord,
    FailureReviewRecordReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_CATEGORIES = tuple(FailureReviewLabel)


class FailureReviewReliabilityError(ValueError):
    """Review reports cannot support the same-case agreement audit."""


class FailureReviewCategoryCount(ContractModel):
    """One category's marginal count for both reviewers."""

    category: FailureReviewLabel
    primary_count: int = Field(ge=0)
    secondary_count: int = Field(ge=0)


class FailureReviewConfusionCell(ContractModel):
    """One cell in the complete primary-by-secondary confusion matrix."""

    primary_category: FailureReviewLabel
    secondary_category: FailureReviewLabel
    count: int = Field(ge=0)


class FailureReviewDisagreement(ContractModel):
    """One case on which the reviewers selected different categories."""

    case_id: str = Field(min_length=1)
    selection_role: Literal["primary_failure", "matched_success"]
    primary_category: FailureReviewLabel
    secondary_category: FailureReviewLabel


class FailureReviewReliabilityPlan(ContractModel):
    """Immutable inputs and software identity for the agreement audit."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_review_reliability_plan.v1"] = (
        "benchmark.failure_review_reliability_plan.v1"
    )
    packet_hash: Sha256
    taxonomy_hash: Sha256
    primary_review_report_hash: Sha256
    secondary_review_report_hash: Sha256
    primary_review_file_hash: Sha256
    secondary_review_file_hash: Sha256
    method: Literal["raw_agreement_unweighted_kappa_full_confusion_v1"] = (
        "raw_agreement_unweighted_kappa_full_confusion_v1"
    )
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> FailureReviewReliabilityPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Failure-review reliability plan hash does not match its content")
        return self


class FailureReviewReliabilityReport(ContractModel):
    """Agreement result, full confusion matrix, and interpretation limits."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_review_reliability_report.v1"] = (
        "benchmark.failure_review_reliability_report.v1"
    )
    analysis_plan_hash: Sha256
    rater_ids: tuple[str, str]
    item_count: Literal[6]
    agreement_count: int = Field(ge=0)
    disagreement_count: int = Field(ge=0)
    raw_agreement: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    kappa_status: Literal["defined", "undefined_degenerate_marginals"]
    category_counts: tuple[FailureReviewCategoryCount, ...] = Field(min_length=9, max_length=9)
    confusion_matrix: tuple[FailureReviewConfusionCell, ...] = Field(min_length=81, max_length=81)
    disagreements: tuple[FailureReviewDisagreement, ...]
    adjudication_status: Literal["not_required_no_disagreements", "required_not_provided"]
    interpretation_limits: tuple[
        Literal["agreement_does_not_establish_category_correctness"],
        Literal["six_records_limit_reliability_inference"],
        Literal["shared_taxonomy_can_produce_shared_interpretation"],
        Literal["final_reports_do_not_encode_review_revision_history"],
    ]
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> FailureReviewReliabilityReport:
        _require_utc(self.completed_at_utc)
        if self.item_count != self.agreement_count + self.disagreement_count:
            raise ValueError("Failure-review agreement counts do not reconcile")
        if abs(self.raw_agreement - self.agreement_count / self.item_count) > 1e-12:
            raise ValueError("Failure-review raw agreement does not match its counts")
        if (self.cohen_kappa is None) is not (
            self.kappa_status == "undefined_degenerate_marginals"
        ):
            raise ValueError("Failure-review kappa value and status are inconsistent")
        if tuple(record.category for record in self.category_counts) != _CATEGORIES:
            raise ValueError("Failure-review marginals are incomplete or out of order")
        expected_cells = tuple((left, right) for left in _CATEGORIES for right in _CATEGORIES)
        actual_cells = tuple(
            (record.primary_category, record.secondary_category) for record in self.confusion_matrix
        )
        if actual_cells != expected_cells:
            raise ValueError("Failure-review confusion matrix is incomplete or out of order")
        if sum(record.count for record in self.confusion_matrix) != self.item_count:
            raise ValueError("Failure-review confusion matrix does not sum to the item count")
        if any(
            sum(getattr(record, field) for record in self.category_counts) != self.item_count
            for field in ("primary_count", "secondary_count")
        ):
            raise ValueError("Failure-review category marginals do not sum to the item count")
        counts = {record.category: record for record in self.category_counts}
        for category in _CATEGORIES:
            row_total = sum(
                record.count
                for record in self.confusion_matrix
                if record.primary_category is category
            )
            column_total = sum(
                record.count
                for record in self.confusion_matrix
                if record.secondary_category is category
            )
            if row_total != counts[category].primary_count:
                raise ValueError("Failure-review confusion row differs from its marginal")
            if column_total != counts[category].secondary_count:
                raise ValueError("Failure-review confusion column differs from its marginal")
        diagonal = sum(
            record.count
            for record in self.confusion_matrix
            if record.primary_category is record.secondary_category
        )
        if diagonal != self.agreement_count:
            raise ValueError("Failure-review confusion diagonal differs from agreement count")
        if len(self.disagreements) != self.disagreement_count:
            raise ValueError("Failure-review disagreement records do not reconcile")
        expected_status = (
            "not_required_no_disagreements"
            if self.disagreement_count == 0
            else "required_not_provided"
        )
        if self.adjudication_status != expected_status:
            raise ValueError("Failure-review adjudication status is inconsistent")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Failure-review reliability report hash does not match its content")
        return self


def run_failure_review_reliability(
    *,
    primary_review_path: Path,
    secondary_review_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> FailureReviewReliabilityReport:
    """Compare two complete reviews of the same failure-review packet."""

    _require_utc(created_at_utc)
    primary = _load_report(primary_review_path)
    secondary = _load_report(secondary_review_path)
    _validate_shared_provenance(primary, secondary)
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
        primary_file_hash = file_sha256(primary_review_path.read_bytes())
        secondary_file_hash = file_sha256(secondary_review_path.read_bytes())
    except OSError as error:
        raise FailureReviewReliabilityError(
            "Could not hash failure-review reliability input"
        ) from error

    plan = _create_plan(
        primary=primary,
        secondary=secondary,
        primary_file_hash=primary_file_hash,
        secondary_file_hash=secondary_file_hash,
        lock_hash=lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "failure_review_reliability_report.json"
    if report_path.exists():
        report = _load_json(report_path, FailureReviewReliabilityReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise FailureReviewReliabilityError(
                "Existing failure-review reliability report belongs to another plan"
            )
        return report
    write_immutable_json(root / "failure_review_reliability_plan.json", plan)

    first = {record.case_id: record for record in primary.records}
    second = {record.case_id: record for record in secondary.records}
    pairs = tuple((first[case_id], second[case_id]) for case_id in sorted(first))
    primary_counts = Counter(left.category for left, _ in pairs)
    secondary_counts = Counter(right.category for _, right in pairs)
    confusion = Counter((left.category, right.category) for left, right in pairs)
    disagreements = tuple(
        FailureReviewDisagreement(
            case_id=left.case_id,
            selection_role=left.selection_role,
            primary_category=left.category,
            secondary_category=right.category,
        )
        for left, right in pairs
        if left.category is not right.category
    )
    category_counts = tuple(
        FailureReviewCategoryCount(
            category=category,
            primary_count=primary_counts[category],
            secondary_count=secondary_counts[category],
        )
        for category in _CATEGORIES
    )
    matrix = tuple(
        FailureReviewConfusionCell(
            primary_category=left,
            secondary_category=right,
            count=confusion[(left, right)],
        )
        for left in _CATEGORIES
        for right in _CATEGORIES
    )
    agreement_count = len(pairs) - len(disagreements)
    kappa, kappa_status = _cohen_kappa(pairs)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_reliability_report.v1",
        "analysis_plan_hash": plan.plan_hash,
        "rater_ids": (primary.rater_id, secondary.rater_id),
        "item_count": len(pairs),
        "agreement_count": agreement_count,
        "disagreement_count": len(disagreements),
        "raw_agreement": agreement_count / len(pairs),
        "cohen_kappa": kappa,
        "kappa_status": kappa_status,
        "category_counts": category_counts,
        "confusion_matrix": matrix,
        "disagreements": disagreements,
        "adjudication_status": (
            "not_required_no_disagreements" if not disagreements else "required_not_provided"
        ),
        "interpretation_limits": (
            "agreement_does_not_establish_category_correctness",
            "six_records_limit_reliability_inference",
            "shared_taxonomy_can_produce_shared_interpretation",
            "final_reports_do_not_encode_review_revision_history",
        ),
        "completed_at_utc": created_at_utc,
    }
    draft = FailureReviewReliabilityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = FailureReviewReliabilityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _validate_shared_provenance(
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
) -> None:
    if primary.rater_id == secondary.rater_id:
        raise FailureReviewReliabilityError("Failure reviews must use different rater IDs")
    if (
        primary.packet_hash != secondary.packet_hash
        or primary.taxonomy_hash != secondary.taxonomy_hash
    ):
        raise FailureReviewReliabilityError("Failure reviews use different packet provenance")
    if primary.record_count != secondary.record_count:
        raise FailureReviewReliabilityError("Failure reviews contain different record counts")
    first = {record.case_id: record for record in primary.records}
    second = {record.case_id: record for record in secondary.records}
    if set(first) != set(second):
        raise FailureReviewReliabilityError("Failure reviews contain different cases")
    for case_id, left in first.items():
        right = second[case_id]
        if (
            left.selection_role != right.selection_role
            or left.matched_case_id != right.matched_case_id
            or left.evidence_references != right.evidence_references
        ):
            raise FailureReviewReliabilityError(
                f"Failure reviews describe different evidence for {case_id}"
            )


def _create_plan(
    *,
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
    primary_file_hash: Sha256,
    secondary_file_hash: Sha256,
    lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> FailureReviewReliabilityPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_reliability_plan.v1",
        "packet_hash": primary.packet_hash,
        "taxonomy_hash": primary.taxonomy_hash,
        "primary_review_report_hash": primary.report_hash,
        "secondary_review_report_hash": secondary.report_hash,
        "primary_review_file_hash": primary_file_hash,
        "secondary_review_file_hash": secondary_file_hash,
        "method": "raw_agreement_unweighted_kappa_full_confusion_v1",
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = FailureReviewReliabilityPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return FailureReviewReliabilityPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _cohen_kappa(
    pairs: tuple[tuple[FailureReviewRecord, FailureReviewRecord], ...],
) -> tuple[float | None, Literal["defined", "undefined_degenerate_marginals"]]:
    count = len(pairs)
    observed = sum(left.category is right.category for left, right in pairs) / count
    primary_counts = Counter(left.category for left, _ in pairs)
    secondary_counts = Counter(right.category for _, right in pairs)
    expected = sum(
        (primary_counts[category] / count) * (secondary_counts[category] / count)
        for category in _CATEGORIES
    )
    if abs(1.0 - expected) < 1e-12:
        return None, "undefined_degenerate_marginals"
    return (observed - expected) / (1.0 - expected), "defined"


def _load_report(path: Path) -> FailureReviewRecordReport:
    return _load_json(path, FailureReviewRecordReport)


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise FailureReviewReliabilityError(
            f"Could not verify failure-review reliability source: {path}"
        ) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Failure-review reliability timestamp must be UTC")
