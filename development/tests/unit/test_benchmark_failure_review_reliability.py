from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.failure_review import (
    FailureReviewLabel,
    FailureReviewRecord,
    FailureReviewRecordReport,
)
from socratic_tutor.benchmark.failure_review_reliability import (
    FailureReviewReliabilityError,
    run_failure_review_reliability,
)
from socratic_tutor.benchmark.hashing import model_content_hash

ROOT = Path(__file__).resolve().parents[2]
PIXI_LOCK = ROOT / "pixi.lock"
STAMP = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)
PACKET_HASH = "1" * 64
TAXONOMY_HASH = "2" * 64


def test_reports_complete_agreement_and_full_confusion_matrix(tmp_path: Path) -> None:
    categories = (
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
    )
    primary_path = _write_report(tmp_path / "primary.json", "primary", categories)
    secondary_path = _write_report(tmp_path / "secondary.json", "secondary", categories)

    report = run_failure_review_reliability(
        primary_review_path=primary_path,
        secondary_review_path=secondary_path,
        pixi_lock_path=PIXI_LOCK,
        output_root=tmp_path / "reliability",
        analysis_code_revision="failure-reliability-unit",
        created_at_utc=STAMP,
    )

    assert report.item_count == 6
    assert report.agreement_count == 6
    assert report.disagreement_count == 0
    assert report.raw_agreement == 1.0
    assert report.cohen_kappa == 1.0
    assert report.kappa_status == "defined"
    assert len(report.category_counts) == 9
    assert len(report.confusion_matrix) == 81
    assert sum(cell.count for cell in report.confusion_matrix) == 6
    assert report.adjudication_status == "not_required_no_disagreements"
    assert (
        run_failure_review_reliability(
            primary_review_path=primary_path,
            secondary_review_path=secondary_path,
            pixi_lock_path=PIXI_LOCK,
            output_root=tmp_path / "reliability",
            analysis_code_revision="failure-reliability-unit",
            created_at_utc=STAMP,
        )
        == report
    )


def test_preserves_disagreement_and_requires_adjudication(tmp_path: Path) -> None:
    primary_categories = (
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
    )
    secondary_categories = (
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.SYNTHETIC_STUDENT_INCONSISTENCY,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
    )
    primary_path = _write_report(tmp_path / "primary.json", "primary", primary_categories)
    secondary_path = _write_report(tmp_path / "secondary.json", "secondary", secondary_categories)

    report = run_failure_review_reliability(
        primary_review_path=primary_path,
        secondary_review_path=secondary_path,
        pixi_lock_path=PIXI_LOCK,
        output_root=tmp_path / "reliability",
        analysis_code_revision="failure-reliability-unit",
        created_at_utc=STAMP,
    )

    assert report.agreement_count == 5
    assert report.disagreement_count == 1
    assert report.disagreements[0].case_id == "case-003"
    assert report.adjudication_status == "required_not_provided"


def test_rejects_reports_from_different_packets(tmp_path: Path) -> None:
    categories = (FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,) * 4 + (
        FailureReviewLabel.NO_FAILURE_OBSERVED,
    ) * 2
    primary_path = _write_report(tmp_path / "primary.json", "primary", categories)
    secondary_path = _write_report(
        tmp_path / "secondary.json",
        "secondary",
        categories,
        packet_hash="3" * 64,
    )

    with pytest.raises(
        FailureReviewReliabilityError,
        match="different packet provenance",
    ):
        run_failure_review_reliability(
            primary_review_path=primary_path,
            secondary_review_path=secondary_path,
            pixi_lock_path=PIXI_LOCK,
            output_root=tmp_path / "reliability",
            analysis_code_revision="failure-reliability-unit",
            created_at_utc=STAMP,
        )


def test_rejects_same_rater_on_both_reports(tmp_path: Path) -> None:
    categories = (FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,) * 4 + (
        FailureReviewLabel.NO_FAILURE_OBSERVED,
    ) * 2
    primary_path = _write_report(tmp_path / "primary.json", "same-rater", categories)
    secondary_path = _write_report(tmp_path / "secondary.json", "same-rater", categories)

    with pytest.raises(FailureReviewReliabilityError, match="different rater IDs"):
        run_failure_review_reliability(
            primary_review_path=primary_path,
            secondary_review_path=secondary_path,
            pixi_lock_path=PIXI_LOCK,
            output_root=tmp_path / "reliability",
            analysis_code_revision="failure-reliability-unit",
            created_at_utc=STAMP,
        )


def _write_report(
    path: Path,
    rater_id: str,
    categories: tuple[FailureReviewLabel, ...],
    *,
    packet_hash: str = PACKET_HASH,
) -> Path:
    records = tuple(
        _record(
            case_id=f"case-{index:03d}",
            rater_id=rater_id,
            category=category,
            packet_hash=packet_hash,
            selection_role="primary_failure" if index <= 4 else "matched_success",
        )
        for index, category in enumerate(categories, start=1)
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_record_report.v1",
        "packet_hash": packet_hash,
        "taxonomy_hash": TAXONOMY_HASH,
        "rater_id": rater_id,
        "record_count": 6,
        "primary_failure_count": 4,
        "matched_success_count": 2,
        "category_counts": {
            category: sum(record.category is category for record in records)
            for category in FailureReviewLabel
        },
        "records": records,
        "completed_at_utc": STAMP,
    }
    draft = FailureReviewRecordReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = FailureReviewRecordReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(path, report)
    return path


def _record(
    *,
    case_id: str,
    rater_id: str,
    category: FailureReviewLabel,
    packet_hash: str,
    selection_role: str,
) -> FailureReviewRecord:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_record.v1",
        "packet_hash": packet_hash,
        "taxonomy_hash": TAXONOMY_HASH,
        "case_id": case_id,
        "selection_role": selection_role,
        "matched_case_id": f"matched-{case_id}",
        "rater_id": rater_id,
        "category": category,
        "rationale": "Visible review evidence supports this category.",
        "evidence_references": tuple(str(index) * 64 for index in range(3, 8)),
        "recorded_at_utc": STAMP,
    }
    draft = FailureReviewRecord.model_construct(
        _fields_set=set(content), **content, record_hash="0" * 64
    )
    return FailureReviewRecord.model_validate(
        {**content, "record_hash": model_content_hash(draft, exclude={"record_hash"})}
    )
