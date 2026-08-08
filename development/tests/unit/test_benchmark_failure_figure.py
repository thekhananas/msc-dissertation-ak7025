from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.failure_figure import FailureFigureError, render_failure_figure
from socratic_tutor.benchmark.failure_review import (
    FailureReviewLabel,
    FailureReviewRecord,
    FailureReviewRecordReport,
)
from socratic_tutor.benchmark.failure_review_reliability import (
    run_failure_review_reliability,
)
from socratic_tutor.benchmark.hashing import model_content_hash

_STAMP = datetime(2026, 9, 3, 18, 0, tzinfo=UTC)
_PACKET_HASH = "1" * 64
_TAXONOMY_HASH = "2" * 64


def test_renders_selected_failures_and_retries_exactly(tmp_path: Path) -> None:
    primary, secondary, plan, report, pixi_lock = _write_sources(tmp_path)
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 19, 0, tzinfo=UTC)
    first = render_failure_figure(
        primary_review_path=primary,
        secondary_review_path=secondary,
        reliability_plan_path=plan,
        reliability_report_path=report,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
        generated_at_utc=generated_at,
    )
    retry = render_failure_figure(
        primary_review_path=primary,
        secondary_review_path=secondary,
        reliability_plan_path=plan,
        reliability_report_path=report,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
    )

    assert first == retry
    assert first.reviewed_item_count == 6
    assert not first.agreement_establishes_correctness
    assert not first.hidden_reasoning_used
    assert (output_root / "failure_analysis.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "failure_analysis.svg").read_text(encoding="utf-8")
    assert "Both reviewers classified all three apparent regressions" in svg
    assert "six deliberately selected records" in svg
    assert "does not prove" in svg
    csv = (output_root / "failure_review_rows.csv").read_text(encoding="utf-8")
    assert len(csv.splitlines()) == 7
    assert "case-001,primary_failure" in csv


def test_rejects_a_changed_review_file(tmp_path: Path) -> None:
    primary, secondary, plan, report, pixi_lock = _write_sources(tmp_path)
    primary.write_text(primary.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(FailureFigureError, match="does not match"):
        render_failure_figure(
            primary_review_path=primary,
            secondary_review_path=secondary,
            reliability_plan_path=plan,
            reliability_report_path=report,
            pixi_lock_path=pixi_lock,
            output_root=tmp_path / "publication",
            publication_code_revision="a" * 40,
        )


def _write_sources(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    categories = (
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
        FailureReviewLabel.NO_FAILURE_OBSERVED,
    )
    primary = _write_review(root / "primary.json", "researcher", categories)
    secondary = _write_review(root / "secondary.json", "independent", categories)
    pixi_lock = root / "pixi.lock"
    pixi_lock.write_text("locked\n", encoding="utf-8")
    reliability_root = root / "reliability"
    run_failure_review_reliability(
        primary_review_path=primary,
        secondary_review_path=secondary,
        pixi_lock_path=pixi_lock,
        output_root=reliability_root,
        analysis_code_revision="failure-figure-test",
        created_at_utc=_STAMP,
    )
    return (
        primary,
        secondary,
        reliability_root / "failure_review_reliability_plan.json",
        reliability_root / "failure_review_reliability_report.json",
        pixi_lock,
    )


def _write_review(
    path: Path,
    rater_id: str,
    categories: tuple[FailureReviewLabel, ...],
) -> Path:
    records = tuple(
        _record(
            case_id=f"case-{index:03d}",
            rater_id=rater_id,
            category=category,
            selection_role="primary_failure" if index <= 3 else "matched_success",
        )
        for index, category in enumerate(categories, start=1)
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_record_report.v1",
        "packet_hash": _PACKET_HASH,
        "taxonomy_hash": _TAXONOMY_HASH,
        "rater_id": rater_id,
        "record_count": 6,
        "primary_failure_count": 3,
        "matched_success_count": 3,
        "category_counts": {
            category: sum(record.category is category for record in records)
            for category in FailureReviewLabel
        },
        "records": records,
        "completed_at_utc": _STAMP,
    }
    draft = FailureReviewRecordReport.model_construct(
        _fields_set=set(content),
        **content,
        report_hash="0" * 64,
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
    selection_role: Literal["primary_failure", "matched_success"],
) -> FailureReviewRecord:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_review_record.v1",
        "packet_hash": _PACKET_HASH,
        "taxonomy_hash": _TAXONOMY_HASH,
        "case_id": case_id,
        "selection_role": selection_role,
        "matched_case_id": f"matched-{case_id}",
        "rater_id": rater_id,
        "category": category,
        "rationale": "Visible evidence supports this category.",
        "evidence_references": tuple(f"{index:064x}" for index in range(10, 15)),
        "recorded_at_utc": _STAMP,
    }
    draft = FailureReviewRecord.model_construct(
        _fields_set=set(content),
        **content,
        record_hash="0" * 64,
    )
    return FailureReviewRecord.model_validate(
        {**content, "record_hash": model_content_hash(draft, exclude={"record_hash"})}
    )
