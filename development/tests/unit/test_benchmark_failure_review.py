# pyright: reportPrivateUsage=false
"""Tests for deterministic failure-to-success matching."""

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.failure_review import (
    FailureReviewItem,
    FailureReviewLabel,
    ObservableResponse,
    _make_packet,
    _match_successes,
    _review_sheet,
    record_failure_review,
)
from socratic_tutor.benchmark.primary_analysis import PrimaryCaseResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
TAXONOMY = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-failure-taxonomy.yaml"


def test_matches_failures_without_reusing_successes() -> None:
    manifest = load_and_verify_manifest(MANIFEST)
    metadata = {case.public.case_id: case for case in manifest.cases}
    failures = [
        _result("h-c1m1-03", correct=False),
        _result("h-c1m2-01", correct=False),
        _result("h-c1m2-02", correct=False),
    ]
    successes = [
        _result("h-c2m2-01", correct=True),
        _result("h-c2m1-02", correct=True),
        _result("h-c3m1-01", correct=True),
        _result("h-c4m1-01", correct=True),
    ]

    matches = _match_successes(failures, tuple((*failures, *successes)), metadata)

    assert [(failure.case_id, success.case_id) for failure, success in matches] == [
        ("h-c1m1-03", "h-c2m2-01"),
        ("h-c1m2-01", "h-c2m1-02"),
        ("h-c1m2-02", "h-c3m1-01"),
    ]
    assert len({success.case_id for _, success in matches}) == len(failures)


def test_records_complete_review_without_allowing_evidence_edits(tmp_path: Path) -> None:
    packet = _make_packet(
        "run-1",
        "0" * 64,
        (
            _item("failure-1", "primary_failure", "success-1", correct=False),
            _item("success-1", "matched_success", "failure-1", correct=True),
        ),
    )
    packet_path = tmp_path / "packet.json"
    sheet_path = tmp_path / "completed.csv"
    output_path = tmp_path / "record.json"
    write_immutable_json(packet_path, packet)
    write_immutable_bytes(sheet_path, _review_sheet(packet))
    with sheet_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = tuple(rows[0])
    rows[0]["category"] = "tracker_update_or_calibration_error"
    rows[0]["rationale"] = "The probe score falls below the frozen threshold."
    rows[0]["criterion_outcome"] = "TRUE"
    rows[0]["probe_score"] = "0.5"
    rows[0]["dialogue_decision"] = "TRUE"
    rows[0]["probe_decision"] = "FALSE"
    rows[1]["category"] = "no_failure_observed"
    rows[1]["rationale"] = "The prediction agrees with the completed criterion tests."
    rows[1]["criterion_outcome"] = "TRUE"
    rows[1]["dialogue_decision"] = "TRUE"
    rows[1]["probe_decision"] = "TRUE"
    with sheet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = record_failure_review(
        taxonomy_path=TAXONOMY,
        packet_path=packet_path,
        completed_sheet_path=sheet_path,
        rater_id="reviewer-a",
        output_path=output_path,
        recorded_at_utc=datetime(2026, 9, 1, 18, 0, tzinfo=UTC),
    )

    assert report.record_count == 2
    assert report.primary_failure_count == 1
    assert report.matched_success_count == 1
    assert report.category_counts[FailureReviewLabel.TRACKER_UPDATE_OR_CALIBRATION_ERROR] == 1
    assert report.category_counts[FailureReviewLabel.NO_FAILURE_OBSERVED] == 1
    assert output_path.exists()
    assert (
        record_failure_review(
            taxonomy_path=TAXONOMY,
            packet_path=packet_path,
            completed_sheet_path=sheet_path,
            rater_id="reviewer-a",
            output_path=output_path,
            recorded_at_utc=datetime(2026, 9, 1, 18, 0, tzinfo=UTC),
        )
        == report
    )


def _result(case_id: str, *, correct: bool) -> PrimaryCaseResult:
    decision = correct
    return PrimaryCaseResult(
        case_id=case_id,
        criterion_outcome=True,
        dialogue_binary_decision=True,
        valid_evidence_binary_decision=decision,
        dialogue_correct=True,
        valid_evidence_correct=correct,
        dialogue_classification_error=0.0,
        valid_evidence_classification_error=float(not correct),
        case_effect=-float(not correct),
        aggregate_record_hash="1" * 64,
        dialogue_repeat_hash="2" * 64,
        valid_evidence_repeat_hash="3" * 64,
        criterion_record_hash="4" * 64,
    )


def _item(
    case_id: str,
    role: Literal["primary_failure", "matched_success"],
    matched_case_id: str,
    *,
    correct: bool,
) -> FailureReviewItem:
    exchange = ObservableResponse(
        channel="public",
        prompt="Visible task",
        response="Visible response",
        request_hash="5" * 64,
        response_hash="6" * 64,
    )
    return FailureReviewItem(
        case_id=case_id,
        selection_role=role,
        matched_case_id=matched_case_id,
        target_concept="concept",
        misconception_id="misconception",
        evidence_pattern="pattern",
        criterion_outcome=True,
        criterion_execution_status="completed",
        criterion_tests_passed=1,
        criterion_tests_failed=0,
        dialogue_score=0.7,
        probe_score=0.7 if correct else 0.5,
        policy_threshold=0.6,
        dialogue_decision=True,
        probe_decision=correct,
        dialogue_correct=True,
        probe_correct=correct,
        public_exchange=exchange,
        evidence_exchange=exchange.model_copy(update={"channel": "evidence"}),
        criterion_exchange=exchange.model_copy(update={"channel": "criterion"}),
        case_content_hash="7" * 64,
        criterion_record_hash="8" * 64,
        dialogue_repeat_hash="9" * 64,
        probe_repeat_hash="a" * 64,
        aggregate_record_hash="b" * 64,
    )
