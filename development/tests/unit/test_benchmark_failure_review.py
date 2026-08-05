# pyright: reportPrivateUsage=false
"""Tests for deterministic failure-to-success matching."""

from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.failure_review import _match_successes
from socratic_tutor.benchmark.primary_analysis import PrimaryCaseResult

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"


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
