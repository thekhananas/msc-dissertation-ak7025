# pyright: reportPrivateUsage=false
"""Tests for the artifact-backed benchmark worked example."""

from datetime import UTC, datetime, timedelta

import pytest

from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public.models import EXPECTED_CONDITIONS
from socratic_tutor.publication.worked_example_figure import (
    _CONDITION_LABELS,
    WorkedExampleData,
    WorkedExamplePrediction,
    _render_vector_files,
)


def test_rejects_predictions_that_do_not_precede_outcome_reveal() -> None:
    committed_at = datetime(2026, 9, 1, 7, 47, tzinfo=UTC)

    with pytest.raises(ValueError, match="must precede criterion reveal"):
        _worked_example(
            committed_at=committed_at,
            criterion_revealed_at=committed_at - timedelta(minutes=1),
        )


def test_figure_exposes_the_scoring_and_test_limits_without_a_visible_timestamp() -> None:
    committed_at = datetime(2026, 9, 1, 7, 47, tzinfo=UTC)
    data = _worked_example(
        committed_at=committed_at,
        criterion_revealed_at=committed_at + timedelta(hours=1),
    )

    pdf, svg_bytes = _render_vector_files(
        data,
        profile="report",
        generated_at_utc=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )
    svg = svg_bytes.decode("utf-8")

    assert pdf.startswith(b"%PDF")
    assert "SAME-CONCEPT CHECK" in svg
    assert "DIFFERENT-CONCEPT CONTROL" in svg
    assert "Hand-set tracker scores, not calibrated probabilities" in svg
    assert "TEST LIMIT" in svg
    assert "returned values only." in svg
    assert "This case does not establish a benefit" in svg
    assert "60%" not in svg
    assert "Generated" not in svg


def _worked_example(
    *,
    committed_at: datetime,
    criterion_revealed_at: datetime,
) -> WorkedExampleData:
    decisions = (False, True, True, False)
    probabilities = (0.4, 0.6, 0.6, 0.2)
    predictions = tuple(
        WorkedExamplePrediction(
            condition=condition,
            label=_CONDITION_LABELS[condition],
            tracker_score=probability,
            policy_threshold=0.6,
            predicts_success=decision,
            matches_authored_outcome=decision,
            input_hash=_sha(index + 1),
            record_hash=_sha(index + 5),
            committed_at_utc=committed_at,
        )
        for index, (condition, probability, decision) in enumerate(
            zip(EXPECTED_CONDITIONS, probabilities, decisions, strict=True)
        )
    )
    content = {
        "schema_version": 2,
        "schema_id": "publication.worked_example_data.v2",
        "figure_id": "FIG-08",
        "benchmark_version": "v1",
        "run_id": "worked-example-test",
        "case_id": "h-c2m1-02",
        "sample_id": "001",
        "evaluation_model": "test-model",
        "selection_reason": "Post-hoc case used to explain a measurement weakness.",
        "public_summary": 'The response opens with "True 3", then concludes "False 2".',
        "public_rating_category": "conflicting",
        "public_response_hash": _sha(9),
        "same_concept_summary": "The same-concept answer passed its authored check.",
        "same_concept_response_hash": _sha(10),
        "same_concept_passed": 1,
        "same_concept_failed": 0,
        "different_concept_case_id": "h-c3m1-02",
        "different_concept_summary": (
            "The different-concept answer passed, although it also used set mutation."
        ),
        "different_concept_response_hash": _sha(15),
        "different_concept_passed": 1,
        "different_concept_failed": 0,
        "tracker_rule_summary": "A passed check added 0.20 to the fixed score.",
        "predictions": predictions,
        "criterion_response_hash": _sha(11),
        "criterion_passed": 2,
        "criterion_failed": 0,
        "criterion_summary": "The later answer passed 2 of 2 authored tests.",
        "criterion_test_warning": (
            "The prompt required mutation; the tests checked returned values only."
        ),
        "criterion_tests_checked_input_mutation": False,
        "criterion_revealed_at_utc": criterion_revealed_at,
        "failure_review_category": "item_ambiguity_or_test_defect",
        "reviewers_agreed_on_category": True,
        "primary_review_record_hash": _sha(16),
        "secondary_review_record_hash": _sha(17),
        "failure_review_reliability_report_hash": _sha(18),
        "interpretation": (
            "Both checks produced the same score. This case does not establish a benefit from "
            "relevant evidence."
        ),
        "claim_boundary": (
            "This does not represent a human learner; tracker scores are not probabilities."
        ),
        "analysis_case_result_hash": _sha(12),
        "evidence_attempt_hash": _sha(13),
        "different_concept_attempt_hash": _sha(19),
        "criterion_attempt_hash": _sha(14),
        "predictions_sealed_before_outcome_reveal": True,
        "same_and_different_concept_predictions_match": True,
        "tracker_scores_are_calibrated_probabilities": False,
        "corrected_result_is_post_hoc": True,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    unhashed = WorkedExampleData.model_construct(
        _fields_set=set(content),
        **content,
        data_hash="0" * 64,
    )
    return WorkedExampleData.model_validate(
        {**content, "data_hash": model_content_hash(unhashed, exclude={"data_hash"})}
    )


def _sha(value: int) -> str:
    return f"{value:064x}"
