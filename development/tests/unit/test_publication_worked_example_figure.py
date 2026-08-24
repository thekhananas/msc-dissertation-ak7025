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


def test_figure_states_the_claim_boundary_without_a_visible_timestamp() -> None:
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
    assert "Outcome still hidden" in svg
    assert "Unrelated evidence" in svg
    assert "relevance was not established." in svg
    assert "does not represent a human learner" in svg
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
            mastery_probability=probability,
            policy_threshold=0.5,
            predicts_success=decision,
            correct=decision,
            input_hash=_sha(index + 1),
            record_hash=_sha(index + 5),
            committed_at_utc=committed_at,
        )
        for index, (condition, probability, decision) in enumerate(
            zip(EXPECTED_CONDITIONS, probabilities, decisions, strict=True)
        )
    )
    content = {
        "schema_version": 1,
        "schema_id": "publication.worked_example_data.v1",
        "figure_id": "FIG-08",
        "benchmark_version": "v1",
        "run_id": "worked-example-test",
        "case_id": "h-c2m1-02",
        "sample_id": "001",
        "evaluation_model": "test-model",
        "selection_reason": "Only corrected case with a positive primary effect.",
        "public_summary": 'The response opens with "True 3", then concludes "False 2".',
        "public_rating_category": "conflicting",
        "public_response_hash": _sha(9),
        "evidence_summary": "The executable answer passed its authored check.",
        "evidence_response_hash": _sha(10),
        "evidence_passed": 1,
        "evidence_failed": 0,
        "predictions": predictions,
        "criterion_response_hash": _sha(11),
        "criterion_passed": 2,
        "criterion_failed": 0,
        "criterion_demonstrated_performance": True,
        "criterion_revealed_at_utc": criterion_revealed_at,
        "interpretation": (
            "Relevant evidence moved this case from a wrong to a correct prediction. "
            "Unrelated evidence caused the same change, so relevance was not established."
        ),
        "claim_boundary": (
            "This does not represent a human learner or show that tutoring improved learning."
        ),
        "analysis_case_result_hash": _sha(12),
        "evidence_attempt_hash": _sha(13),
        "criterion_attempt_hash": _sha(14),
        "predictions_sealed_before_outcome_reveal": True,
        "relevant_and_unrelated_predictions_match": True,
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
