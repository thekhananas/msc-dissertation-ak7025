# pyright: reportPrivateUsage=false
"""Tests for the corrected case-level benchmark figure."""

from datetime import UTC, datetime

from socratic_tutor.benchmark.harness_correction_analysis import (
    CorrectedConditionScore,
    HarnessCorrectionAnalysisReport,
    HarnessCorrectionCaseResult,
)
from socratic_tutor.benchmark.harness_correction_case_figure import (
    _case_table_csv,
    _condition_correct,
    _render_vector_files,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    HarnessCorrectionClosureReport,
)
from socratic_tutor.benchmark.public.models import BenchmarkCondition


def test_case_figure_keeps_each_condition_and_missing_case_visible() -> None:
    cases = (
        _case("h-c1m1-01", (False, True, True, False), 1.0),
        _case("h-c1m1-02", (True, True, True, False), 0.0),
        _case("h-c2m2-03", (None, None, None, None), None),
    )
    analysis = HarnessCorrectionAnalysisReport.model_construct(case_results=cases)
    closure = HarnessCorrectionClosureReport.model_construct(
        total_case_count=24,
        eligible_case_count=23,
        missing_case_ids=("h-c2m2-03",),
    )

    pdf, svg_bytes = _render_vector_files(
        analysis,
        closure,
        datetime(2026, 9, 3, 19, 30, tzinfo=UTC),
    )
    svg = svg_bytes.decode("utf-8")
    table = _case_table_csv(analysis).decode("utf-8")

    assert pdf.startswith(b"%PDF")
    assert "What changed in each benchmark case?" in svg
    assert "Only one of 23 eligible cases" in svg
    assert "Relevant and unrelated evidence led to the same decision" in svg
    assert "h-c2m2-03" in svg
    assert "h-c1m1-01,eligible,wrong,right,right,wrong,1.0" in table
    assert "h-c2m2-03,missing_criterion,not_scored" in table


def test_condition_lookup_does_not_depend_on_tuple_position() -> None:
    case = _case("h-c1m1-01", (False, True, True, False), 1.0)

    assert _condition_correct(case, BenchmarkCondition.PROBE_INFORMED)
    assert not _condition_correct(case, BenchmarkCondition.DIALOGUE_ONLY)


def _case(
    case_id: str,
    correctness: tuple[bool | None, bool | None, bool | None, bool | None],
    effect: float | None,
) -> HarnessCorrectionCaseResult:
    scores = tuple(
        CorrectedConditionScore.model_construct(condition=condition, correct=correct)
        for condition, correct in zip(BenchmarkCondition, correctness, strict=True)
    )
    return HarnessCorrectionCaseResult.model_construct(
        case_id=case_id,
        conditions=scores,
        primary_effect=effect,
    )
