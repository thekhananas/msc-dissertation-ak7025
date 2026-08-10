# pyright: reportPrivateUsage=false
"""Tests for the complete harness-correction publication figure."""

from datetime import UTC, datetime

import pytest

from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisPlan,
    HarnessCorrectionAnalysisReport,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    CorrectionClaim,
    HarnessCorrectionClosurePlan,
    HarnessCorrectionClosureReport,
)
from socratic_tutor.benchmark.harness_correction_figure import (
    HarnessCorrectionFigureError,
    _render_vector_files,
    _summary_csv,
    _validate_sources,
)
from socratic_tutor.benchmark.statistics import PairedCaseInference

HASH = "a" * 64
OTHER_HASH = "b" * 64


def test_figure_explains_the_material_correction_in_pdf_svg_and_csv() -> None:
    analysis, closure = _reports()

    pdf, svg_bytes = _render_vector_files(
        analysis,
        closure,
        datetime(2026, 9, 3, 18, 30, tzinfo=UTC),
    )
    svg = svg_bytes.decode("utf-8")
    summary = _summary_csv(analysis, closure).decode("utf-8")

    assert pdf.startswith(b"%PDF")
    assert "A test-harness fault changed the benchmark conclusion" in svg
    assert "Corrected: 1 net case" in svg
    assert "Relevant vs unrelated" in svg
    assert "This does not show human learning" in svg
    assert "dialogue_only,original,13,23,0.5652173913043478" in summary
    assert "probe_minus_dialogue,corrected,0.043478260869565216" in summary


def test_source_validation_rejects_a_closure_from_another_analysis() -> None:
    analysis, closure = _reports()
    analysis_plan = HarnessCorrectionAnalysisPlan.model_construct(
        plan_hash=HASH,
        pixi_lock_sha256=HASH,
    )
    closure_plan = HarnessCorrectionClosurePlan.model_construct(
        plan_hash=HASH,
        pixi_lock_sha256=HASH,
        correction_analysis_plan_hash=HASH,
        correction_analysis_report_hash=OTHER_HASH,
    )
    with pytest.raises(HarnessCorrectionFigureError, match="another correction analysis report"):
        _validate_sources(
            analysis_plan=analysis_plan,
            analysis=analysis,
            closure_plan=closure_plan,
            closure=closure,
            pixi_lock_hash=HASH,
        )


def _reports() -> tuple[HarnessCorrectionAnalysisReport, HarnessCorrectionClosureReport]:
    analysis = HarnessCorrectionAnalysisReport.model_construct(
        analysis_plan_hash=HASH,
        report_hash=HASH,
        original_sealed_primary_effect=7 / 23,
        original_sealed_primary_interval=(0.0, 13 / 23),
        corrected_primary_effect=1 / 23,
        primary_inference=_inference(1 / 23, 0.0, 3 / 23),
        valid_vs_unrelated_inference=_inference(0.0, 0.0, 0.0),
        valid_vs_corrupted_inference=_inference(15 / 23, 7 / 23, 21 / 23),
    )
    closure = HarnessCorrectionClosureReport.model_construct(
        closure_plan_hash=HASH,
        run_id="correction-figure-test",
        changed_execution_outcome_count=19,
        eligible_case_count=23,
        original_dialogue_correct_count=13,
        original_probe_correct_count=20,
        corrected_dialogue_correct_count=18,
        corrected_probe_correct_count=19,
        original_primary_effect=7 / 23,
        corrected_primary_effect=1 / 23,
        corrected_primary_interval=(0.0, 3 / 23),
        corrected_valid_vs_unrelated_effect=0.0,
        corrected_valid_vs_corrupted_effect=15 / 23,
        claims=(
            CorrectionClaim(
                claim_id="probe_prediction_advantage",
                status="not_supported_after_complete_correction",
                estimate=1 / 23,
                plain_language="Not supported.",
                evidence_scope="Post-hoc correction.",
            ),
            CorrectionClaim(
                claim_id="relevance_specific_advantage",
                status="not_supported_after_complete_correction",
                estimate=0.0,
                plain_language="Not supported.",
                evidence_scope="Post-hoc correction.",
            ),
        ),
    )
    return analysis, closure


def _inference(mean: float, lower: float, upper: float) -> PairedCaseInference:
    return PairedCaseInference.model_construct(
        mean_case_effect=mean,
        interval_lower=lower,
        interval_upper=upper,
    )
