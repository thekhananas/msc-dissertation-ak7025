"""Tests for the corrected quality-and-resource publication."""

# These tests intentionally exercise private helpers to isolate validation failures.
# pyright: reportPrivateUsage=false

from datetime import UTC, datetime

import pytest

from socratic_tutor.benchmark.corrected_resource_figure import (
    CorrectedResourceFigureError,
    _figure_data_csv,
    _render_vector_files,
    _validate_join,
)
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisReport,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    HarnessCorrectionClosureReport,
)
from socratic_tutor.benchmark.resource_reconciliation import (
    ResourceReconciliationReport,
    SandboxWorkloadSummary,
)


def test_renders_corrected_quality_beside_unchanged_workload() -> None:
    resource = _resource_report()
    analysis = _analysis_report()
    closure = _closure_report()

    pdf, svg_bytes = _render_vector_files(
        resource,
        closure,
        datetime(2026, 9, 3, 21, 0, tzinfo=UTC),
    )
    svg = svg_bytes.decode("utf-8")
    table = _figure_data_csv(resource, analysis, closure).decode("utf-8")

    assert pdf.startswith(b"%PDF")
    assert "one of 23 eligible decisions" in svg
    assert "18/23" in svg
    assert "19/23" in svg
    assert "72" in svg
    assert "32,583" in svg
    assert "Post-hoc result" in svg
    assert "corrected_quality,net_changed_decisions,1,cases" in table
    assert "full_run,total_tokens,32583,tokens" in table


def test_rejects_a_correction_from_another_run() -> None:
    resource = _resource_report()
    analysis = _analysis_report()
    closure = _closure_report(run_id="another-run")

    with pytest.raises(CorrectedResourceFigureError, match="different runs"):
        _validate_join(resource=resource, analysis=analysis, closure=closure)


def test_rejects_a_result_that_is_not_the_reclosed_comparison() -> None:
    resource = _resource_report()
    analysis = _analysis_report()
    closure = _closure_report(corrected_probe_correct_count=20)

    with pytest.raises(CorrectedResourceFigureError, match="reclosed result"):
        _validate_join(resource=resource, analysis=analysis, closure=closure)


def _analysis_report() -> HarnessCorrectionAnalysisReport:
    return HarnessCorrectionAnalysisReport.model_construct(
        improvement_count=1,
        regression_count=0,
    )


def _resource_report() -> ResourceReconciliationReport:
    sandbox = SandboxWorkloadSummary.model_construct(
        evidence_execution_count=24,
        evidence_completed_count=24,
        evidence_test_pass_count=12,
        evidence_test_fail_count=12,
        reviewed_test_defect_count=3,
        evidence_test_fail_count_after_reviewed_correction=9,
        criterion_execution_count=24,
        criterion_completed_count=23,
        criterion_missing_count=1,
        total_execution_count=48,
        total_completed_count=47,
        execution_time_status="not_recorded",
        cpu_and_memory_status="not_recorded",
    )
    return ResourceReconciliationReport.model_construct(
        run_id="worked-run",
        total_provider_response_count=72,
        total_input_tokens=15899,
        total_output_tokens=16684,
        total_summed_provider_latency_ms=32700,
        sandbox=sandbox,
        measured_artifact_bytes_excluding_this_report=2269069,
    )


def _closure_report(
    *,
    run_id: str = "worked-run",
    corrected_probe_correct_count: int = 19,
) -> HarnessCorrectionClosureReport:
    return HarnessCorrectionClosureReport.model_construct(
        run_id=run_id,
        eligible_case_count=23,
        corrected_dialogue_correct_count=18,
        corrected_probe_correct_count=corrected_probe_correct_count,
    )
