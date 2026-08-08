from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.resource_figure import (
    ResourceFigureError,
    render_resource_figure,
)
from socratic_tutor.benchmark.resource_reconciliation import (
    ConditionBurdenSummary,
    ProviderPhaseSummary,
    ResourceReconciliationPlan,
    ResourceReconciliationReport,
    SandboxWorkloadSummary,
)
from socratic_tutor.contracts import ContractModel


def test_renders_resource_trade_off_and_retries_exactly(tmp_path: Path) -> None:
    plan_path, report_path, pixi_lock = _write_sources(tmp_path)
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 18, 0, tzinfo=UTC)

    first = render_resource_figure(
        resource_plan_path=plan_path,
        resource_report_path=report_path,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
        generated_at_utc=generated_at,
    )
    retry = render_resource_figure(
        resource_plan_path=plan_path,
        resource_report_path=report_path,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
    )

    assert first == retry
    assert not first.post_hoc_result_used_as_primary
    assert not first.deployment_cost_claim_supported
    assert (output_root / "quality_and_resource_burden.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "quality_and_resource_burden.svg").read_text(encoding="utf-8")
    assert "Observed accuracy rose from 13/23 to 20/23 fixed cases" in svg
    assert "It is shown for transparency" in svg
    assert "Not measured: provider cost" in svg
    assert "model responses (48 inputs + 24 outcomes)" in svg
    csv_text = (output_root / "resource_figure_data.csv").read_text(encoding="utf-8")
    assert "probe_informed_observed,accuracy,0.8695652173913043" in csv_text
    assert "provider,total_tokens,32583,tokens" in csv_text


def test_rejects_a_different_environment_lock(tmp_path: Path) -> None:
    plan_path, report_path, pixi_lock = _write_sources(tmp_path)
    pixi_lock.write_text("different environment\n", encoding="utf-8")

    with pytest.raises(ResourceFigureError, match="another Pixi lock"):
        render_resource_figure(
            resource_plan_path=plan_path,
            resource_report_path=report_path,
            pixi_lock_path=pixi_lock,
            output_root=tmp_path / "publication",
            publication_code_revision="a" * 40,
        )


def _write_sources(root: Path) -> tuple[Path, Path, Path]:
    pixi_lock = root / "pixi.lock"
    pixi_lock.write_text("locked environment\n", encoding="utf-8")
    created_at = datetime(2026, 9, 1, 23, 0, tzinfo=UTC)
    plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.resource_reconciliation_plan.v1",
        "benchmark_version": "v1",
        "run_id": "resource-figure-test",
        "generation_report_hash": _sha(1),
        "criterion_report_hash": _sha(2),
        "decision_seal_report_hash": _sha(3),
        "primary_report_hash": _sha(4),
        "correction_report_hash": _sha(5),
        "decision_responses_sha256": _sha(6),
        "criterion_responses_sha256": _sha(7),
        "method": "observed_external_workload_reconciliation_v1",
        "analysis_code_revision": "9" * 40,
        "analysis_pixi_lock_hash": file_sha256(pixi_lock.read_bytes()),
        "created_at_utc": created_at,
    }
    plan = _hashed_model(ResourceReconciliationPlan, plan_content, "plan_hash")

    decision_phase = ProviderPhaseSummary(
        phase="decision_generation",
        response_count=48,
        responses_with_token_usage=48,
        responses_with_latency=48,
        responses_with_reported_cost=0,
        input_tokens=10936,
        output_tokens=8909,
        summed_provider_latency_ms=22199,
        first_response_at_utc=created_at,
        last_response_at_utc=created_at + timedelta(seconds=633.094638),
        response_capture_span_seconds=633.094638,
        provider_ids=("cerebras",),
        model_ids=("gpt-oss-120b",),
        provider_failure_count=0,
        retry_count=0,
    )
    criterion_phase = ProviderPhaseSummary(
        phase="criterion_generation",
        response_count=24,
        responses_with_token_usage=24,
        responses_with_latency=24,
        responses_with_reported_cost=0,
        input_tokens=4963,
        output_tokens=7775,
        summed_provider_latency_ms=10501,
        first_response_at_utc=created_at + timedelta(days=1),
        last_response_at_utc=created_at + timedelta(days=1, seconds=364.194266),
        response_capture_span_seconds=364.194266,
        provider_ids=("cerebras",),
        model_ids=("gpt-oss-120b",),
        provider_failure_count=0,
        retry_count=None,
    )
    sandbox = SandboxWorkloadSummary(
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
    )
    conditions = (
        _condition("dialogue_only", "sealed_primary", 13, 1, 0, ("public",)),
        _condition(
            "probe_informed_observed",
            "sealed_primary",
            20,
            2,
            1,
            ("public", "evidence"),
        ),
        _condition(
            "probe_informed_post_hoc_corrected",
            "post_hoc_correction",
            23,
            2,
            1,
            ("public", "evidence"),
        ),
    )
    report_content = {
        "schema_version": 1,
        "schema_id": "benchmark.resource_reconciliation_report.v1",
        "benchmark_version": "v1",
        "run_id": plan.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "provider_phases": (decision_phase, criterion_phase),
        "total_provider_response_count": 72,
        "total_input_tokens": 15899,
        "total_output_tokens": 16684,
        "total_summed_provider_latency_ms": 32700,
        "provider_failure_count": 0,
        "provider_cost_status": "not_reported_by_provider",
        "provider_cost_usd": None,
        "sandbox": sandbox,
        "condition_burden": conditions,
        "measured_artifact_bytes_excluding_this_report": 2269069,
        "local_analysis_network_calls": 0,
        "local_analysis_sandbox_calls": 0,
        "interpretation_scope": "observed_single_run_workload_not_deployment_cost",
        "completed_at_utc": created_at + timedelta(days=2),
    }
    report = _hashed_model(ResourceReconciliationReport, report_content, "report_hash")
    plan_path = root / "resource_plan.json"
    report_path = root / "resource_report.json"
    write_immutable_json(plan_path, plan)
    write_immutable_json(report_path, report)
    return plan_path, report_path, pixi_lock


def _condition(
    condition: str,
    evidence_status: str,
    correct: int,
    response_channels: int,
    sandbox_executions: int,
    required_channels: tuple[str, ...],
) -> ConditionBurdenSummary:
    return ConditionBurdenSummary.model_validate(
        {
            "condition": condition,
            "evidence_status": evidence_status,
            "eligible_case_count": 23,
            "correct_case_count": correct,
            "accuracy_on_fixed_authored_cases": correct / 23,
            "external_model_response_channels_per_case": response_channels,
            "sandbox_executions_per_case": sandbox_executions,
            "required_channels": required_channels,
        }
    )


def _hashed_model[ModelT: ContractModel](
    model: type[ModelT], content: Mapping[str, object], hash_field: str
) -> ModelT:
    draft = model.model_construct(
        _fields_set=set(content),
        **content,
        **{hash_field: "0" * 64},
    )
    return model.model_validate(
        {**content, hash_field: model_content_hash(draft, exclude={hash_field})}
    )


def _sha(value: int) -> str:
    return f"{value:064x}"
