"""Contracts for the pre-criterion external decision audit."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.external_audit import ExternalDecisionAccountingReport
from socratic_tutor.benchmark.generation import ModelRoute
from socratic_tutor.benchmark.hashing import model_content_hash


def test_accounting_report_reconciles_complete_case_and_request_counts() -> None:
    report = _report()

    assert report.gate_passed
    assert report.complete_case_count == 24
    assert report.condition_prediction_count == 96
    assert report.criterion_artifact_count == 0


def test_accounting_report_rejects_inconsistent_prediction_count() -> None:
    report = _report()

    with pytest.raises(ValidationError, match="prediction count"):
        ExternalDecisionAccountingReport.model_validate(
            {
                **report.model_dump(mode="python"),
                "condition_prediction_count": 95,
            }
        )


def _report() -> ExternalDecisionAccountingReport:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_accounting_report.v1",
        "run_id": "external-audit-unit-001",
        "benchmark_version": "v1",
        "source_manifest_hash": "1" * 64,
        "public_projection_hash": "2" * 64,
        "protocol_hash": "3" * 64,
        "generation_report_hash": "4" * 64,
        "seal_plan_hash": "5" * 64,
        "decision_run_plan_hash": "6" * 64,
        "global_seal_hash": "7" * 64,
        "decision_publication_hash": "8" * 64,
        "model_route": ModelRoute(provider="cerebras", model="gpt-oss-120b"),
        "system_prompt_version": "external-evaluation-system-v1",
        "system_prompt_sha256": "9" * 64,
        "decision_code_revision": "decision-revision",
        "decision_dirty_worktree": False,
        "audit_code_revision": "audit-revision",
        "audit_pixi_lock_hash": "a" * 64,
        "tracker_version": "simple-v1",
        "policy_version": "heuristic-v1",
        "planned_case_count": 24,
        "complete_case_count": 24,
        "missing_case_count": 0,
        "invalid_case_count": 0,
        "recorded_request_count": 48,
        "public_request_count": 24,
        "evidence_request_count": 24,
        "evidence_execution_count": 24,
        "usable_evidence_count": 24,
        "condition_prediction_count": 96,
        "backend_fingerprints": ("fp-unit",),
        "criterion_artifact_count": 0,
        "checks_passed": ("source_hashes", "criterion_absence"),
        "audited_at_utc": datetime(2026, 9, 1, 9, 0, tzinfo=UTC),
        "gate_passed": True,
    }
    draft = ExternalDecisionAccountingReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    return ExternalDecisionAccountingReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
