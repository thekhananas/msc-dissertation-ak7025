"""Regression checks for the pre-freeze v1 readiness report."""

from pathlib import Path

import yaml

from socratic_tutor.benchmark.readiness import build_benchmark_readiness_report
from socratic_tutor.benchmark.research_checks import (
    CaseLinkedShortcutAuditPlan,
    run_case_linked_shortcut_audit,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
SHORTCUT_PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-case-linked-shortcuts.yaml"


def test_v1_readiness_report_passes_all_authored_content_checks(tmp_path: Path) -> None:
    shortcut_summary_path = tmp_path / "case-shortcuts.json"
    run_case_linked_shortcut_audit(
        CaseLinkedShortcutAuditPlan.model_validate(
            yaml.safe_load(SHORTCUT_PLAN_PATH.read_text(encoding="utf-8"))
        ),
        manifest_path=MANIFEST_PATH,
        output_path=shortcut_summary_path,
    )

    report = build_benchmark_readiness_report(
        manifest_path=MANIFEST_PATH,
        case_linked_shortcut_summary_path=shortcut_summary_path,
        output_path=tmp_path / "readiness.json",
    )

    assert report.gate_passed is True
    assert report.case_count == 24
    assert report.approved_review_count == 24
    assert all(item.passed for item in report.case_reports)
    assert all(check.passed for check in report.global_checks)
