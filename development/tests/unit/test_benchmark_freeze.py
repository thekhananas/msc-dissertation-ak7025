"""Tests for the explicit pre-freeze binding step."""

from datetime import UTC, datetime
from pathlib import Path

import yaml

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import (
    ManifestStatus,
    authored_manifest_content_hash,
)
from socratic_tutor.benchmark.freeze import prepare_benchmark_freeze
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.readiness import build_benchmark_readiness_report
from socratic_tutor.benchmark.research_checks import (
    CaseLinkedShortcutAuditPlan,
    run_case_linked_shortcut_audit,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
SHORTCUT_PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-case-linked-shortcuts.yaml"
ANALYSIS_SPEC_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"


def test_freeze_preparation_binds_current_manifest_readiness_and_analysis(tmp_path: Path) -> None:
    shortcut_summary_path = tmp_path / "case-shortcuts.json"
    run_case_linked_shortcut_audit(
        plan=CaseLinkedShortcutAuditPlan.model_validate(
            yaml.safe_load(SHORTCUT_PLAN_PATH.read_text(encoding="utf-8"))
        ),
        manifest_path=MANIFEST_PATH,
        output_path=shortcut_summary_path,
    )
    readiness_path = tmp_path / "readiness.json"
    readiness = build_benchmark_readiness_report(
        manifest_path=MANIFEST_PATH,
        case_linked_shortcut_summary_path=shortcut_summary_path,
        output_path=readiness_path,
    )

    freeze_time = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
    preparation = prepare_benchmark_freeze(
        manifest_path=MANIFEST_PATH,
        readiness_report_path=readiness_path,
        analysis_specification_path=ANALYSIS_SPEC_PATH,
        frozen_at_utc=freeze_time,
        output_path=tmp_path / "freeze-preparation.json",
    )

    manifest = load_and_verify_manifest(MANIFEST_PATH)
    expected = manifest.model_copy(
        update={
            "status": ManifestStatus.FROZEN,
            "frozen_at_utc": freeze_time,
            "manifest_hash": None,
        }
    )
    assert preparation.source_draft_manifest_hash == model_content_hash(manifest)
    assert preparation.frozen_manifest_hash == authored_manifest_content_hash(expected)
    assert preparation.readiness_report_hash == readiness.report_hash
