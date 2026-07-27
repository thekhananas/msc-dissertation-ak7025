"""Regression checks for the v1 lexical shortcut audit plan."""

from pathlib import Path

import yaml

from socratic_tutor.benchmark.research_checks import (
    ShortcutAuditPlan,
    ShortcutVariantKind,
    run_shortcut_audit,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-shortcuts.yaml"


def test_v1_shortcut_plan_covers_surface_and_semantic_variants() -> None:
    plan = ShortcutAuditPlan.model_validate(yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8")))

    kinds = {example.variant_kind for example in plan.audit_examples}
    assert kinds == {
        ShortcutVariantKind.IDENTIFIER_RENAME,
        ShortcutVariantKind.PARAPHRASE,
        ShortcutVariantKind.CLUE_REMOVAL,
        ShortcutVariantKind.SEMANTIC_CHANGE,
    }
    assert len(plan.audit_examples) == 8
    assert plan.maximum_surface_accuracy == 0.5


def test_v1_shortcut_audit_records_the_current_surface_leakage(tmp_path: Path) -> None:
    plan = ShortcutAuditPlan.model_validate(yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8")))

    summary = run_shortcut_audit(plan, output_path=tmp_path / "shortcuts.json")

    assert summary.surface_accuracy == 0.625
    assert summary.gate_passed is False
