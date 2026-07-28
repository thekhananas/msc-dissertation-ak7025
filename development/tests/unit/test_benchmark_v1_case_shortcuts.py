"""Checks for the case-linked lexical retrieval audit."""

import ast
import contextlib
import io
import re
from pathlib import Path

import yaml

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.research_checks import (
    CaseLinkedShortcutAuditPlan,
    run_case_linked_shortcut_audit,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-case-linked-shortcuts.yaml"
PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


def test_case_linked_shortcut_variants_execute_as_declared() -> None:
    plan = _load_plan()
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    cases = {item.public.case_id: item for item in manifest.cases}

    for anchor in plan.anchors:
        prompt_path = BENCHMARK_ROOT / cases[anchor.source_case_id].public.evidence_probe_ref
        prompt = prompt_path.read_text(encoding="utf-8")
        source_code = PYTHON_BLOCK.findall(prompt)[0]
        assert _run_code(source_code, anchor.source_case_id) == anchor.expected_stdout

    for variant in plan.audit_examples:
        ast.parse(variant.code)
        assert _run_code(variant.code, variant.example_id) == variant.expected_stdout


def test_case_linked_shortcut_audit_covers_v1_and_rejects_answer_reuse(tmp_path: Path) -> None:
    summary = run_case_linked_shortcut_audit(
        _load_plan(),
        manifest_path=MANIFEST_PATH,
        output_path=tmp_path / "case-shortcuts.json",
    )

    assert summary.gate_passed is True
    assert summary.surface_accuracy == 0.0
    assert summary.anchor_count == 8
    assert summary.audit_count == 16
    assert len(summary.covered_concepts) == 4
    assert len(summary.covered_misconceptions) == 8
    assert all(item.predicted_stdout != item.expected_stdout for item in summary.predictions)


def _load_plan() -> CaseLinkedShortcutAuditPlan:
    return CaseLinkedShortcutAuditPlan.model_validate(
        yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    )


def _run_code(code: str, identifier: str) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, f"<case-shortcut:{identifier}>", "exec"), {})
    return stdout.getvalue()
