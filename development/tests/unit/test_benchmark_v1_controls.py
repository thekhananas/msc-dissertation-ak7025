"""Checks for v1 control specifications before manifest assembly."""

from pathlib import Path

import yaml

from socratic_tutor.benchmark.design import load_design

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"
CONTROL_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "controls"


def test_unrelated_control_is_complete_balanced_and_cross_concept() -> None:
    plan = load_design(DESIGN_PATH)
    cases = {case.case_id: case for case in plan.held_out_cases}
    spec = yaml.safe_load((CONTROL_ROOT / "unrelated.yaml").read_text(encoding="utf-8"))

    assert spec["schema_id"] == "benchmark.unrelated_control.v1"
    assignments = spec["assignments"]
    assert set(assignments) == set(cases)
    assert set(assignments.values()) == set(cases)
    assert all(
        cases[target].concept_id != cases[source].concept_id
        for target, source in assignments.items()
    )
    assert spec["rule"] == "use_the_other_cases_evidence_summary_without_criterion_access"


def test_corruption_control_has_only_declared_summary_inputs() -> None:
    spec = yaml.safe_load((CONTROL_ROOT / "corruption.yaml").read_text(encoding="utf-8"))

    assert spec == {
        "schema_id": "benchmark.corruption_control.v1",
        "transformation": "invert_pass_fail_summary",
        "input_fields": ["passed", "failed"],
        "reject_undeclared_fields": True,
        "seed": 20260812,
    }
