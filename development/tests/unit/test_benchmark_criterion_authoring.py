"""Checks for evaluator-private criterion probes before the full manifest exists."""

import ast
import re
from pathlib import Path

import yaml

from socratic_tutor.benchmark.design import load_design

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"
PUBLIC_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
CRITERION_ROOT = PUBLIC_ROOT / "evaluator"
PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)
FUNCTION = re.compile(r"def (?P<name>[a-zA-Z_]\w*)\(")


def test_c1_criterion_probes_are_structurally_distinct_and_tested() -> None:
    plan = load_design(DESIGN_PATH)
    cases = tuple(
        case for case in plan.held_out_cases if case.concept_id == "assignment-evaluation"
    )

    assert len(cases) == 6
    for case in cases:
        prompt = _read(CRITERION_ROOT / "criterion" / f"{case.case_id}-prompt.md")
        tests = yaml.safe_load(_read(CRITERION_ROOT / "criterion" / f"{case.case_id}-tests.yaml"))
        rubric = yaml.safe_load(_read(CRITERION_ROOT / "criterion" / f"{case.case_id}-rubric.yaml"))
        rationale = _read(CRITERION_ROOT / "criterion" / f"{case.case_id}-label-rationale.md")
        structural = _read(CRITERION_ROOT / "structural" / f"{case.case_id}.md")
        evidence = _read(PUBLIC_ROOT / "evidence" / f"{case.case_id}-prompt.md")

        blocks = PYTHON_BLOCK.findall(prompt)
        assert len(blocks) == 1
        tree = ast.parse(blocks[0])
        function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert function_names == [tests["function"]]
        assert tests["schema_id"] == "benchmark.authored_tests.v1"
        assert len(tests["checks"]) >= 2
        assert rubric["schema_id"] == "benchmark.criterion_rubric.v1"
        assert rubric["true_when"] == "all_authored_tests_pass"
        assert rubric["false_when"] == "any_authored_test_fails"
        assert rubric["null_when"] == "execution_is_unavailable"
        assert "not evidence of general mastery" in rationale
        assert "held-out task" in structural
        assert tests["function"] not in evidence
        public = _read(PUBLIC_ROOT / "public" / f"{case.case_id}-dialogue.md")
        assert "criterion" not in public.casefold()


def _read(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    assert content.strip()
    return content
