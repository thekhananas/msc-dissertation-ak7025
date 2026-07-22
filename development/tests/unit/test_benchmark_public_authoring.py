"""Checks for public/evidence files before the full v1 manifest exists."""

import ast
import re
from pathlib import Path

from socratic_tutor.benchmark.design import load_design
from socratic_tutor.benchmark.public.safety import find_public_payload_violations

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


def test_authored_public_and_evidence_files_are_separate_and_policy_safe() -> None:
    plan = load_design(DESIGN_PATH)
    authored_concepts = {
        "assignment-evaluation",
        "conditional-control",
        "function-arguments",
        "object-aliasing",
    }
    cases = tuple(case for case in plan.held_out_cases if case.concept_id in authored_concepts)

    assert len(cases) == 24
    for case in cases:
        public = _read(f"public/{case.case_id}-dialogue.md")
        evidence = _read(f"evidence/{case.case_id}-prompt.md")
        combined = f"{public}\n{evidence}"

        assert "Tutor:" in public
        assert "Student:" not in public
        assert "Without running the code" in public
        assert "def predicted_" in evidence
        assert "return ..." in evidence
        assert find_public_payload_violations(combined) == ()
        assert case.criterion_context.casefold() not in combined.casefold()

        public_blocks = PYTHON_BLOCK.findall(public)
        evidence_blocks = PYTHON_BLOCK.findall(evidence)
        assert len(public_blocks) == 1
        assert len(evidence_blocks) == 2
        for code in (*public_blocks, *evidence_blocks):
            ast.parse(code)


def _read(relative_path: str) -> str:
    content = (BENCHMARK_ROOT / relative_path).read_text(encoding="utf-8")
    assert content.strip()
    return content
