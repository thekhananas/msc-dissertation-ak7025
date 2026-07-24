"""Check that every private criterion prompt matches its authored tests and rubric."""

import ast
import re
from pathlib import Path
from typing import Any, cast

import yaml

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


def test_all_v1_criterion_prompts_match_tests_and_rubrics() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)

    for item in manifest.cases:
        criterion = item.criterion
        prompt = (BENCHMARK_ROOT / criterion.criterion_prompt_ref).read_text(encoding="utf-8")
        tests = _load_yaml(criterion.test_bundle_ref)
        rubric = _load_yaml(criterion.rubric_ref)
        rationale = (BENCHMARK_ROOT / criterion.label_rationale_ref).read_text(encoding="utf-8")
        blocks = PYTHON_BLOCK.findall(prompt)

        assert len(blocks) == 1, item.public.case_id
        tree = ast.parse(blocks[0])
        function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert function_names == [tests["function"]], item.public.case_id
        assert tests["schema_id"] == "benchmark.authored_tests.v1"
        assert len(tests["checks"]) >= 2
        assert rubric == {
            "schema_id": "benchmark.criterion_rubric.v1",
            "label": rubric["label"],
            "true_when": "all_authored_tests_pass",
            "false_when": "any_authored_test_fails",
            "null_when": "execution_is_unavailable",
        }
        assert "not evidence of general mastery" in rationale


def _load_yaml(relative_path: str) -> dict[str, Any]:
    value = yaml.safe_load((BENCHMARK_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)
