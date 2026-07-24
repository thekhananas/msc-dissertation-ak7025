"""Check that v1 public, evidence, and criterion tasks are not copied across channels."""

import ast
import re
from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


def test_v1_channels_have_distinct_code_and_function_names() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)

    for item in manifest.cases:
        public = _read(item.public.public_fixture_ref)
        evidence = _read(item.public.evidence_probe_ref)
        criterion = _read(item.criterion.criterion_prompt_ref)
        public_blocks = PYTHON_BLOCK.findall(public)
        evidence_blocks = PYTHON_BLOCK.findall(evidence)
        criterion_blocks = PYTHON_BLOCK.findall(criterion)

        assert len(public_blocks) == 1, item.public.case_id
        assert len(evidence_blocks) == 2, item.public.case_id
        assert len(criterion_blocks) == 1, item.public.case_id

        criterion_ast = _ast_fingerprint(criterion_blocks[0])
        assert criterion_blocks[0] not in public_blocks
        assert criterion_blocks[0] not in evidence_blocks
        assert criterion_ast not in {
            _ast_fingerprint(public_blocks[0]),
            _ast_fingerprint(evidence_blocks[0]),
            _ast_fingerprint(evidence_blocks[1]),
        }

        public_functions = _function_names(public_blocks[0])
        evidence_functions = _function_names(evidence_blocks[1])
        criterion_functions = _function_names(criterion_blocks[0])
        assert len(criterion_functions) == 1, item.public.case_id
        assert not set(criterion_functions) & set(public_functions)
        assert not set(criterion_functions) & set(evidence_functions)
        assert "predicted_" not in criterion_functions[0]


def _read(relative_path: str) -> str:
    return (BENCHMARK_ROOT / relative_path).read_text(encoding="utf-8")


def _ast_fingerprint(code: str) -> str:
    return ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)


def _function_names(code: str) -> list[str]:
    tree = ast.parse(code)
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
