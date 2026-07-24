"""Check that v1 evidence answer keys agree with their authored code snippets."""

import ast
import contextlib
import io
import re
from pathlib import Path
from typing import Any, cast

import yaml

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


def test_all_v1_evidence_oracles_match_their_prompt_code() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)

    for item in manifest.cases:
        prompt = (BENCHMARK_ROOT / item.public.evidence_probe_ref).read_text(encoding="utf-8")
        tests = yaml.safe_load(
            (BENCHMARK_ROOT / item.public.evidence_test_ref).read_text(encoding="utf-8")
        )
        blocks = PYTHON_BLOCK.findall(prompt)

        assert len(blocks) == 2, item.public.case_id
        ast.parse(blocks[0])
        output = _run_authored_code(blocks[0], item.public.case_id)
        assert len(tests["checks"]) == 1, item.public.case_id
        expected = tests["checks"][0]["expected"]
        assert _normalise_output(output, expected) == _normalise_value(expected), (
            item.public.case_id,
            output,
            expected,
        )


def _run_authored_code(code: str, case_id: str) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, f"<evidence:{case_id}>", "exec"), {})
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 1, (case_id, lines)
    return lines[0]


def _normalise_output(output: str, expected: Any) -> Any:
    if isinstance(expected, bool):
        return output.casefold() == "true"
    if isinstance(expected, (int, float, str)):
        return output if isinstance(expected, str) else type(expected)(output)
    try:
        parsed = ast.literal_eval(output)
    except (SyntaxError, ValueError):
        tokens = _split_top_level(output)
        assert len(tokens) == len(expected), (output, expected)
        parsed = [
            _normalise_output(token, item) for token, item in zip(tokens, expected, strict=True)
        ]
    return _normalise_value(parsed)


def _split_top_level(output: str) -> list[str]:
    tokens: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(output):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char.isspace() and depth == 0:
            if output[start:index].strip():
                tokens.append(output[start:index].strip())
            start = index + 1
    if output[start:].strip():
        tokens.append(output[start:].strip())
    return tokens


def _normalise_value(value: Any) -> Any:
    if isinstance(value, list | tuple):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(_normalise_value(item) for item in sequence)
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return tuple(
            sorted(
                ((key, _normalise_value(item)) for key, item in mapping.items()),
                key=lambda pair: str(pair[0]),
            )
        )
    return value
