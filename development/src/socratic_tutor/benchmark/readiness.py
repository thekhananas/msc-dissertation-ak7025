"""Pre-freeze checks for the authored benchmark, independent of model outcomes."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ReviewDecision
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.research_checks import CaseLinkedShortcutAuditSummary
from socratic_tutor.contracts import ContractModel

_PYTHON_BLOCK = re.compile(r"```python\n(?P<code>.*?)```", re.DOTALL)


class ReadinessCheck(ContractModel):
    name: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class CaseReadiness(ContractModel):
    case_id: str = Field(min_length=1)
    checks: tuple[ReadinessCheck, ...] = Field(min_length=1)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class BenchmarkReadinessReport(ContractModel):
    """Content-integrity evidence needed before a separate freeze decision."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.readiness_report.v1"] = "benchmark.readiness_report.v1"
    benchmark_version: str = Field(min_length=1)
    source_manifest_hash: Sha256
    case_count: int = Field(ge=1)
    approved_review_count: int = Field(ge=0)
    global_checks: tuple[ReadinessCheck, ...] = Field(min_length=1)
    case_reports: tuple[CaseReadiness, ...] = Field(min_length=1)
    case_linked_shortcut_summary_hash: Sha256
    gate_passed: bool
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> BenchmarkReadinessReport:
        expected_gate = all(check.passed for check in self.global_checks) and all(
            item.passed for item in self.case_reports
        )
        if self.gate_passed is not expected_gate:
            raise ValueError("Readiness gate does not match its check results")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Readiness report hash does not match its content")
        return self


def load_case_linked_shortcut_summary(path: Path) -> CaseLinkedShortcutAuditSummary:
    """Load a content-addressed result from the case-linked shortcut audit."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read case-linked shortcut summary: {path}") from error
    return CaseLinkedShortcutAuditSummary.model_validate(value)


def build_benchmark_readiness_report(
    *,
    manifest_path: Path,
    case_linked_shortcut_summary_path: Path,
    output_path: Path,
) -> BenchmarkReadinessReport:
    """Run deterministic authored-content checks and publish an immutable report."""

    manifest = load_and_verify_manifest(manifest_path)
    root = manifest_path.resolve().parent
    shortcut_summary = load_case_linked_shortcut_summary(case_linked_shortcut_summary_path)
    source_manifest_hash = model_content_hash(manifest)
    if shortcut_summary.benchmark_version != manifest.benchmark_version:
        raise ValueError("Case-linked shortcut summary belongs to another benchmark version")
    if shortcut_summary.source_manifest_hash != source_manifest_hash:
        raise ValueError("Case-linked shortcut summary belongs to another manifest revision")

    case_reports = tuple(_case_readiness(item, root) for item in manifest.cases)
    approved_reviews = sum(
        review.decision is ReviewDecision.APPROVED for review in manifest.reviews
    )
    public_projection = project_public_manifest(manifest)
    global_checks = (
        ReadinessCheck(
            name="public_projection",
            passed=len(public_projection.cases) == len(manifest.cases),
            detail="Public projection was built after recursive evaluator-field checks.",
        ),
        ReadinessCheck(
            name="independent_review",
            passed=approved_reviews == len(manifest.cases)
            and all(
                review.author.casefold() != review.reviewer.casefold()
                and review.reviewed_on is not None
                for review in manifest.reviews
            ),
            detail="Every case has an approved, dated review by a different named reviewer.",
        ),
        ReadinessCheck(
            name="case_linked_shortcuts",
            passed=shortcut_summary.gate_passed,
            detail=(
                "The retrieval baseline covers four concepts and eight misconceptions "
                f"with surface accuracy {shortcut_summary.surface_accuracy:.3f}."
            ),
        ),
    )
    content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "benchmark.readiness_report.v1",
        "benchmark_version": manifest.benchmark_version,
        "source_manifest_hash": source_manifest_hash,
        "case_count": len(manifest.cases),
        "approved_review_count": approved_reviews,
        "global_checks": global_checks,
        "case_reports": case_reports,
        "case_linked_shortcut_summary_hash": shortcut_summary.summary_hash,
        "gate_passed": all(check.passed for check in global_checks)
        and all(item.passed for item in case_reports),
    }
    draft = BenchmarkReadinessReport.model_construct(
        _fields_set=set(content),
        **content,
        report_hash="0" * 64,
    )
    report = BenchmarkReadinessReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_path, report)
    return report


def _case_readiness(item: Any, root: Path) -> CaseReadiness:
    public = item.public
    criterion = item.criterion
    public_blocks = _python_blocks(root / public.public_fixture_ref)
    evidence_blocks = _python_blocks(root / public.evidence_probe_ref)
    criterion_blocks = _python_blocks(root / criterion.criterion_prompt_ref)
    structural = _structural_check(public_blocks, evidence_blocks, criterion_blocks)
    evidence = _evidence_oracle_check(
        code=evidence_blocks[0] if len(evidence_blocks) == 2 else "",
        tests_path=root / public.evidence_test_ref,
        case_id=public.case_id,
    )
    criterion_check = _criterion_authoring_check(
        code=criterion_blocks[0] if len(criterion_blocks) == 1 else "",
        tests_path=root / criterion.test_bundle_ref,
        rubric_path=root / criterion.rubric_ref,
        rationale_path=root / criterion.label_rationale_ref,
    )
    return CaseReadiness(
        case_id=public.case_id,
        checks=(
            ReadinessCheck(
                name="channel_structure",
                passed=structural,
                detail="Public, evidence, and criterion code remain structurally distinct.",
            ),
            ReadinessCheck(
                name="evidence_oracle",
                passed=evidence,
                detail="Trusted evidence code produces the authored expected output.",
            ),
            ReadinessCheck(
                name="criterion_authoring",
                passed=criterion_check,
                detail="Criterion prompt, tests, rubric, and limitation rationale agree.",
            ),
        ),
    )


def _python_blocks(path: Path) -> list[str]:
    try:
        return _PYTHON_BLOCK.findall(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def _structural_check(
    public_blocks: list[str], evidence_blocks: list[str], criterion_blocks: list[str]
) -> bool:
    if len(public_blocks) != 1 or len(evidence_blocks) != 2 or len(criterion_blocks) != 1:
        return False
    try:
        criterion_fingerprint = _ast_fingerprint(criterion_blocks[0])
        if criterion_blocks[0] in {*public_blocks, *evidence_blocks}:
            return False
        if criterion_fingerprint in {
            _ast_fingerprint(public_blocks[0]),
            _ast_fingerprint(evidence_blocks[0]),
            _ast_fingerprint(evidence_blocks[1]),
        }:
            return False
        criterion_functions = _function_names(criterion_blocks[0])
        public_functions = _function_names(public_blocks[0])
        evidence_functions = _function_names(evidence_blocks[1])
    except SyntaxError:
        return False
    return (
        len(criterion_functions) == 1
        and not set(criterion_functions) & set(public_functions)
        and not set(criterion_functions) & set(evidence_functions)
        and not criterion_functions[0].startswith("predicted_")
    )


def _evidence_oracle_check(*, code: str, tests_path: Path, case_id: str) -> bool:
    try:
        tests = _load_yaml(tests_path)
        checks = cast(list[dict[str, Any]], tests["checks"])
        if len(checks) != 1:
            return False
        expected = checks[0]["expected"]
        stdout = io.StringIO()
        # These are inventory-verified, repository-authored fixtures, never model output.
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, f"<readiness-evidence:{case_id}>", "exec"), {})
        lines = stdout.getvalue().splitlines()
        return len(lines) == 1 and _normalise_output(lines[0], expected) == _normalise_value(
            expected
        )
    except (KeyError, OSError, SyntaxError, TypeError, ValueError, yaml.YAMLError):
        return False


def _criterion_authoring_check(
    *, code: str, tests_path: Path, rubric_path: Path, rationale_path: Path
) -> bool:
    try:
        tests = _load_yaml(tests_path)
        rubric = _load_yaml(rubric_path)
        rationale = rationale_path.read_text(encoding="utf-8")
        functions = _function_names(code)
        return (
            functions == [tests["function"]]
            and tests["schema_id"] == "benchmark.authored_tests.v1"
            and len(cast(list[object], tests["checks"])) >= 2
            and rubric
            == {
                "schema_id": "benchmark.criterion_rubric.v1",
                "label": rubric["label"],
                "true_when": "all_authored_tests_pass",
                "false_when": "any_authored_test_fails",
                "null_when": "execution_is_unavailable",
            }
            and "not evidence of general mastery" in rationale
        )
    except (KeyError, OSError, SyntaxError, TypeError, ValueError, yaml.YAMLError):
        return False


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a YAML mapping")
    return cast(dict[str, Any], value)


def _ast_fingerprint(code: str) -> str:
    return ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)


def _function_names(code: str) -> list[str]:
    tree = ast.parse(code)
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]


def _normalise_output(output: str, expected: Any) -> Any:
    if isinstance(expected, bool):
        return output.casefold() == "true"
    if isinstance(expected, (int, float, str)):
        return output if isinstance(expected, str) else type(expected)(output)
    try:
        parsed = ast.literal_eval(output)
    except (SyntaxError, ValueError):
        parsed = [
            _normalise_output(token, item)
            for token, item in zip(_split_top_level(output), expected, strict=True)
        ]
    return _normalise_value(parsed)


def _normalise_value(value: Any) -> Any:
    if isinstance(value, list | tuple):
        sequence = cast(list[Any] | tuple[Any, ...], value)
        return tuple(_normalise_value(item) for item in sequence)
    if isinstance(value, dict):
        mapping = cast(dict[object, Any], value)
        return tuple(
            sorted(
                ((str(key), _normalise_value(item)) for key, item in mapping.items()),
                key=lambda pair: pair[0],
            )
        )
    return value


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
