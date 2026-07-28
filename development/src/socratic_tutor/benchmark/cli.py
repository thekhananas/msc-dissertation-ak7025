"""Fail-closed command interface for benchmark validation and rehearsals."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.offline import (
    OfflineCriterionPlan,
    run_offline_criterion_phase,
)
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.reporting import evaluate_published_run
from socratic_tutor.benchmark.evidence_scoring import score_evidence_responses
from socratic_tutor.benchmark.failure_taxonomy import (
    failure_taxonomy_hash,
    load_failure_taxonomy,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public.offline import (
    OfflineDecisionPlan,
    run_offline_decision_phase,
)
from socratic_tutor.benchmark.research_checks import (
    SensitivityPlan,
    ShortcutAuditPlan,
    run_sensitivity_analysis,
    run_shortcut_audit,
)
from socratic_tutor.contracts import ContractModel


class SmokeManifest(ContractModel):
    """Final marker for one complete two-phase offline rehearsal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.smoke_manifest.v1"] = "benchmark.smoke_manifest.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    source_manifest_hash: Sha256
    public_projection_hash: Sha256
    evaluator_projection_hash: Sha256
    decision_summary_hash: Sha256
    criterion_summary_hash: Sha256
    evaluation_report_hash: Sha256
    smoke_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> "SmokeManifest":
        if self.smoke_hash != model_content_hash(self, exclude={"smoke_hash"}):
            raise ValueError("Smoke manifest hash does not match its content")
        return self


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate one authored manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--require-frozen", action="store_true")

    smoke = commands.add_parser("smoke", help="Run the two-case offline pipeline")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--decision-plan", type=Path, required=True)
    smoke.add_argument("--criterion-plan", type=Path, required=True)
    smoke.add_argument("--output-root", type=Path, required=True)

    shortcuts = commands.add_parser("shortcuts", help="Run the lexical shortcut baseline")
    shortcuts.add_argument("--input", type=Path, required=True)
    shortcuts.add_argument("--output", type=Path, required=True)

    evidence = commands.add_parser("evidence-score", help="Score recorded evidence responses")
    evidence.add_argument("--manifest", type=Path, required=True)
    evidence.add_argument("--responses", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)

    sensitivity = commands.add_parser("sensitivity", help="Run seeded design sensitivity")
    sensitivity.add_argument("--input", type=Path, required=True)
    sensitivity.add_argument("--output", type=Path, required=True)

    taxonomy = commands.add_parser("failure-taxonomy-validate", help="Validate review taxonomy")
    taxonomy.add_argument("--input", type=Path, required=True)

    analysis = commands.add_parser(
        "analysis-spec-validate", help="Validate pre-registered analysis rules"
    )
    analysis.add_argument("--input", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Verify and summarize local datasets")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, gate_passed = _dispatch(args)
    except Exception as error:
        _print_json(
            {
                "command": str(args.command),
                "error_type": type(error).__name__,
                "message": str(error),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(
        {
            "command": str(args.command),
            "result": _json_value(result),
            "status": "ok" if gate_passed else "gate_failed",
        },
        stream=sys.stdout,
    )
    return 0 if gate_passed else 2


def _dispatch(args: argparse.Namespace) -> tuple[BaseModel | dict[str, object], bool]:
    command = cast(str, args.command)
    if command == "validate":
        return _validate(args), True
    if command == "smoke":
        return _smoke(args), True
    if command == "shortcuts":
        summary = run_shortcut_audit(
            _load_model(cast(Path, args.input), ShortcutAuditPlan),
            output_path=cast(Path, args.output),
        )
        return summary, summary.gate_passed
    if command == "evidence-score":
        summary = score_evidence_responses(
            manifest_path=cast(Path, args.manifest),
            responses_path=cast(Path, args.responses),
            output_path=cast(Path, args.output),
        )
        return summary, summary.complete
    if command == "sensitivity":
        summary = run_sensitivity_analysis(
            _load_model(cast(Path, args.input), SensitivityPlan),
            output_path=cast(Path, args.output),
        )
        return summary, summary.gate_passed
    if command == "failure-taxonomy-validate":
        taxonomy = load_failure_taxonomy(cast(Path, args.input))
        return {
            "taxonomy_version": taxonomy.taxonomy_version,
            "category_count": len(taxonomy.categories),
            "taxonomy_hash": failure_taxonomy_hash(taxonomy),
        }, True
    if command == "analysis-spec-validate":
        specification = load_analysis_specification(cast(Path, args.input))
        return {
            "benchmark_version": specification.benchmark_version,
            "primary_metric": specification.primary_metric,
            "specification_hash": analysis_specification_hash(specification),
        }, True
    if command == "evaluate":
        return (
            evaluate_published_run(
                dataset_root=cast(Path, args.dataset_root),
                output_path=cast(Path, args.output),
            ),
            True,
        )
    raise ValueError(f"Unknown benchmark command: {command}")


def _validate(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_and_verify_manifest(cast(Path, args.manifest))
    if cast(bool, args.require_frozen) and manifest.status is not ManifestStatus.FROZEN:
        raise ValueError("A frozen benchmark manifest is required for this run")
    public = project_public_manifest(manifest)
    evaluator = project_evaluator_manifest(manifest, public)
    return {
        "benchmark_version": manifest.benchmark_version,
        "case_count": len(manifest.cases),
        "evaluator_projection_hash": evaluator.projection_hash,
        "file_count": len(manifest.files),
        "pending_review_count": sum(
            review.decision.value == "pending" for review in manifest.reviews
        ),
        "public_projection_hash": public.projection_hash,
        "source_manifest_hash": public.source_manifest_hash,
        "status": manifest.status.value,
    }


def _smoke(args: argparse.Namespace) -> SmokeManifest:
    manifest_path = cast(Path, args.manifest)
    benchmark_root = manifest_path.resolve().parent
    output_root = cast(Path, args.output_root).resolve()
    decision_plan = _load_model(cast(Path, args.decision_plan), OfflineDecisionPlan)
    authored = load_and_verify_manifest(manifest_path)
    public = project_public_manifest(authored, projected_at_utc=decision_plan.created_at_utc)
    write_immutable_json(output_root / "public_manifest.json", public)
    write_immutable_json(output_root / "offline_decision_plan.json", decision_plan)
    decision_summary = run_offline_decision_phase(
        manifest=public,
        benchmark_root=benchmark_root,
        plan=decision_plan,
        output_root=output_root,
    )
    criterion_plan = _load_model(cast(Path, args.criterion_plan), OfflineCriterionPlan)
    evaluator = project_evaluator_manifest(
        authored,
        public,
        projected_at_utc=decision_plan.created_at_utc,
    )
    write_immutable_json(output_root / "evaluator_manifest.json", evaluator)
    write_immutable_json(output_root / "offline_criterion_plan.json", criterion_plan)
    criterion_summary = run_offline_criterion_phase(
        public_manifest=public,
        evaluator_manifest=evaluator,
        benchmark_root=benchmark_root,
        plan=criterion_plan,
        output_root=output_root,
    )
    evaluation = evaluate_published_run(
        dataset_root=output_root / "datasets",
        output_path=output_root / "evaluation_summary.json",
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.smoke_manifest.v1",
        "benchmark_version": public.benchmark_version,
        "run_id": decision_plan.run_id,
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "evaluator_projection_hash": evaluator.projection_hash,
        "decision_summary_hash": model_content_hash(decision_summary),
        "criterion_summary_hash": model_content_hash(criterion_summary),
        "evaluation_report_hash": evaluation.report_hash,
    }
    draft = SmokeManifest.model_construct(
        _fields_set=set(content),
        **content,
        smoke_hash="0" * 64,
    )
    smoke = SmokeManifest.model_validate(
        {**content, "smoke_hash": model_content_hash(draft, exclude={"smoke_hash"})}
    )
    write_immutable_json(output_root / "smoke_manifest.json", smoke)
    return smoke


def _load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read command input: {path}") from error
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            raw = yaml.safe_load(content)
        else:
            raw = json.loads(content)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Command input is not valid JSON or YAML: {path}") from error
    return model_type.model_validate(raw)


def _json_value(value: BaseModel | dict[str, object]) -> object:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _print_json(value: dict[str, object], *, stream: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), file=stream)
