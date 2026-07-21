"""Argument boundary for evaluator-only criterion generation and scoring."""

import argparse
from pathlib import Path

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.command_io import load_command_model
from socratic_tutor.benchmark.evaluator.models import EvaluatorBenchmarkManifest
from socratic_tutor.benchmark.evaluator.offline import (
    CriterionGenerationSummary,
    OfflineCriterionPlan,
    run_offline_criterion_phase,
)
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest


def run_criterion_command(argv: list[str]) -> CriterionGenerationSummary:
    """Parse explicit sealed-run paths and execute the private evaluator phase."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    public = load_command_model(args.public_manifest, PublicBenchmarkManifest)
    evaluator = load_command_model(args.evaluator_manifest, EvaluatorBenchmarkManifest)
    plan = load_command_model(args.plan, OfflineCriterionPlan)
    output_root = args.output_root.resolve()
    write_immutable_json(output_root / "public_manifest.json", public)
    write_immutable_json(output_root / "evaluator_manifest.json", evaluator)
    write_immutable_json(output_root / "offline_criterion_plan.json", plan)
    return run_offline_criterion_phase(
        public_manifest=public,
        evaluator_manifest=evaluator,
        benchmark_root=args.benchmark_root,
        plan=plan,
        output_root=output_root,
    )
