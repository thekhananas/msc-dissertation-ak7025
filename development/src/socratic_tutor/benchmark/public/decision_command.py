"""Argument boundary for development-only decision generation."""

import argparse
from pathlib import Path

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.command_io import load_command_model
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.public.offline import (
    DecisionGenerationSummary,
    OfflineDecisionPlan,
    run_offline_decision_phase,
)


def run_decision_command(argv: list[str]) -> DecisionGenerationSummary:
    """Parse explicit public paths and execute one sealed decision phase."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    public = load_command_model(args.public_manifest, PublicBenchmarkManifest)
    plan = load_command_model(args.plan, OfflineDecisionPlan)
    output_root = args.output_root.resolve()
    write_immutable_json(output_root / "public_manifest.json", public)
    write_immutable_json(output_root / "offline_decision_plan.json", plan)
    return run_offline_decision_phase(
        manifest=public,
        benchmark_root=args.benchmark_root,
        plan=plan,
        output_root=output_root,
    )
