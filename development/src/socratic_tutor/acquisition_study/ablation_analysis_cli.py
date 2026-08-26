"""Command interface for publishing the development ablation analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.ablation_analysis import (
    run_development_ablation_analysis,
)
from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.benchmark.artifacts import artifact_locations
from socratic_tutor.repository_state import current_clean_revision

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the descriptive development acquisition-ablation analysis"
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--environments",
        type=Path,
        default=PROJECT_ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=PROJECT_ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
    )
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code_revision = current_clean_revision(
            PROJECT_ROOT,
            dirty_message="Commit source changes before publishing the ablation analysis",
        )
        short_revision = code_revision[:7]
        output_root = args.output_root or (
            args.source_manifest.parent / f"analysis-{short_revision}"
        )
        run_id = args.run_id or f"acquisition-development-ablation-analysis-{short_revision}"
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        report = run_development_ablation_analysis(
            source_manifest_path=args.source_manifest,
            specification=specification,
            analysis=analysis,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            run_id=run_id,
            analysis_code_revision=code_revision,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "analyse-development-ablations",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "error",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "artifact_locations": artifact_locations(output_root),
                "command": "analyse-development-ablations",
                "result": {
                    "analysis_execution_plan_hash": report.analysis_execution_plan_hash,
                    "analysis_status": report.analysis_status,
                    "bounded_source_reproduction_verified": (
                        report.bounded_source_reproduction_verified
                    ),
                    "canonical_claim_allowed": report.canonical_claim_allowed,
                    "dense_sparse_selection_effects": [
                        row.model_dump(mode="json") for row in report.dense_sparse_selection_effects
                    ],
                    "episodes_per_environment": report.episodes_per_environment,
                    "held_out_effects": [
                        row.model_dump(mode="json") for row in report.held_out_effects
                    ],
                    "held_out_factorial_interaction": (
                        report.held_out_factorial_interaction.model_dump(mode="json")
                    ),
                    "report_hash": report.report_hash,
                    "result_status": report.result_status,
                    "run_id": report.run_id,
                    "source_ablation_manifest_hash": (report.source_ablation_manifest_hash),
                },
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
