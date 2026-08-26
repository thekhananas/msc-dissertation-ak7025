"""Command interface for the acquisition-study development analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.acquisition_study.primary_analysis import run_development_primary_analysis
from socratic_tutor.benchmark.artifacts import artifact_locations
from socratic_tutor.repository_state import current_clean_revision

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse a verified acquisition-policy development matrix"
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
            dirty_message="Commit source changes before publishing the development analysis",
        )
        short_revision = code_revision[:7]
        output_root = args.output_root or (
            args.source_manifest.parent / f"primary-analysis-{short_revision}"
        )
        run_id = args.run_id or f"acquisition-development-primary-{short_revision}"
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        report = run_development_primary_analysis(
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
                    "command": "analyse-development",
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
                "command": "analyse-development",
                "result": report.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
