"""Command interface for the sealed canonical primary analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    load_canonical_primary_source_plan,
    run_canonical_primary_analysis,
)
from socratic_tutor.benchmark.artifacts import artifact_locations
from socratic_tutor.repository_state import current_clean_revision

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "acquisition-study" / "v1-canonical-primary-analysis.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse the public stream from the sealed canonical acquisition study"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        revision = current_clean_revision(
            PROJECT_ROOT,
            dirty_message="Commit or remove all visible changes before canonical analysis",
            untracked_files="all",
            required_branch="main",
            operation_name="Canonical analysis",
        )
        source_plan = load_canonical_primary_source_plan(args.plan)
        manifest = run_canonical_primary_analysis(
            args.plan,
            analysis_code_revision=revision,
        )
        output_root = PROJECT_ROOT / source_plan.output_root
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-canonical-analyse-primary",
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
                "command": "acquisition-canonical-analyse-primary",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
