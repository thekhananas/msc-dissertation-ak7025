"""Command interface for publishing the development negative-control analysis."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.negative_control_analysis import (
    run_development_negative_control_analysis,
)
from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.benchmark.artifacts import artifact_locations

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the descriptive development negative-control analysis"
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
        code_revision = _current_clean_revision()
        short_revision = code_revision[:7]
        output_root = args.output_root or (
            args.source_manifest.parent / f"analysis-{short_revision}"
        )
        run_id = args.run_id or f"acquisition-development-negative-analysis-{short_revision}"
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        report = run_development_negative_control_analysis(
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
                    "command": "analyse-development-negative-control",
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
                "command": "analyse-development-negative-control",
                "result": {
                    "analysis_execution_plan_hash": report.analysis_execution_plan_hash,
                    "analysis_status": report.analysis_status,
                    "canonical_claim_allowed": report.canonical_claim_allowed,
                    "environment_effects": [
                        row.model_dump(mode="json") for row in report.environment_effects
                    ],
                    "held_out_effect": report.held_out_effect.model_dump(mode="json"),
                    "report_hash": report.report_hash,
                    "result_status": report.result_status,
                    "run_id": report.run_id,
                    "source_negative_control_manifest_hash": (
                        report.source_negative_control_manifest_hash
                    ),
                },
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _current_clean_revision() -> str:
    status = _git("status", "--porcelain")
    if status:
        raise ValueError("Commit source changes before publishing the negative-control analysis")
    revision = _git("rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Git did not return a full source revision")
    return revision


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
