"""Command interface for the acquisition-study runtime forecast."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.acquisition_study.runtime_forecast import (
    run_development_runtime_forecast,
    runtime_seconds,
)
from socratic_tutor.benchmark.artifacts import artifact_locations

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the development matrix and forecast the canonical workload"
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
            args.source_manifest.parent / f"runtime-forecast-{short_revision}"
        )
        run_id = args.run_id or f"acquisition-development-runtime-{short_revision}"
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        report = run_development_runtime_forecast(
            source_manifest_path=args.source_manifest,
            environment_path=args.environments,
            analysis_path=args.analysis,
            specification=specification,
            analysis=analysis,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            run_id=run_id,
            runtime_code_revision=code_revision,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "forecast-development-runtime",
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
                "command": "forecast-development-runtime",
                "result": {
                    "current_runner_feasible_for_canonical_run": (
                        report.current_runner_feasible_for_canonical_run
                    ),
                    "guarded_projected_canonical_elapsed_seconds": runtime_seconds(
                        report.guarded_projected_canonical_elapsed_nanoseconds
                    ),
                    "guarded_projected_canonical_peak_rss_bytes": (
                        report.guarded_projected_canonical_peak_rss_bytes
                    ),
                    "maximum_development_peak_rss_bytes": (
                        report.maximum_development_peak_rss_bytes
                    ),
                    "median_development_elapsed_seconds": runtime_seconds(
                        report.median_development_elapsed_nanoseconds
                    ),
                    "memory_within_planning_limit": report.memory_within_planning_limit,
                    "report_hash": report.report_hash,
                    "runtime_within_remaining_m7b_time": (report.runtime_within_remaining_m7b_time),
                    "source_matrix_parity_verified": report.source_matrix_parity_verified,
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
        raise ValueError("Commit source changes before publishing the runtime forecast")
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
