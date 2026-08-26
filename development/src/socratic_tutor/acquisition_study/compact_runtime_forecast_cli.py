"""Command interface for the compact acquisition-stream resource forecast."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.compact_runtime_forecast import (
    run_development_compact_runtime_forecast,
    runtime_seconds,
)
from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.benchmark.artifacts import artifact_locations
from socratic_tutor.repository_state import current_clean_revision

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the compact development stream and forecast evaluation resources"
    )
    parser.add_argument("--compact-parity-report", type=Path, required=True)
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
            dirty_message="Commit source changes before publishing the compact runtime forecast",
        )
        short_revision = code_revision[:7]
        output_root = args.output_root or (
            args.compact_parity_report.parent / f"compact-runtime-{short_revision}"
        )
        run_id = args.run_id or f"acquisition-development-compact-runtime-{short_revision}"
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        report = run_development_compact_runtime_forecast(
            compact_parity_report_path=args.compact_parity_report,
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
                    "command": "forecast-compact-acquisition-runtime",
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
                "command": "forecast-compact-acquisition-runtime",
                "result": {
                    "compact_runner_feasible_for_canonical_run": (
                        report.compact_runner_feasible_for_canonical_run
                    ),
                    "exact_stream_parity_verified": report.exact_stream_parity_verified,
                    "guarded_planning_canonical_peak_rss_bytes": (
                        report.guarded_planning_canonical_peak_rss_bytes
                    ),
                    "guarded_projected_canonical_elapsed_seconds": runtime_seconds(
                        report.guarded_projected_canonical_elapsed_nanoseconds
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
                },
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
