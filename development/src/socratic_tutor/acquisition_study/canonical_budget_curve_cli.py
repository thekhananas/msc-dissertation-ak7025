"""Command interface for the sealed canonical secondary budget curve."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.canonical_budget_curve import (
    run_canonical_budget_curve,
)
from socratic_tutor.benchmark.artifacts import artifact_locations

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_PLAN = (
    PROJECT_ROOT / "configs" / "acquisition-study" / "v1-canonical-primary-analysis.yaml"
)
DEFAULT_PRIMARY_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "acquisition-study"
    / "canonical-v1"
    / "primary-analysis-v1"
    / "canonical_primary_analysis_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts" / "acquisition-study" / "canonical-v1" / "secondary-budget-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the sealed 50% results and publish the canonical budget curve"
    )
    parser.add_argument("--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN)
    parser.add_argument("--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        revision = _current_clean_revision()
        manifest = run_canonical_budget_curve(
            args.source_plan,
            args.primary_manifest,
            output_root=args.output_root,
            secondary_code_revision=revision,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-canonical-budget-curve",
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
                "artifact_locations": artifact_locations(args.output_root),
                "command": "acquisition-canonical-budget-curve",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _current_clean_revision() -> str:
    branch = _git("branch", "--show-current")
    status = _git("status", "--porcelain", "--untracked-files=all")
    revision = _git("rev-parse", "HEAD")
    if branch != "main":
        raise ValueError(
            f"Canonical analysis requires branch main, not {branch or 'detached HEAD'}"
        )
    if status:
        raise ValueError("Commit or remove all visible changes before canonical analysis")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Git did not return a full source revision")
    return revision


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
