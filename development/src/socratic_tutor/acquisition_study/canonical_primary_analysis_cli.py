"""Command interface for the sealed canonical primary analysis."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    load_canonical_primary_source_plan,
    run_canonical_primary_analysis,
)
from socratic_tutor.benchmark.artifacts import artifact_locations

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
        revision = _current_clean_revision()
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
