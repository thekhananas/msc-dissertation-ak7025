"""Command interface for publishing the development acquisition ablations."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.ablation_runner import (
    publish_development_ablation_matrix,
    run_verified_development_ablation_matrix,
)
from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.benchmark.artifacts import artifact_locations

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the verified development acquisition ablations"
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
            args.source_manifest.parent / f"ablations-{short_revision}"
        )
        run_id = args.run_id or f"acquisition-development-ablations-{short_revision}"
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        matrix, replay_hash = run_verified_development_ablation_matrix(
            args.source_manifest,
            specification=specification,
            analysis=analysis,
        )
        manifest = publish_development_ablation_matrix(
            matrix,
            replay_content_hash=replay_hash,
            output_root=output_root,
            run_id=run_id,
            code_revision=code_revision,
            pixi_lock_path=args.pixi_lock,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "publish-development-ablations",
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
                "command": "publish-development-ablations",
                "result": {
                    "bounded_source_reproduction_verified": (
                        manifest.bounded_source_reproduction_verified
                    ),
                    "canonical_claim_allowed": manifest.canonical_claim_allowed,
                    "case_prediction_count": manifest.case_prediction_count,
                    "cell_count": manifest.cell_count,
                    "code_revision": manifest.code_revision,
                    "deterministic_replay_verified": manifest.deterministic_replay_verified,
                    "environment_count": manifest.environment_count,
                    "episodes_per_environment": manifest.episodes_per_environment,
                    "manifest_hash": manifest.manifest_hash,
                    "matrix_content_hash": manifest.matrix_content_hash,
                    "row_count": manifest.row_count,
                    "run_id": manifest.run_id,
                    "selected_probe_count": manifest.selected_probe_count,
                    "source_manifest_hash": manifest.source_manifest_hash,
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
        raise ValueError("Commit source changes before publishing the development ablations")
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
