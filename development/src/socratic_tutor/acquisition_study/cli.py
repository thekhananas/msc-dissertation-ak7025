"""Command interface for checking the frozen acquisition-study design."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the selective-probing study design")
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        environments, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
    except Exception as error:
        print(
            json.dumps(
                {
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
                "analysis_plan_hash": analysis.plan_hash,
                "candidates_per_episode": environments.episodes.candidates_per_episode,
                "environment_count": len(environments.evaluation_environments),
                "environment_specification_hash": environments.specification_hash,
                "evaluation_episodes_per_environment": (
                    environments.episodes.evaluation_episodes_per_environment
                ),
                "held_out_environment_count": len(analysis.primary.held_out_environments),
                "primary_budget_fraction": analysis.primary.primary_budget_fraction,
                "selected_probes_per_episode": (analysis.primary.exact_selected_probes_per_episode),
                "status": "ok",
                "study_id": analysis.study_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
