"""Command interface for the bounded tracker study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from socratic_tutor.tracker_study.analysis import run_development_analysis
from socratic_tutor.tracker_study.analysis_spec import (
    load_tracker_study_analysis_specification,
)
from socratic_tutor.tracker_study.config import (
    load_hand_worked_trace,
    load_tracker_study_configuration,
)
from socratic_tutor.tracker_study.simulation import (
    publish_development_matrix,
    simulate_verified_development_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the glass-box tracker study")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Verify the frozen study inputs")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--trace", type=Path, required=True)
    validate_analysis = commands.add_parser(
        "validate-analysis",
        help="Verify the prespecified tracker-study analysis",
    )
    validate_analysis.add_argument("--config", type=Path, required=True)
    validate_analysis.add_argument("--analysis", type=Path, required=True)
    analyse = commands.add_parser(
        "analyse-development",
        help="Analyse verified development trajectories under the frozen specification",
    )
    analyse.add_argument("--config", type=Path, required=True)
    analyse.add_argument("--analysis", type=Path, required=True)
    analyse.add_argument("--simulation-manifest", type=Path, required=True)
    analyse.add_argument("--pixi-lock", type=Path, required=True)
    analyse.add_argument("--output-root", type=Path, required=True)
    analyse.add_argument("--run-id", required=True)
    analyse.add_argument("--code-revision", required=True)
    simulate = commands.add_parser(
        "simulate-development",
        help="Run and publish the frozen development stress matrix",
    )
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--output-root", type=Path, required=True)
    simulate.add_argument("--run-id", required=True)
    simulate.add_argument("--code-revision", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except Exception as error:
        _print_json(
            {
                "command": str(args.command),
                "error_type": type(error).__name__,
                "message": str(error),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(
        {"command": str(args.command), "result": result, "status": "ok"},
        stream=sys.stdout,
    )
    return 0


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    configuration = load_tracker_study_configuration(cast(Path, args.config))
    if args.command == "validate":
        trace = load_hand_worked_trace(cast(Path, args.trace), configuration)
        return {
            "study_id": configuration.study_id,
            "configuration_hash": configuration.configuration_hash,
            "trace_hash": trace.trace_hash,
            "tracker_count": len(configuration.trackers.tracker_ids),
            "channel_count": len(configuration.observation_models),
            "stress_condition_count": len(configuration.stress_conditions),
            "test_episodes_per_condition": configuration.experiment.test_episodes_per_condition,
            "turns_per_episode": configuration.experiment.turns_per_episode,
            "claim_scope": configuration.claim_scope,
        }
    if args.command == "validate-analysis":
        specification = load_tracker_study_analysis_specification(
            cast(Path, args.analysis),
            configuration,
        )
        return {
            "study_id": specification.study_id,
            "configuration_hash": specification.configuration_hash,
            "analysis_specification_hash": specification.analysis_specification_hash,
            "primary_metric": specification.primary.metric,
            "candidate_tracker": specification.primary.candidate_tracker,
            "reference_tracker": specification.primary.reference_tracker,
            "adverse_condition_count": len(specification.primary.adverse_conditions),
            "development_results_inspected_before_freeze": (
                specification.development_results_inspected_before_freeze
            ),
            "claim_scope": specification.claim_scope,
        }
    if args.command == "analyse-development":
        specification = load_tracker_study_analysis_specification(
            cast(Path, args.analysis),
            configuration,
        )
        report = run_development_analysis(
            simulation_manifest_path=cast(Path, args.simulation_manifest),
            configuration=configuration,
            specification=specification,
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
            analysis_code_revision=cast(str, args.code_revision),
        )
        return report.model_dump(mode="json")
    if args.command == "simulate-development":
        matrix, replay_hash = simulate_verified_development_matrix(configuration)
        manifest = publish_development_matrix(
            matrix,
            replay_content_hash=replay_hash,
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
            code_revision=cast(str, args.code_revision),
        )
        return manifest.model_dump(mode="json")
    raise ValueError(f"Unknown tracker-study command: {args.command}")


def _print_json(value: dict[str, Any], *, stream: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), file=stream)
