"""Command interface for the bounded tracker study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from socratic_tutor.tracker_study.config import (
    load_hand_worked_trace,
    load_tracker_study_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the glass-box tracker study")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Verify the frozen study inputs")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--trace", type=Path, required=True)
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
    if args.command != "validate":
        raise ValueError(f"Unknown tracker-study command: {args.command}")
    configuration = load_tracker_study_configuration(cast(Path, args.config))
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


def _print_json(value: dict[str, Any], *, stream: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), file=stream)
