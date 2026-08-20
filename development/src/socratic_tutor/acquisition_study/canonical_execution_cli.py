"""Short command interface for sealing and running the canonical study."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from socratic_tutor.acquisition_study.canonical_execution import (
    load_canonical_execution_plan,
    run_canonical_execution,
    seal_canonical_execution,
)
from socratic_tutor.benchmark.artifacts import artifact_locations

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "acquisition-study" / "v1-canonical.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal or run the canonical reliability-aware acquisition study"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal", help="Verify prerequisites and create the pre-run seal")
    seal.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    seal.add_argument("--sealed-on", type=date.fromisoformat, default=date.today())
    run = subparsers.add_parser("run", help="Run the exact study authorised by the seal")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_canonical_execution_plan(args.plan)
        if args.command == "seal":
            manifest, gate = seal_canonical_execution(
                args.plan,
                sealed_on=args.sealed_on,
            )
            result_payload: dict[str, object] = {
                "gate_passed": gate.gate_passed,
                "pre_run_manifest_hash": manifest.manifest_hash,
                "pre_run_gate_report_hash": gate.report_hash,
            }
            output_root = PROJECT_ROOT / plan.pre_run_output_root
        else:
            print(
                "Running the sealed canonical matrix and exact retry. "
                "This can take several minutes.",
                file=sys.stderr,
            )
            run_manifest = run_canonical_execution(args.plan)
            result_payload = run_manifest.model_dump(mode="json")
            output_root = PROJECT_ROOT / plan.canonical_output_root
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": f"acquisition-canonical-{args.command}",
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
                "command": f"acquisition-canonical-{args.command}",
                "result": result_payload,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
