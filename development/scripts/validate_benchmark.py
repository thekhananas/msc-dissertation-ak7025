"""Validate and summarize one authored benchmark manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to the authored benchmark YAML")
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Reject draft manifests intended only for development",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_and_verify_manifest(args.manifest)
    if args.require_frozen and manifest.status is not ManifestStatus.FROZEN:
        raise SystemExit("A frozen benchmark manifest is required for this run")

    public_manifest = project_public_manifest(manifest)
    evaluator_manifest = project_evaluator_manifest(manifest, public_manifest)
    summary = {
        "benchmark_version": manifest.benchmark_version,
        "case_count": len(manifest.cases),
        "evaluator_projection_hash": evaluator_manifest.projection_hash,
        "file_count": len(manifest.files),
        "pending_review_count": sum(
            review.decision.value == "pending" for review in manifest.reviews
        ),
        "public_projection_hash": public_manifest.projection_hash,
        "source_manifest_hash": public_manifest.source_manifest_hash,
        "status": manifest.status.value,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
