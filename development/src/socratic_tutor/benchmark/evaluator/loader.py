"""YAML loading for authored benchmark manifests."""

from pathlib import Path
from typing import Any, cast

import yaml

from socratic_tutor.benchmark.evaluator.inventory import verify_inventory
from socratic_tutor.benchmark.evaluator.models import AuthoredBenchmarkManifest
from socratic_tutor.benchmark.evaluator.validation import validate_manifest_structure


def load_authored_manifest(path: Path) -> AuthoredBenchmarkManifest:
    """Load one strict authored manifest without opening referenced files."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Benchmark manifest must be a mapping")
    return AuthoredBenchmarkManifest.model_validate(cast(dict[str, Any], raw))


def load_and_verify_manifest(path: Path) -> AuthoredBenchmarkManifest:
    """Load a manifest and verify structure plus every referenced file byte."""

    manifest = load_authored_manifest(path)
    validate_manifest_structure(manifest)
    verify_inventory(path.parent, manifest.files, ignored_paths=frozenset({path.name}))
    return manifest
