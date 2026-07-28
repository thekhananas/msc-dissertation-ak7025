"""Regression checks for the frozen v1 authored benchmark manifest."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ArtifactClass, ManifestStatus
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.validation import (
    BenchmarkValidationError,
    validate_manifest_structure,
)
from socratic_tutor.benchmark.public.safety import find_public_payload_violations

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"


def test_v1_manifest_is_complete_frozen_and_content_addressed() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)

    assert manifest.benchmark_version == "v1"
    assert manifest.status is ManifestStatus.FROZEN
    assert manifest.frozen_at_utc == datetime(2026, 8, 22, 17, 8, 51, tzinfo=UTC)
    assert manifest.manifest_hash == (
        "cd68ac06118ed60bb7ca70face733622052bf9723bbb34240a93ff56fb3836cc"
    )
    assert len(manifest.cases) == 24
    assert len(manifest.reviews) == 24
    assert all(review.decision.value == "approved" for review in manifest.reviews)
    assert all(review.reviewer == "Konstantinos Gkoutzis" for review in manifest.reviews)
    assert all(str(review.reviewed_on) == "2026-08-22" for review in manifest.reviews)
    assert len(manifest.files) == 222


def test_v1_public_projection_excludes_evaluator_material() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(manifest)
    evaluator = project_evaluator_manifest(manifest, public)

    assert find_public_payload_violations(public.model_dump(mode="json")) == ()
    assert len(public.cases) == 24
    assert all("evaluator/" not in entry.path for entry in public.files)
    assert all(
        entry.artifact_class
        in {
            ArtifactClass.CRITERION_PROBE,
            ArtifactClass.CRITERION_TEST,
            ArtifactClass.CRITERION_RUBRIC,
            ArtifactClass.LABEL_RATIONALE,
            ArtifactClass.REVIEW,
            ArtifactClass.STRUCTURAL_DIFFERENCE,
        }
        for entry in evaluator.files
    )
    serialized = json.dumps(public.model_dump(mode="json"), sort_keys=True)
    assert "criterion_probe_id" not in serialized
    assert "evidence_pattern" not in serialized
    assert "misconception_id" not in serialized


def test_frozen_v1_manifest_requires_its_own_content_hash() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    frozen = manifest.model_copy(
        update={
            "status": ManifestStatus.FROZEN,
            "frozen_at_utc": datetime(2026, 8, 22, tzinfo=UTC),
            "manifest_hash": "0" * 64,
        }
    )

    with pytest.raises(BenchmarkValidationError, match="frozen-manifest-hash-mismatch"):
        validate_manifest_structure(frozen)
