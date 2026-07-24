"""Regression checks for the draft v1 authored benchmark manifest."""

import json
from pathlib import Path

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ArtifactClass, ManifestStatus
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.public.safety import find_public_payload_violations

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"


def test_v1_manifest_is_complete_draft_and_content_addressed() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)

    assert manifest.benchmark_version == "v1"
    assert manifest.status is ManifestStatus.DRAFT
    assert len(manifest.cases) == 24
    assert len(manifest.reviews) == 24
    assert all(review.decision.value == "pending" for review in manifest.reviews)
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
