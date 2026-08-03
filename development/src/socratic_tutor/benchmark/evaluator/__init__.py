"""Evaluator-only benchmark construction and outcome-reveal package."""

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import AuthoredBenchmarkManifest
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.requests import (
    CriterionArtifactReadError,
    CriterionChannelRequestBuilder,
)

__all__ = [
    "AuthoredBenchmarkManifest",
    "CriterionArtifactReadError",
    "CriterionChannelRequestBuilder",
    "load_and_verify_manifest",
    "project_evaluator_manifest",
    "project_public_manifest",
]
