"""Tests for the versioned observable failure taxonomy."""

from pathlib import Path

import pytest

from socratic_tutor.benchmark.failure_taxonomy import (
    FailureCategory,
    FailureTaxonomy,
    failure_taxonomy_hash,
    load_failure_taxonomy,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-failure-taxonomy.yaml"


def test_v1_failure_taxonomy_defines_all_observable_categories() -> None:
    taxonomy = load_failure_taxonomy(TAXONOMY_PATH)

    assert taxonomy.taxonomy_version == "v1"
    assert {spec.category for spec in taxonomy.categories} == set(FailureCategory)
    assert len(failure_taxonomy_hash(taxonomy)) == 64


def test_failure_taxonomy_rejects_missing_categories() -> None:
    taxonomy = load_failure_taxonomy(TAXONOMY_PATH)
    payload = taxonomy.model_dump(mode="python")
    payload["categories"] = payload["categories"][:-1]

    with pytest.raises(ValueError, match="every required category"):
        FailureTaxonomy.model_validate(payload)


def test_failure_taxonomy_rejects_hidden_reasoning_label() -> None:
    taxonomy = load_failure_taxonomy(TAXONOMY_PATH)
    payload = taxonomy.model_dump(mode="python")
    payload["forbidden_labels"] = ("hallucinated reasoning",)

    with pytest.raises(ValueError, match="cannot be used"):
        FailureTaxonomy.model_validate(payload)
