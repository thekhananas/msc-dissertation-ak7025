"""Tests for the pre-registered v1 analysis specification."""

from pathlib import Path

import pytest

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
    load_analysis_specification,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"


def test_v1_analysis_specification_is_hashable_and_pre_registered() -> None:
    specification = load_analysis_specification(SPEC_PATH)

    assert specification.benchmark_version == "v1"
    assert specification.aggregation_unit == "case"
    assert specification.bootstrap_resamples == 10_000
    assert specification.primary_metric == "paired_brier_difference"
    assert len(analysis_specification_hash(specification)) == 64


def test_analysis_specification_rejects_mismatched_resampling_counts() -> None:
    specification = load_analysis_specification(SPEC_PATH)
    payload = specification.model_dump(mode="python")
    payload["permutation_resamples"] = 9_999

    with pytest.raises(ValueError, match="one frozen count"):
        AnalysisSpecification.model_validate(payload)


def test_analysis_specification_rejects_non_case_exclusion_rule() -> None:
    specification = load_analysis_specification(SPEC_PATH)
    payload = specification.model_dump(mode="python")
    payload["confirmatory_exclusions"] = ("Remove inconvenient outcomes.",)

    with pytest.raises(ValueError, match="case-level independence"):
        AnalysisSpecification.model_validate(payload)
