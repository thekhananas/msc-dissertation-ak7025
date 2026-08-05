# pyright: reportPrivateUsage=false
"""Tests for the post-hoc executable-probe correction analysis."""

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.benchmark.probe_correction_sensitivity import (
    ProbeCorrectionDirective,
    _summarize,
)
from socratic_tutor.benchmark.statistics import paired_case_inference


def test_removing_three_false_regressions_recomputes_paired_result() -> None:
    original_pairs = (
        *((False, True),) * 10,
        *((True, False),) * 3,
        *((True, True),) * 10,
    )
    original_effects = tuple(float(not left) - float(not right) for left, right in original_pairs)
    inference = paired_case_inference(
        original_effects,
        bootstrap_resamples=1000,
        permutation_resamples=1000,
        minimum_interpretable_effect=0.1,
        random_seed=17,
    )
    primary = PrimaryAnalysisReport.model_construct(inference=inference)

    original = _summarize(primary, original_pairs)
    corrected = _summarize(primary, (*original_pairs[:10], *((True, True),) * 13))

    assert original.valid_evidence_improvement_count == 10
    assert original.valid_evidence_regression_count == 3
    assert abs(original.paired_effect_from_counts - 7 / 23) < 1e-12
    assert corrected.valid_evidence_improvement_count == 10
    assert corrected.valid_evidence_regression_count == 0
    assert abs(corrected.paired_effect_from_counts - 10 / 23) < 1e-12
    assert abs(corrected.mcnemar.two_sided_exact_p_value - 0.001953125) < 1e-12


def test_correction_directive_must_change_the_decision() -> None:
    with pytest.raises(ValidationError, match="must change"):
        ProbeCorrectionDirective(
            case_id="case-1",
            observed_probe_decision=True,
            corrected_probe_decision=True,
            correction_basis="correct_tuple_rejected_as_yaml_list",
            failure_review_record_hash="1" * 64,
            evidence_response_hash="2" * 64,
            evidence_execution_artifact_hash="3" * 64,
            evidence_test_sha256="4" * 64,
        )
