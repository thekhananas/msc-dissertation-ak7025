from __future__ import annotations

import pytest

from socratic_tutor.benchmark.result_interpretation import (
    CombinedConclusion,
    PrimaryConclusion,
    SpecificityConclusion,
    classify_primary,
    combine_conclusions,
)


@pytest.mark.parametrize(
    ("lower", "upper", "threshold", "expected"),
    (
        (0.10, 0.40, 0.10, "positive"),
        (0.11, 0.40, 0.10, "positive"),
        (-0.50, -0.01, 0.10, "negative"),
        (0.00, 0.57, 0.10, "inconclusive"),
        (-0.20, 0.20, 0.10, "inconclusive"),
        (0.05, 0.20, 0.10, "inconclusive"),
    ),
)
def test_classifies_primary_result_conservatively(
    lower: float,
    upper: float,
    threshold: float,
    expected: PrimaryConclusion,
) -> None:
    assert (
        classify_primary(
            interval_lower=lower,
            interval_upper=upper,
            minimum_interpretable_effect=threshold,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("primary", "specificity", "expected"),
    (
        ("positive", "specific", "positive_and_specific"),
        ("positive", "non_specific", "positive_but_non_specific"),
        ("negative", "specific", "negative"),
        ("negative", "non_specific", "negative"),
        ("inconclusive", "specific", "inconclusive"),
        ("inconclusive", "non_specific", "inconclusive"),
    ),
)
def test_secondary_specificity_cannot_rescue_primary(
    primary: PrimaryConclusion,
    specificity: SpecificityConclusion,
    expected: CombinedConclusion,
) -> None:
    assert combine_conclusions(primary, specificity) == expected
