"""Hand-calculated checks for the frozen case-level statistical functions."""

import math
from typing import cast

import pytest

from socratic_tutor.benchmark.statistics import (
    StatisticalAnalysisError,
    exact_mcnemar,
    paired_case_inference,
)


def test_paired_inference_uses_one_case_effect_per_independent_case() -> None:
    result = paired_case_inference(
        (0.2, 0.4, 0.6),
        bootstrap_resamples=10_000,
        permutation_resamples=10_000,
        minimum_interpretable_effect=0.1,
        random_seed=20260813,
    )

    assert result.case_count == 3
    assert math.isclose(result.mean_case_effect, 0.4)
    assert math.isclose(result.median_case_effect, 0.4)
    assert result.reaches_minimum_interpretable_effect is True
    assert result.bootstrap_method == "paired_case_percentile"
    assert result.permutation_draw_count == 8
    assert math.isclose(result.two_sided_permutation_p_value, 0.25)
    assert result.interval_lower <= result.mean_case_effect <= result.interval_upper


def test_paired_inference_is_seeded_and_rejects_invalid_effects() -> None:
    first = paired_case_inference(
        (0.4, -0.2, 0.2),
        bootstrap_resamples=1_000,
        permutation_resamples=1_000,
        minimum_interpretable_effect=0.1,
        random_seed=7,
    )
    second = paired_case_inference(
        (0.4, -0.2, 0.2),
        bootstrap_resamples=1_000,
        permutation_resamples=1_000,
        minimum_interpretable_effect=0.1,
        random_seed=7,
    )

    assert first == second
    with pytest.raises(StatisticalAnalysisError, match="finite values"):
        paired_case_inference(
            (float("nan"),),
            bootstrap_resamples=1_000,
            permutation_resamples=1_000,
            minimum_interpretable_effect=0.1,
            random_seed=7,
        )


def test_exact_mcnemar_matches_the_hand_calculated_discordant_result() -> None:
    result = exact_mcnemar(
        (
            (False, True),
            (False, True),
            (False, True),
            (False, True),
            (True, False),
            (True, True),
            (True, True),
            (False, False),
        )
    )

    assert result.pair_count == 8
    assert result.left_only_correct_count == 1
    assert result.right_only_correct_count == 4
    assert result.discordant_pair_count == 5
    assert math.isclose(result.two_sided_exact_p_value, 0.375)


def test_exact_mcnemar_requires_paired_boolean_outcomes() -> None:
    with pytest.raises(StatisticalAnalysisError, match="at least one"):
        exact_mcnemar(())
    malformed = cast(tuple[bool, bool], (True, 1))
    with pytest.raises(StatisticalAnalysisError, match="boolean"):
        exact_mcnemar((malformed,))
