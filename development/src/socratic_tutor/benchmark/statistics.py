"""Deterministic case-level inference for the frozen benchmark analysis plan."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import product
from statistics import fmean, median
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel


class StatisticalAnalysisError(ValueError):
    """Effects or binary outcomes do not satisfy the prespecified analysis contract."""


class PairedCaseInference(ContractModel):
    """One transparent case-level bootstrap and sign-swap result."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.paired_case_inference.v1"] = "benchmark.paired_case_inference.v1"
    case_count: int = Field(ge=1)
    effects_hash: Sha256
    mean_case_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    median_case_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    bootstrap_method: Literal["paired_case_percentile"] = "paired_case_percentile"
    bootstrap_resamples: int = Field(ge=1)
    interval_lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    interval_upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    permutation_method: Literal["within_case_sign_swap"] = "within_case_sign_swap"
    permutation_draw_count: int = Field(ge=1)
    two_sided_permutation_p_value: float = Field(ge=0.0, le=1.0)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    reaches_minimum_interpretable_effect: bool
    random_seed: int = Field(ge=0, le=2**32 - 1)
    result_hash: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> PairedCaseInference:
        if self.interval_lower > self.interval_upper:
            raise ValueError("Bootstrap interval lower bound cannot exceed its upper bound")
        if self.reaches_minimum_interpretable_effect is not (
            self.mean_case_effect >= self.minimum_interpretable_effect
        ):
            raise ValueError("Minimum-effect flag does not match the mean case effect")
        if self.result_hash != model_content_hash(self, exclude={"result_hash"}):
            raise ValueError("Paired inference hash does not match its content")
        return self


class McNemarResult(ContractModel):
    """Exact two-sided McNemar result for a paired binary exploratory comparison."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.mcnemar_result.v1"] = "benchmark.mcnemar_result.v1"
    pair_count: int = Field(ge=1)
    both_correct_count: int = Field(ge=0)
    left_only_correct_count: int = Field(ge=0)
    right_only_correct_count: int = Field(ge=0)
    both_incorrect_count: int = Field(ge=0)
    discordant_pair_count: int = Field(ge=0)
    two_sided_exact_p_value: float = Field(ge=0.0, le=1.0)
    result_hash: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> McNemarResult:
        if self.pair_count != sum(
            (
                self.both_correct_count,
                self.left_only_correct_count,
                self.right_only_correct_count,
                self.both_incorrect_count,
            )
        ):
            raise ValueError("McNemar outcome counts do not equal pair count")
        if self.discordant_pair_count != (
            self.left_only_correct_count + self.right_only_correct_count
        ):
            raise ValueError("McNemar discordant count does not match outcome counts")
        if self.result_hash != model_content_hash(self, exclude={"result_hash"}):
            raise ValueError("McNemar result hash does not match its content")
        return self


def paired_case_inference(
    effects: Sequence[float],
    *,
    bootstrap_resamples: int,
    permutation_resamples: int,
    minimum_interpretable_effect: float,
    random_seed: int,
) -> PairedCaseInference:
    """Summarise independent case effects using the frozen paired analysis rules.

    Repeated generations are averaged into one effect before this function is called.
    That keeps the benchmark case, rather than a generation, as the unit of analysis.
    """

    frozen_effects = _validate_effects(effects)
    if bootstrap_resamples < 1 or permutation_resamples < 1:
        raise StatisticalAnalysisError("Resample counts must be positive")
    if not 0.0 < minimum_interpretable_effect <= 1.0:
        raise StatisticalAnalysisError("Minimum interpretable effect must be in (0, 1]")
    if not 0 <= random_seed <= 2**32 - 1:
        raise StatisticalAnalysisError("Random seed must fit an unsigned 32-bit integer")

    bootstrap = _bootstrap_means(frozen_effects, bootstrap_resamples, random_seed)
    interval_lower = _quantile(bootstrap, 0.025)
    interval_upper = _quantile(bootstrap, 0.975)
    observed_mean = fmean(frozen_effects)
    permutation_p_value, permutation_draw_count = _sign_swap_p_value(
        frozen_effects,
        permutation_resamples,
        random_seed,
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.paired_case_inference.v1",
        "case_count": len(frozen_effects),
        "effects_hash": canonical_sha256(frozen_effects),
        "mean_case_effect": observed_mean,
        "median_case_effect": median(frozen_effects),
        "confidence_level": 0.95,
        "bootstrap_method": "paired_case_percentile",
        "bootstrap_resamples": bootstrap_resamples,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "permutation_method": "within_case_sign_swap",
        "permutation_draw_count": permutation_draw_count,
        "two_sided_permutation_p_value": permutation_p_value,
        "minimum_interpretable_effect": minimum_interpretable_effect,
        "reaches_minimum_interpretable_effect": observed_mean >= minimum_interpretable_effect,
        "random_seed": random_seed,
    }
    draft = PairedCaseInference.model_construct(
        _fields_set=set(content), **content, result_hash="0" * 64
    )
    return PairedCaseInference.model_validate(
        {**content, "result_hash": model_content_hash(draft, exclude={"result_hash"})}
    )


def exact_mcnemar(pairs: Sequence[tuple[bool, bool]]) -> McNemarResult:
    """Calculate the exact two-sided McNemar p-value from paired correctness values."""

    frozen_pairs = tuple(pairs)
    if not frozen_pairs:
        raise StatisticalAnalysisError("McNemar analysis requires at least one paired outcome")
    if any(type(value) is not bool for pair in frozen_pairs for value in pair):
        raise StatisticalAnalysisError("McNemar outcomes must be boolean pairs")
    both_correct = sum(left and right for left, right in frozen_pairs)
    left_only = sum(left and not right for left, right in frozen_pairs)
    right_only = sum(not left and right for left, right in frozen_pairs)
    both_incorrect = sum(not left and not right for left, right in frozen_pairs)
    discordant = left_only + right_only
    p_value = _two_sided_binomial_p_value(left_only, discordant)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.mcnemar_result.v1",
        "pair_count": len(frozen_pairs),
        "both_correct_count": both_correct,
        "left_only_correct_count": left_only,
        "right_only_correct_count": right_only,
        "both_incorrect_count": both_incorrect,
        "discordant_pair_count": discordant,
        "two_sided_exact_p_value": p_value,
    }
    draft = McNemarResult.model_construct(_fields_set=set(content), **content, result_hash="0" * 64)
    return McNemarResult.model_validate(
        {**content, "result_hash": model_content_hash(draft, exclude={"result_hash"})}
    )


def _validate_effects(effects: Sequence[float]) -> tuple[float, ...]:
    frozen = tuple(float(value) for value in effects)
    if not frozen:
        raise StatisticalAnalysisError("Paired inference requires at least one case effect")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in frozen):
        raise StatisticalAnalysisError("Case effects must be finite values in [-1, 1]")
    return frozen


def _bootstrap_means(
    effects: tuple[float, ...],
    resamples: int,
    random_seed: int,
) -> list[float]:
    generator = random.Random(random_seed)
    count = len(effects)
    values = [
        fmean(effects[generator.randrange(count)] for _ in range(count)) for _ in range(resamples)
    ]
    return sorted(values)


def _sign_swap_p_value(
    effects: tuple[float, ...],
    resamples: int,
    random_seed: int,
) -> tuple[float, int]:
    observed = abs(fmean(effects))
    count = len(effects)
    if count <= 16:
        means = (
            abs(fmean(sign * effect for sign, effect in zip(signs, effects, strict=True)))
            for signs in product((-1.0, 1.0), repeat=count)
        )
        values = tuple(means)
        return sum(value >= observed - 1e-12 for value in values) / len(values), len(values)

    generator = random.Random(random_seed + 1)
    exceedances = 0
    for _ in range(resamples):
        simulated = fmean(
            (1.0 if generator.getrandbits(1) else -1.0) * effect for effect in effects
        )
        exceedances += abs(simulated) >= observed - 1e-12
    return (exceedances + 1) / (resamples + 1), resamples


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise StatisticalAnalysisError("Quantile requires at least one value")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _two_sided_binomial_p_value(left_only_correct: int, discordant_pair_count: int) -> float:
    if discordant_pair_count == 0:
        return 1.0
    lower_tail = (
        sum(
            math.comb(discordant_pair_count, value)
            for value in range(
                min(left_only_correct, discordant_pair_count - left_only_correct) + 1
            )
        )
        / 2**discordant_pair_count
    )
    return min(1.0, 2.0 * lower_tail)
