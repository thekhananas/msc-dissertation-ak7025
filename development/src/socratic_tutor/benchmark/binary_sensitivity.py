"""Sensitivity analysis for the executed 24-case paired binary design."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, median
from typing import Literal

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel


class BinarySensitivityScenario(ContractModel):
    """One plausible joint distribution of paired criterion correctness."""

    scenario_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    both_correct_probability: float = Field(ge=0.0, le=1.0)
    dialogue_only_correct_probability: float = Field(ge=0.0, le=1.0)
    valid_only_correct_probability: float = Field(ge=0.0, le=1.0)
    both_incorrect_probability: float = Field(ge=0.0, le=1.0)
    expected_effect: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_probabilities(self) -> BinarySensitivityScenario:
        total = sum(
            (
                self.both_correct_probability,
                self.dialogue_only_correct_probability,
                self.valid_only_correct_probability,
                self.both_incorrect_probability,
            )
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Paired binary outcome probabilities must sum to one")
        expected = self.valid_only_correct_probability - self.dialogue_only_correct_probability
        if abs(self.expected_effect - expected) > 1e-9:
            raise ValueError(
                "Expected effect must equal valid-only minus dialogue-only probability"
            )
        return self


class BinarySensitivityPlan(ContractModel):
    """Frozen simulation assumptions matching the executed benchmark design."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.binary_sensitivity_plan.v1"] = (
        "benchmark.binary_sensitivity_plan.v1"
    )
    case_count: Literal[24]
    model_runs_per_case: Literal[1]
    policy_threshold: float = Field(ge=0.0, le=1.0)
    simulation_count: int = Field(ge=1000, le=100_000)
    case_missing_probability: float = Field(ge=0.0, lt=1.0)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=0.5)
    seed: int = Field(ge=0, le=2**32 - 1)
    methodology_clarification_hash: Sha256
    analysis_specification_hash: Sha256
    external_protocol_hash: Sha256
    historical_plan_hash: Sha256
    historical_summary_hash: Sha256
    scenarios: tuple[BinarySensitivityScenario, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_plan(self) -> BinarySensitivityPlan:
        if self.policy_threshold != 0.6:
            raise ValueError("Binary sensitivity requires the frozen 0.60 policy threshold")
        if self.minimum_interpretable_effect != 0.1:
            raise ValueError("Binary sensitivity requires the frozen 0.10 effect threshold")
        if self.alpha != 0.05:
            raise ValueError("Binary sensitivity v1 requires alpha 0.05")
        ids = tuple(item.scenario_id for item in self.scenarios)
        if len(set(ids)) != len(ids):
            raise ValueError("Binary sensitivity scenario identifiers must be unique")
        effects = tuple(item.expected_effect for item in self.scenarios)
        if 0.0 not in effects:
            raise ValueError("Binary sensitivity scenarios must include the null effect")
        resolution = 1 / self.case_count
        if not any(abs(effect - 2 * resolution) < 1e-9 for effect in effects):
            raise ValueError("Scenarios must include the two-net-case effect")
        if not any(abs(effect - 3 * resolution) < 1e-9 for effect in effects):
            raise ValueError("Scenarios must include the three-net-case effect")
        return self


class BinarySensitivityScenarioSummary(ContractModel):
    """Monte Carlo operating characteristics for one paired outcome scenario."""

    scenario_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_effect: float = Field(ge=-1.0, le=1.0)
    equivalent_net_cases_at_full_corpus: float
    simulation_count: int = Field(ge=1)
    mean_eligible_cases: float = Field(ge=0.0)
    mean_observed_effect: float = Field(ge=-1.0, le=1.0)
    median_interval_width: float = Field(ge=0.0, le=2.0)
    interval_width_p05: float = Field(ge=0.0, le=2.0)
    interval_width_p95: float = Field(ge=0.0, le=2.0)
    minimum_effect_attainment_rate: float = Field(ge=0.0, le=1.0)
    missed_minimum_effect_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    exact_mcnemar_rejection_rate: float = Field(ge=0.0, le=1.0)


class BinarySensitivityReport(ContractModel):
    """Seeded sensitivity report that supersedes only the old design assumption."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.binary_sensitivity_report.v1"] = (
        "benchmark.binary_sensitivity_report.v1"
    )
    method: Literal["paired_binary_monte_carlo_v1"]
    interval_method: Literal["exact_empirical_percentile_bootstrap"]
    primary_test: Literal["exact_two_sided_mcnemar"]
    case_count: Literal[24]
    model_runs_per_case: Literal[1]
    simulation_count: int = Field(ge=1000)
    case_missing_probability: float = Field(ge=0.0, lt=1.0)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    full_corpus_effect_resolution: float = Field(gt=0.0, le=1.0)
    two_net_cases_effect: float = Field(gt=0.0, le=1.0)
    three_net_cases_effect: float = Field(gt=0.0, le=1.0)
    threshold_location: Literal["strictly_between_two_and_three_net_cases"]
    historical_design_status: Literal["retained_but_not_applicable_to_executed_design"]
    historical_plan_hash: Sha256
    historical_summary_hash: Sha256
    methodology_clarification_hash: Sha256
    scenarios: tuple[BinarySensitivityScenarioSummary, ...] = Field(min_length=4)
    plan_hash: Sha256
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> BinarySensitivityReport:
        if not (
            self.two_net_cases_effect
            < self.minimum_interpretable_effect
            < self.three_net_cases_effect
        ):
            raise ValueError("The effect threshold must lie between two and three net cases")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Binary sensitivity report hash does not match its content")
        return self


def load_binary_sensitivity_plan(path: Path) -> BinarySensitivityPlan:
    """Load the corrected sensitivity assumptions from YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read binary sensitivity plan: {path}") from error
    return BinarySensitivityPlan.model_validate(raw)


def run_binary_sensitivity(
    plan: BinarySensitivityPlan,
    *,
    output_path: Path,
) -> BinarySensitivityReport:
    """Simulate paired binary cases using the exact executed sample structure."""

    summaries = tuple(_simulate_scenario(plan, scenario) for scenario in plan.scenarios)
    resolution = 1 / plan.case_count
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.binary_sensitivity_report.v1",
        "method": "paired_binary_monte_carlo_v1",
        "interval_method": "exact_empirical_percentile_bootstrap",
        "primary_test": "exact_two_sided_mcnemar",
        "case_count": plan.case_count,
        "model_runs_per_case": plan.model_runs_per_case,
        "simulation_count": plan.simulation_count,
        "case_missing_probability": plan.case_missing_probability,
        "minimum_interpretable_effect": plan.minimum_interpretable_effect,
        "full_corpus_effect_resolution": resolution,
        "two_net_cases_effect": 2 * resolution,
        "three_net_cases_effect": 3 * resolution,
        "threshold_location": "strictly_between_two_and_three_net_cases",
        "historical_design_status": "retained_but_not_applicable_to_executed_design",
        "historical_plan_hash": plan.historical_plan_hash,
        "historical_summary_hash": plan.historical_summary_hash,
        "methodology_clarification_hash": plan.methodology_clarification_hash,
        "scenarios": summaries,
        "plan_hash": model_content_hash(plan),
    }
    draft = BinarySensitivityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = BinarySensitivityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_path, report)
    return report


def _simulate_scenario(
    plan: BinarySensitivityPlan,
    scenario: BinarySensitivityScenario,
) -> BinarySensitivityScenarioSummary:
    randomizer = random.Random(f"{plan.seed}:{scenario.scenario_id}")
    interval_cache: dict[tuple[int, int, int], tuple[float, float]] = {}
    mcnemar_cache: dict[tuple[int, int], float] = {}
    eligible_counts: list[int] = []
    observed_effects: list[float] = []
    interval_widths: list[float] = []
    threshold_hits = 0
    rejections = 0
    probabilities = _cumulative_probabilities(scenario)

    for _ in range(plan.simulation_count):
        dialogue_only = 0
        valid_only = 0
        ties = 0
        for _case in range(plan.case_count):
            if randomizer.random() < plan.case_missing_probability:
                continue
            outcome = _draw_outcome(randomizer.random(), probabilities)
            if outcome == "dialogue_only":
                dialogue_only += 1
            elif outcome == "valid_only":
                valid_only += 1
            else:
                ties += 1
        eligible = dialogue_only + valid_only + ties
        if eligible == 0:
            continue
        effect = (valid_only - dialogue_only) / eligible
        interval_key = (dialogue_only, ties, valid_only)
        if interval_key not in interval_cache:
            interval_cache[interval_key] = _exact_empirical_bootstrap_interval(*interval_key)
        lower, upper = interval_cache[interval_key]
        mcnemar_key = (dialogue_only, valid_only)
        if mcnemar_key not in mcnemar_cache:
            mcnemar_cache[mcnemar_key] = _exact_mcnemar_p_value(*mcnemar_key)
        p_value = mcnemar_cache[mcnemar_key]
        eligible_counts.append(eligible)
        observed_effects.append(effect)
        interval_widths.append(upper - lower)
        threshold_hits += effect >= plan.minimum_interpretable_effect
        rejections += p_value < plan.alpha

    completed = len(observed_effects)
    if completed != plan.simulation_count:
        raise ValueError("Sensitivity simulation produced an empty eligible sample")
    attainment = threshold_hits / completed
    return BinarySensitivityScenarioSummary(
        scenario_id=scenario.scenario_id,
        rationale=scenario.rationale,
        expected_effect=scenario.expected_effect,
        equivalent_net_cases_at_full_corpus=scenario.expected_effect * plan.case_count,
        simulation_count=completed,
        mean_eligible_cases=fmean(eligible_counts),
        mean_observed_effect=fmean(observed_effects),
        median_interval_width=median(interval_widths),
        interval_width_p05=_quantile(interval_widths, 0.05),
        interval_width_p95=_quantile(interval_widths, 0.95),
        minimum_effect_attainment_rate=attainment,
        missed_minimum_effect_rate=(
            1.0 - attainment
            if scenario.expected_effect >= plan.minimum_interpretable_effect
            else None
        ),
        exact_mcnemar_rejection_rate=rejections / completed,
    )


def _cumulative_probabilities(
    scenario: BinarySensitivityScenario,
) -> tuple[float, float, float]:
    both = scenario.both_correct_probability
    dialogue = both + scenario.dialogue_only_correct_probability
    valid = dialogue + scenario.valid_only_correct_probability
    return both, dialogue, valid


def _draw_outcome(
    value: float,
    cumulative: tuple[float, float, float],
) -> Literal["both_correct", "dialogue_only", "valid_only", "both_incorrect"]:
    if value < cumulative[0]:
        return "both_correct"
    if value < cumulative[1]:
        return "dialogue_only"
    if value < cumulative[2]:
        return "valid_only"
    return "both_incorrect"


def _exact_empirical_bootstrap_interval(
    negative_count: int,
    zero_count: int,
    positive_count: int,
) -> tuple[float, float]:
    """Return exact percentile bounds for resampling the empirical {-1,0,1} values."""

    count = negative_count + zero_count + positive_count
    probabilities: Mapping[int, float] = {
        -1: negative_count / count,
        0: zero_count / count,
        1: positive_count / count,
    }
    distribution: dict[int, float] = {0: 1.0}
    for _ in range(count):
        next_distribution: dict[int, float] = {}
        for current_sum, current_probability in distribution.items():
            for value, probability in probabilities.items():
                next_sum = current_sum + value
                next_distribution[next_sum] = (
                    next_distribution.get(next_sum, 0.0) + current_probability * probability
                )
        distribution = next_distribution
    lower = _weighted_quantile(distribution, 0.025) / count
    upper = _weighted_quantile(distribution, 0.975) / count
    return lower, upper


def _exact_mcnemar_p_value(dialogue_only: int, valid_only: int) -> float:
    discordant = dialogue_only + valid_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value) * 0.5**discordant
        for value in range(0, min(dialogue_only, valid_only) + 1)
    )
    return min(1.0, 2 * tail)


def _weighted_quantile(distribution: Mapping[int, float], probability: float) -> int:
    cumulative = 0.0
    for value, weight in sorted(distribution.items()):
        cumulative += weight
        if cumulative >= probability - 1e-15:
            return value
    return max(distribution)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
