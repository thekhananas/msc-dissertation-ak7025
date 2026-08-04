"""Tests for sensitivity analysis matching the executed paired binary design."""

from pathlib import Path

import pytest

from socratic_tutor.benchmark.binary_sensitivity import (
    BinarySensitivityPlan,
    load_binary_sensitivity_plan,
    run_binary_sensitivity,
)

WORKSPACE_ROOT = Path(__file__).parents[2]
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-binary-sensitivity.yaml"


def _small_plan() -> BinarySensitivityPlan:
    plan = load_binary_sensitivity_plan(PLAN_PATH)
    return plan.model_copy(update={"simulation_count": 1000})


def test_plan_matches_executed_design_and_discrete_effect_scale() -> None:
    plan = load_binary_sensitivity_plan(PLAN_PATH)

    assert plan.case_count == 24
    assert plan.model_runs_per_case == 1
    assert plan.policy_threshold == 0.6
    assert 2 / 24 < plan.minimum_interpretable_effect < 3 / 24


def test_seeded_report_is_deterministic_and_retains_historical_comparison(
    tmp_path: Path,
) -> None:
    plan = _small_plan()
    first = run_binary_sensitivity(plan, output_path=tmp_path / "first.json")
    second = run_binary_sensitivity(plan, output_path=tmp_path / "second.json")

    assert first.report_hash == second.report_hash
    assert first.model_runs_per_case == 1
    assert first.threshold_location == "strictly_between_two_and_three_net_cases"
    assert first.historical_design_status == ("retained_but_not_applicable_to_executed_design")
    assert first.scenarios[0].exact_mcnemar_rejection_rate <= 0.08


def test_positive_scenario_reports_the_chance_of_missing_the_effect(tmp_path: Path) -> None:
    report = run_binary_sensitivity(_small_plan(), output_path=tmp_path / "report.json")
    by_id = {item.scenario_id: item for item in report.scenarios}

    assert by_id["two-net-cases"].missed_minimum_effect_rate is None
    assert by_id["three-net-cases"].missed_minimum_effect_rate is not None
    assert by_id["six-net-cases"].missed_minimum_effect_rate is not None
    assert by_id["six-net-cases"].mean_observed_effect > 0.20


def test_plan_rejects_an_effect_inconsistent_with_joint_probabilities() -> None:
    plan = load_binary_sensitivity_plan(PLAN_PATH)
    payload = plan.model_dump(mode="python")
    payload["scenarios"][1]["expected_effect"] = 0.5

    with pytest.raises(ValueError, match="Expected effect"):
        BinarySensitivityPlan.model_validate(payload)
