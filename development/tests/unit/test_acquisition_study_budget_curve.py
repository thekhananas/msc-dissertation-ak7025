from __future__ import annotations

from pathlib import Path

from socratic_tutor.acquisition_study import (
    EvaluationEnvironmentId,
    estimate_probe_reliability,
    load_acquisition_study_plan,
    run_episode_budget_curve,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_budget_curve_pairs_all_policies_and_budget_endpoints() -> None:
    calibration = estimate_probe_reliability(SPECIFICATION)

    curve = run_episode_budget_curve(
        SPECIFICATION,
        ANALYSIS,
        calibration,
        environment_id=EvaluationEnvironmentId.INVERTED_EVIDENCE,
        episode_index=4,
    )
    retry = run_episode_budget_curve(
        SPECIFICATION,
        ANALYSIS,
        calibration,
        environment_id=EvaluationEnvironmentId.INVERTED_EVIDENCE,
        episode_index=4,
    )

    assert curve == retry
    assert curve.content_hash == retry.content_hash
    assert len(curve.results) == 25
    assert {
        fraction: {
            row.selected_probe_count for row in curve.results if row.budget_fraction == fraction
        }
        for fraction in curve.budget_fractions
    } == {
        0.0: {0},
        0.25: {10},
        0.5: {20},
        0.75: {30},
        1.0: {40},
    }

    zero_results = tuple(
        row.result.case_results for row in curve.results if row.budget_fraction == 0.0
    )
    full_results = tuple(
        row.result.case_results for row in curve.results if row.budget_fraction == 1.0
    )
    assert len(set(zero_results)) == 1
    assert len(set(full_results)) == 1
    assert curve.access_scope == "simulator_restricted"
