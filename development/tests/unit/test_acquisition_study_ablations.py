from pathlib import Path

from socratic_tutor.acquisition_study import (
    AblationPolicyCellId,
    EvaluationEnvironmentId,
    ProbeClassId,
    estimate_probe_reliability,
    load_acquisition_study_plan,
    run_episode_ablation_comparison,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)
CALIBRATION = estimate_probe_reliability(SPECIFICATION)


def test_ablation_cells_share_one_episode_budget_and_observed_evidence() -> None:
    comparison = run_episode_ablation_comparison(
        SPECIFICATION,
        ANALYSIS,
        CALIBRATION,
        environment_id=EvaluationEnvironmentId.INVERTED_EVIDENCE,
        episode_index=7,
    )
    retry = run_episode_ablation_comparison(
        SPECIFICATION,
        ANALYSIS,
        CALIBRATION,
        environment_id=EvaluationEnvironmentId.INVERTED_EVIDENCE,
        episode_index=7,
    )

    assert retry == comparison
    assert tuple(cell.cell_id for cell in comparison.cells) == tuple(AblationPolicyCellId)
    assert all(
        cell.result.burden.selected_probe_count
        == ANALYSIS.primary.exact_selected_probes_per_episode
        for cell in comparison.cells
    )
    assert all(
        tuple(row.available_case_count for row in cell.selection_by_probe_class) == (10, 10, 10, 10)
        for cell in comparison.cells
    )
    assert all(
        tuple(row.probe_class for row in cell.selection_by_probe_class) == tuple(ProbeClassId)
        for cell in comparison.cells
    )
    latent_outcomes = tuple(case.latent_success for case in comparison.cells[0].result.case_results)
    assert all(
        tuple(case.latent_success for case in cell.result.case_results) == latent_outcomes
        for cell in comparison.cells
    )
