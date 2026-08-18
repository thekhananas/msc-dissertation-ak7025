from pathlib import Path

from socratic_tutor.acquisition_study import (
    load_acquisition_study_plan,
    publish_development_policy_matrix,
    run_development_policy_matrix,
    run_development_runtime_forecast,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"
SPECIFICATION, ANALYSIS_PLAN = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)


def test_runtime_forecast_measures_full_matrix_in_a_fresh_process(tmp_path: Path) -> None:
    matrix = run_development_policy_matrix(SPECIFICATION, ANALYSIS_PLAN)
    source_root = tmp_path / "source"
    manifest = publish_development_policy_matrix(
        matrix,
        replay_content_hash=matrix.content_hash,
        output_root=source_root,
        run_id="acquisition-runtime-source-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    output_root = tmp_path / "runtime"
    report = run_development_runtime_forecast(
        source_manifest_path=source_root / "development_policy_matrix_manifest.json",
        environment_path=ENVIRONMENTS,
        analysis_path=ANALYSIS,
        specification=SPECIFICATION,
        analysis=ANALYSIS_PLAN,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-runtime-test",
        runtime_code_revision="7654321",
        measurement_repetitions=1,
    )

    assert report.source_matrix_parity_verified
    assert report.measurements[0].matrix_content_hash == manifest.matrix_content_hash
    assert report.measurements[0].episodes_per_environment == 50
    assert report.measurements[0].comparison_count == 350
    assert report.measurements[0].policy_result_count == 2450
    assert report.measurements[0].case_prediction_count == 98_000
    assert report.linear_projected_canonical_elapsed_nanoseconds == (
        report.median_development_elapsed_nanoseconds * 40
    )
    assert report.guarded_projected_canonical_elapsed_nanoseconds == (
        report.linear_projected_canonical_elapsed_nanoseconds * 2
    )
    assert report.current_runner_feasible_for_canonical_run == (
        report.runtime_within_remaining_m7b_time and report.memory_within_planning_limit
    )
    assert not report.canonical_claim_allowed
    assert (output_root / "development_runtime_forecast_plan.json").is_file()
    assert (output_root / "development_runtime_forecast_report.json").is_file()
