from pathlib import Path

from socratic_tutor.acquisition_study import (
    load_acquisition_study_plan,
    publish_development_calibration_sensitivity,
    publish_development_policy_matrix,
    run_development_calibration_sensitivity_analysis,
    run_verified_development_calibration_sensitivity,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_calibration_sensitivity_analysis_repeats_primary_rules_without_claims(
    tmp_path: Path,
) -> None:
    source_matrix, source_replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=1,
    )
    source_root = tmp_path / "source"
    publish_development_policy_matrix(
        source_matrix,
        replay_content_hash=source_replay_hash,
        output_root=source_root,
        run_id="acquisition-sensitivity-analysis-source-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    matrix, replay_hash = run_verified_development_calibration_sensitivity(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    matrix_root = tmp_path / "sensitivity"
    matrix_manifest = publish_development_calibration_sensitivity(
        matrix,
        replay_content_hash=replay_hash,
        output_root=matrix_root,
        run_id="acquisition-sensitivity-analysis-matrix-test",
        code_revision="7654321",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    output_root = tmp_path / "analysis"
    report = run_development_calibration_sensitivity_analysis(
        source_manifest_path=(matrix_root / "development_calibration_sensitivity_manifest.json"),
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-sensitivity-analysis-test",
        analysis_code_revision="abcdef1",
    )
    retry = run_development_calibration_sensitivity_analysis(
        source_manifest_path=(matrix_root / "development_calibration_sensitivity_manifest.json"),
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-sensitivity-analysis-test",
        analysis_code_revision="abcdef1",
    )

    assert retry == report
    assert report.source_sensitivity_manifest_hash == matrix_manifest.manifest_hash
    assert len(report.calibration_results) == 20
    assert len(report.comparator_summaries) == 3
    assert all(len(row.comparator_results) == 3 for row in report.calibration_results)
    assert all(
        len(comparison.environment_effects) == 7
        for row in report.calibration_results
        for comparison in row.comparator_results
    )
    assert report.robustness_pattern_seed_count == sum(
        row.robustness_pattern_met for row in report.calibration_results
    )
    assert not report.canonical_claim_allowed
    assert not report.human_learning_claim_supported
    assert not report.tutoring_efficacy_claim_supported
    assert (output_root / "development_calibration_sensitivity_analysis_plan.json").is_file()
    assert (output_root / "development_calibration_sensitivity_analysis_report.json").is_file()
