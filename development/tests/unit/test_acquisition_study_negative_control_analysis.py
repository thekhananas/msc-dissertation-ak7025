from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean

from socratic_tutor.acquisition_study import (
    load_acquisition_study_plan,
    publish_development_negative_control,
    publish_development_policy_matrix,
    run_development_negative_control_analysis,
    run_verified_development_negative_control,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_negative_control_analysis_preserves_pairing_and_claim_boundary(
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
        run_id="acquisition-negative-analysis-source-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    control_matrix, control_replay_hash = run_verified_development_negative_control(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    control_root = tmp_path / "control"
    control_manifest = publish_development_negative_control(
        control_matrix,
        replay_content_hash=control_replay_hash,
        output_root=control_root,
        run_id="acquisition-negative-analysis-matrix-test",
        code_revision="7654321",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    output_root = tmp_path / "analysis"
    report = run_development_negative_control_analysis(
        source_manifest_path=control_root / "development_negative_control_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-negative-analysis-test",
        analysis_code_revision="abcdef1",
    )
    retry = run_development_negative_control_analysis(
        source_manifest_path=control_root / "development_negative_control_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-negative-analysis-test",
        analysis_code_revision="abcdef1",
    )

    direct_held_out_differences = tuple(
        row.candidate.final_classification_error - row.shuffled_control.final_classification_error
        for row in control_matrix.rows
        if row.environment_id in ANALYSIS.primary.held_out_environments
    )
    assert retry == report
    assert report.source_negative_control_manifest_hash == control_manifest.manifest_hash
    assert len(report.environment_summaries) == 14
    assert len(report.held_out_policy_summaries) == 2
    assert len(report.environment_effects) == 7
    assert report.held_out_effect.paired_episode_count == 6
    assert math.isclose(
        report.held_out_effect.macro_mean_paired_difference,
        fmean(direct_held_out_differences),
        abs_tol=1e-12,
    )
    assert report.candidate_source_reproduction_verified
    assert not report.canonical_claim_allowed
    assert not report.human_learning_claim_supported
    assert not report.tutoring_efficacy_claim_supported
    assert (output_root / "development_negative_control_analysis_plan.json").is_file()
    assert (output_root / "development_negative_control_analysis_report.json").is_file()
