from pathlib import Path

from socratic_tutor.acquisition_study import (
    PolicyId,
    load_acquisition_study_plan,
    load_verified_development_negative_control,
    publish_development_negative_control,
    publish_development_policy_matrix,
    run_verified_development_negative_control,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_negative_control_reproduces_candidate_and_publishes_paired_rows(
    tmp_path: Path,
) -> None:
    source_matrix, source_replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=1,
    )
    source_root = tmp_path / "source"
    source_manifest = publish_development_policy_matrix(
        source_matrix,
        replay_content_hash=source_replay_hash,
        output_root=source_root,
        run_id="acquisition-negative-control-source-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    matrix, replay_hash = run_verified_development_negative_control(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    output_root = tmp_path / "negative-control"
    manifest = publish_development_negative_control(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id="acquisition-negative-control-test",
        code_revision="7654321",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    retry = publish_development_negative_control(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id="acquisition-negative-control-test",
        code_revision="7654321",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    loaded_manifest, loaded_matrix = load_verified_development_negative_control(
        output_root / "development_negative_control_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )

    assert retry == manifest == loaded_manifest
    assert loaded_matrix == matrix
    assert manifest.source_manifest_hash == source_manifest.manifest_hash
    assert matrix.row_count == 7
    assert all(row.candidate.policy_id is PolicyId.RELIABILITY_AWARE_BOUNDED for row in matrix.rows)
    assert all(
        row.shuffled_control.policy_id is PolicyId.SHUFFLED_RELIABILITY_BOUNDED
        for row in matrix.rows
    )
    assert manifest.case_prediction_count == 7 * 40 * 2
    assert manifest.selected_probe_count == 7 * 20 * 2
    assert manifest.deterministic_replay_verified
    assert manifest.candidate_source_reproduction_verified
    assert not manifest.canonical_claim_allowed
    assert (output_root / "development_negative_control.jsonl").is_file()
    assert (output_root / "development_negative_control_manifest.json").is_file()
