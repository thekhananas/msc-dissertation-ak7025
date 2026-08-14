from pathlib import Path

from socratic_tutor.acquisition_study import (
    load_acquisition_study_plan,
    load_verified_development_ablation_matrix,
    publish_development_ablation_matrix,
    publish_development_policy_matrix,
    run_verified_development_ablation_matrix,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_ablation_matrix_reproduces_bounded_sources_and_publishes_compact_rows(
    tmp_path: Path,
) -> None:
    source_matrix, source_replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=2,
    )
    source_root = tmp_path / "source"
    source_manifest = publish_development_policy_matrix(
        source_matrix,
        replay_content_hash=source_replay_hash,
        output_root=source_root,
        run_id="acquisition-ablation-source-test",
        code_revision="4e14544",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    matrix, replay_hash = run_verified_development_ablation_matrix(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    output_root = tmp_path / "ablations"
    manifest = publish_development_ablation_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id="acquisition-ablation-matrix-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    retry = publish_development_ablation_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id="acquisition-ablation-matrix-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    loaded_manifest, loaded_matrix = load_verified_development_ablation_matrix(
        output_root / "development_ablation_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )

    assert retry == manifest == loaded_manifest
    assert loaded_matrix == matrix
    assert manifest.source_manifest_hash == source_manifest.manifest_hash
    assert matrix.row_count == 7 * 2 * 4
    assert manifest.case_prediction_count == matrix.row_count * 40
    assert manifest.selected_probe_count == matrix.row_count * 20
    assert matrix.bounded_source_reproduction_verified
    assert manifest.deterministic_replay_verified
    assert not manifest.canonical_claim_allowed
    assert (output_root / "development_ablation_matrix.jsonl").is_file()
    assert (output_root / "development_ablation_manifest.json").is_file()
