from pathlib import Path

from socratic_tutor.acquisition_study import (
    AcquisitionStudyPartition,
    CompactPolicyEpisodeRecord,
    acquisition_episode_indices,
    load_acquisition_study_plan,
    publish_development_policy_matrix,
    run_development_compact_parity,
    run_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_compact_stream_exactly_matches_source_projection_and_replay(tmp_path: Path) -> None:
    source_matrix = run_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=1,
    )
    source_root = tmp_path / "source"
    publish_development_policy_matrix(
        source_matrix,
        replay_content_hash=source_matrix.content_hash,
        output_root=source_root,
        run_id="acquisition-compact-source-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    output_root = tmp_path / "compact"
    report = run_development_compact_parity(
        source_manifest_path=source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-compact-parity-test",
        compact_code_revision="7654321",
    )

    rows = tuple(
        CompactPolicyEpisodeRecord.model_validate_json(line)
        for line in (output_root / report.compact_file).read_text(encoding="utf-8").splitlines()
    )
    assert report.exact_source_projection_parity_verified
    assert report.deterministic_replay_verified
    assert report.episode_count == 7
    assert report.row_count == 49
    assert report.case_prediction_count == 1_960
    assert len(rows) == report.row_count
    assert {row.partition for row in rows} == {AcquisitionStudyPartition.DEVELOPMENT}
    assert len({row.policy_result_hash for row in rows}) > 1
    assert not report.canonical_claim_allowed


def test_development_and_evaluation_episode_streams_do_not_overlap() -> None:
    development = acquisition_episode_indices(
        SPECIFICATION,
        partition=AcquisitionStudyPartition.DEVELOPMENT,
    )
    evaluation = acquisition_episode_indices(
        SPECIFICATION,
        partition=AcquisitionStudyPartition.EVALUATION,
    )

    assert development == tuple(range(50))
    assert evaluation == tuple(range(50, 2050))
    assert set(development).isdisjoint(evaluation)
