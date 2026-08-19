from pathlib import Path

import pytest

import socratic_tutor.acquisition_study.canonical_stream as canonical_stream_module
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.canonical_stream import (
    CanonicalPublicPolicyMetric,
    CanonicalRestrictedEpisode,
    canonical_stream_schema_hashes,
    project_canonical_episode,
    write_verified_canonical_streams,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.acquisition_study.runner import run_episode_policy_bundle
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    file_sha256,
    model_content_hash,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def _one_development_fixture_episode(
    _specification: AcquisitionEnvironmentSpecification,
) -> tuple[int, ...]:
    return (1,)


def test_canonical_projection_separates_public_metrics_from_simulator_truth() -> None:
    run = run_episode_policy_bundle(
        SPECIFICATION,
        ANALYSIS,
        estimate_probe_reliability(SPECIFICATION),
        environment_id=EvaluationEnvironmentId.MATCHED,
        episode_index=0,
    )

    restricted, public = project_canonical_episode(run, ANALYSIS)
    public_bytes = b"".join(canonical_json_bytes(record) for record in public)
    restricted_bytes = canonical_json_bytes(restricted)

    assert len(public) == 7
    assert [record.metric.selected_probe_count for record in public] == [
        20,
        20,
        20,
        20,
        0,
        40,
        20,
    ]
    assert all(record.metric.episode_index == 0 for record in public)
    assert all(record.access_scope == "public_aggregate" for record in public)
    assert b"latent_success" not in public_bytes
    assert b"effective_sensitivity" not in public_bytes
    assert b"privileged_episode" not in public_bytes
    assert b"latent_success" in restricted_bytes
    assert b"effective_sensitivity" in restricted_bytes
    assert restricted.comparison_hash == model_content_hash(run.comparison)
    assert tuple(decision.policy_id for decision in restricted.decisions) == tuple(
        result.policy_id for result in run.comparison.results
    )
    assert all(len(schema_hash) == 64 for schema_hash in canonical_stream_schema_hashes())


def test_canonical_writer_streams_all_environments_and_verifies_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        canonical_stream_module,
        "_evaluation_episode_indices",
        _one_development_fixture_episode,
    )

    report = write_verified_canonical_streams(
        SPECIFICATION,
        ANALYSIS,
        output_root=tmp_path,
    )
    public_path = tmp_path / report.public_file
    restricted_path = tmp_path / report.restricted_file
    public_lines = public_path.read_text(encoding="utf-8").splitlines()
    restricted_lines = restricted_path.read_text(encoding="utf-8").splitlines()

    CanonicalPublicPolicyMetric.model_validate_json(public_lines[0])
    CanonicalRestrictedEpisode.model_validate_json(restricted_lines[0])
    assert report.deterministic_replay_verified
    assert report.episodes_per_environment == 1
    assert report.episode_count == 7
    assert report.public_row_count == 49
    assert report.restricted_row_count == 7
    assert report.case_prediction_count == 1_960
    assert report.selected_probe_count == 980
    assert report.public_file_sha256 == file_sha256(public_path.read_bytes())
    assert report.restricted_file_sha256 == file_sha256(restricted_path.read_bytes())
