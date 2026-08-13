"""Checks for paired, deterministic acquisition-policy development runs."""

from __future__ import annotations

from pathlib import Path

from socratic_tutor.acquisition_study import (
    EvaluationEnvironmentId,
    PolicyId,
    load_acquisition_study_plan,
    publish_development_policy_matrix,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"
SPECIFICATION, ANALYSIS_PLAN = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)


def test_development_runner_pairs_and_publishes_every_policy_without_external_work(
    tmp_path: Path,
) -> None:
    matrix, replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS_PLAN,
        episodes_per_environment=1,
    )

    assert matrix.content_hash == replay_hash
    assert matrix.environment_count == len(EvaluationEnvironmentId)
    assert matrix.comparison_count == len(EvaluationEnvironmentId)
    assert matrix.policy_count == 7
    for comparison in matrix.comparisons:
        results = {result.policy_id: result for result in comparison.results}
        assert results[PolicyId.NEVER_PROBE].burden.selected_probe_count == 0
        assert results[PolicyId.ALWAYS_PROBE_BOUNDED].burden.selected_probe_count == 40
        assert all(
            result.burden.selected_probe_count == 20
            for policy_id, result in results.items()
            if policy_id not in {PolicyId.NEVER_PROBE, PolicyId.ALWAYS_PROBE_BOUNDED}
        )
        assert all(
            result.burden.external_workload.model_request_count == 0
            and result.burden.external_workload.sandbox_execution_count == 0
            for result in comparison.results
        )

    manifest = publish_development_policy_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=tmp_path,
        run_id="acquisition-development-test",
        code_revision="37cfef2",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    retry = publish_development_policy_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=tmp_path,
        run_id="acquisition-development-test",
        code_revision="37cfef2",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    assert manifest == retry
    assert manifest.deterministic_replay_verified
    assert manifest.comparison_count == len(EvaluationEnvironmentId)
    assert manifest.policy_result_count == len(EvaluationEnvironmentId) * 7
    assert manifest.case_prediction_count == len(EvaluationEnvironmentId) * 7 * 40
    assert (tmp_path / "development_comparisons.jsonl").is_file()
    assert (tmp_path / "development_policy_matrix_manifest.json").is_file()
