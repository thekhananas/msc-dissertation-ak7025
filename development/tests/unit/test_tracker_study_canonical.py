from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.tracker_study import (
    StudySplit,
    load_canonical_execution_plan,
    load_tracker_study_analysis_specification,
    load_tracker_study_configuration,
    publish_canonical_matrix,
    run_canonical_analysis,
    simulate_stress_matrix,
)
from socratic_tutor.tracker_study.analysis import TrackerStudyAnalysisError

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
SPECIFICATION = load_tracker_study_analysis_specification(
    ROOT / "configs" / "tracker-study" / "v1-analysis.yaml",
    CONFIGURATION,
)
REVISION = "a" * 40


def test_canonical_plan_fixes_inputs_size_output_and_retry_rule() -> None:
    plan = load_canonical_execution_plan(
        ROOT / "configs" / "tracker-study" / "v1-canonical-run.yaml"
    )

    assert plan.configuration_hash == CONFIGURATION.configuration_hash
    assert plan.analysis_specification_hash == SPECIFICATION.analysis_specification_hash
    assert plan.episodes_per_condition == 500
    assert plan.turns_per_episode == 40
    assert plan.canonical_test_run_limit == 1
    assert plan.exact_identical_retry_allowed
    assert not plan.parameter_selection_after_test_allowed
    assert plan.output_root == "artifacts/tracker-study/canonical-v1"


def test_canonical_matrix_and_analysis_are_immutable_and_claim_limited(
    tmp_path: Path,
) -> None:
    test_configuration = CONFIGURATION.model_copy(
        update={
            "experiment": CONFIGURATION.experiment.model_copy(
                update={"test_episodes_per_condition": 2}
            )
        }
    )
    matrix = simulate_stress_matrix(
        test_configuration,
        split=StudySplit.TEST,
    )
    replay = simulate_stress_matrix(
        test_configuration,
        split=StudySplit.TEST,
    )
    manifest = publish_canonical_matrix(
        matrix,
        replay_content_hash=replay.content_hash,
        output_root=tmp_path,
        run_id="canonical-unit-test",
        code_revision=REVISION,
        configuration=test_configuration,
        canonical_execution_plan_hash="b" * 64,
    )
    report = run_canonical_analysis(
        simulation_manifest_path=tmp_path / "canonical_stress_matrix_manifest.json",
        configuration=test_configuration,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=tmp_path,
        run_id="canonical-unit-test",
        analysis_code_revision=REVISION,
        canonical_execution_plan_hash="b" * 64,
    )
    retry = run_canonical_analysis(
        simulation_manifest_path=tmp_path / "canonical_stress_matrix_manifest.json",
        configuration=test_configuration,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=tmp_path,
        run_id="canonical-unit-test",
        analysis_code_revision=REVISION,
        canonical_execution_plan_hash="b" * 64,
    )

    assert report == retry
    assert manifest.split is StudySplit.TEST
    assert manifest.episodes_per_condition == 2
    assert report.primary_adverse_brier.episode_count == 2
    assert report.primary_decision_status in {
        "robust_under_declared_simulator",
        "not_robust_primary_interval",
        "not_robust_clean_guardrail",
        "not_robust_primary_and_clean_guardrail",
    }
    assert report.simulator_robustness_claim_allowed
    assert not report.human_learning_claim_supported
    assert not report.tutoring_efficacy_claim_supported
    assert not report.reduced_cognitive_offloading_claim_supported


def test_canonical_publisher_rejects_development_data(tmp_path: Path) -> None:
    matrix = simulate_stress_matrix(
        CONFIGURATION,
        split=StudySplit.DEVELOPMENT,
        episodes_per_condition=1,
    )

    with pytest.raises(ValueError, match="test trajectories only"):
        publish_canonical_matrix(
            matrix,
            replay_content_hash=matrix.content_hash,
            output_root=tmp_path,
            run_id="canonical-unit-test",
            code_revision=REVISION,
            configuration=CONFIGURATION,
            canonical_execution_plan_hash="b" * 64,
        )


def test_canonical_analysis_rejects_another_execution_plan(tmp_path: Path) -> None:
    test_configuration = CONFIGURATION.model_copy(
        update={
            "experiment": CONFIGURATION.experiment.model_copy(
                update={"test_episodes_per_condition": 1}
            )
        }
    )
    matrix = simulate_stress_matrix(test_configuration, split=StudySplit.TEST)
    publish_canonical_matrix(
        matrix,
        replay_content_hash=matrix.content_hash,
        output_root=tmp_path,
        run_id="canonical-plan-link-test",
        code_revision=REVISION,
        configuration=test_configuration,
        canonical_execution_plan_hash="b" * 64,
    )

    with pytest.raises(TrackerStudyAnalysisError, match="simulation and analysis plans differ"):
        run_canonical_analysis(
            simulation_manifest_path=tmp_path / "canonical_stress_matrix_manifest.json",
            configuration=test_configuration,
            specification=SPECIFICATION,
            pixi_lock_path=ROOT / "pixi.lock",
            output_root=tmp_path,
            run_id="canonical-plan-link-test",
            analysis_code_revision=REVISION,
            canonical_execution_plan_hash="c" * 64,
        )
