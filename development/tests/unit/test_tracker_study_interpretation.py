from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pytest import MonkeyPatch

from socratic_tutor.tracker_study import (
    CanonicalExecutionPlan,
    StudySplit,
    load_canonical_execution_plan,
    load_tracker_study_analysis_specification,
    load_tracker_study_configuration,
    publish_canonical_matrix,
    publish_canonical_results,
    publish_development_matrix,
    run_canonical_analysis,
    run_canonical_interpretation,
    run_development_analysis,
    run_development_runtime_profile,
    run_development_sensitivity,
    simulate_stress_matrix,
    simulate_verified_development_matrix,
)
from socratic_tutor.tracker_study import interpretation as interpretation_module

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
SPECIFICATION = load_tracker_study_analysis_specification(
    ROOT / "configs" / "tracker-study" / "v1-analysis.yaml",
    CONFIGURATION,
)
EXECUTION_PLAN_PATH = ROOT / "configs" / "tracker-study" / "v1-canonical-run.yaml"
REVISION = "a" * 40


def test_interpretation_reconciles_sources_and_limits_claims(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    execution = load_canonical_execution_plan(EXECUTION_PLAN_PATH)
    test_execution = execution.model_copy(update={"episodes_per_condition": 2})

    def load_test_execution(_: Path) -> CanonicalExecutionPlan:
        return test_execution

    monkeypatch.setattr(
        interpretation_module,
        "load_canonical_execution_plan",
        load_test_execution,
    )
    canonical_root, sensitivity_root, runtime_root = _publish_sources(
        tmp_path,
        execution_plan_hash=execution.plan_hash,
    )
    output_root = tmp_path / "interpretation"
    completed_at = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
    first = run_canonical_interpretation(
        execution_plan_path=EXECUTION_PLAN_PATH,
        canonical_root=canonical_root,
        sensitivity_root=sensitivity_root,
        runtime_root=runtime_root,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        interpretation_code_revision="baf40df",
        created_at_utc=completed_at,
    )
    retry = run_canonical_interpretation(
        execution_plan_path=EXECUTION_PLAN_PATH,
        canonical_root=canonical_root,
        sensitivity_root=sensitivity_root,
        runtime_root=runtime_root,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        interpretation_code_revision="baf40df",
    )

    assert first == retry
    assert first.canonical_episode_count_per_condition == 2
    assert "prespecified simulator rule" in first.plain_result
    assert "clean-evidence Brier error" in first.trade_off
    assert first.simulator_robustness_claim_allowed
    assert not first.human_learning_claim_allowed
    assert not first.tutoring_efficacy_claim_allowed
    assert not first.reduced_cognitive_offloading_claim_allowed
    assert not first.cognitive_bandwidth_validity_claim_allowed
    assert first.learned_trust_extension_status == (
        "omitted_to_protect_time_box_and_avoid_post_test_model_selection"
    )

    publication_root = tmp_path / "canonical-publication"
    publication_time = datetime(2026, 9, 3, 12, 30, tzinfo=UTC)
    publication = publish_canonical_results(
        canonical_root=canonical_root,
        interpretation_root=output_root,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=publication_root,
        publication_code_revision="baf40df",
        generated_at_utc=publication_time,
    )
    publication_retry = publish_canonical_results(
        canonical_root=canonical_root,
        interpretation_root=output_root,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=publication_root,
        publication_code_revision="baf40df",
    )

    assert publication == publication_retry
    assert publication.generated_at_utc == publication_time
    assert not publication.human_learning_claim_supported
    assert not publication.tutoring_efficacy_claim_supported
    assert (publication_root / "tracker_study_canonical.pdf").read_bytes().startswith(b"%PDF")
    svg = (publication_root / "tracker_study_canonical.svg").read_text(encoding="utf-8")
    assert "Held-out simulation" in svg
    assert "not evidence of student learning" in svg
    claims = (publication_root / "canonical_claim_boundaries.csv").read_text(encoding="utf-8")
    assert "human_learning_improvement,False" in claims


def _publish_sources(
    root: Path,
    *,
    execution_plan_hash: str,
) -> tuple[Path, Path, Path]:
    development_matrix, development_replay = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=1,
    )
    development_root = root / "development"
    publish_development_matrix(
        development_matrix,
        replay_content_hash=development_replay,
        output_root=development_root,
        run_id="tracker-interpretation-development-test",
        code_revision="e58124e",
    )
    development_analysis_root = root / "development-analysis"
    run_development_analysis(
        simulation_manifest_path=development_root / "stress_matrix_manifest.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=development_analysis_root,
        run_id="tracker-interpretation-analysis-test",
        analysis_code_revision="7caac33",
    )
    sensitivity_root = root / "sensitivity"
    run_development_sensitivity(
        simulation_manifest_path=development_root / "stress_matrix_manifest.json",
        development_analysis_plan_path=(
            development_analysis_root / "development_analysis_plan.json"
        ),
        development_analysis_report_path=(
            development_analysis_root / "development_analysis_report.json"
        ),
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=sensitivity_root,
        run_id="tracker-interpretation-sensitivity-test",
        sensitivity_code_revision="e774e9f",
    )
    runtime_root = root / "runtime"
    run_development_runtime_profile(
        simulation_manifest_path=development_root / "stress_matrix_manifest.json",
        development_analysis_plan_path=(
            development_analysis_root / "development_analysis_plan.json"
        ),
        development_analysis_report_path=(
            development_analysis_root / "development_analysis_report.json"
        ),
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=runtime_root,
        run_id="tracker-interpretation-runtime-test",
        runtime_code_revision="79599b2",
    )

    test_configuration = CONFIGURATION.model_copy(
        update={
            "experiment": CONFIGURATION.experiment.model_copy(
                update={"test_episodes_per_condition": 2}
            )
        }
    )
    canonical_matrix = simulate_stress_matrix(test_configuration, split=StudySplit.TEST)
    canonical_root = root / "canonical"
    publish_canonical_matrix(
        canonical_matrix,
        replay_content_hash=canonical_matrix.content_hash,
        output_root=canonical_root,
        run_id="tracker-interpretation-canonical-test",
        code_revision=REVISION,
        configuration=test_configuration,
        canonical_execution_plan_hash=execution_plan_hash,
    )
    run_canonical_analysis(
        simulation_manifest_path=canonical_root / "canonical_stress_matrix_manifest.json",
        configuration=test_configuration,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=canonical_root,
        run_id="tracker-interpretation-canonical-test",
        analysis_code_revision=REVISION,
        canonical_execution_plan_hash=execution_plan_hash,
    )
    return canonical_root, sensitivity_root, runtime_root
