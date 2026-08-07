from __future__ import annotations

import gc
from pathlib import Path

from socratic_tutor.tracker_study import (
    load_tracker_study_analysis_specification,
    load_tracker_study_configuration,
    publish_development_matrix,
    run_development_analysis,
    run_development_runtime_profile,
    simulate_verified_development_matrix,
)
from socratic_tutor.tracker_study.config import EvidenceCategory, TrackerId

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
SPECIFICATION = load_tracker_study_analysis_specification(
    ROOT / "configs" / "tracker-study" / "v1-analysis.yaml",
    CONFIGURATION,
)


def test_runtime_profile_uses_equal_fixed_workloads_and_restores_gc(tmp_path: Path) -> None:
    matrix, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=1,
    )
    simulation_root = tmp_path / "simulation"
    publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=simulation_root,
        run_id="tracker-runtime-source-test",
        code_revision="e58124e",
    )
    analysis_root = tmp_path / "analysis"
    run_development_analysis(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=analysis_root,
        run_id="tracker-runtime-analysis-test",
        analysis_code_revision="7caac33",
    )
    collection_was_enabled = gc.isenabled()
    output_root = tmp_path / "runtime"
    report = run_development_runtime_profile(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        development_analysis_plan_path=analysis_root / "development_analysis_plan.json",
        development_analysis_report_path=analysis_root / "development_analysis_report.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="tracker-development-runtime-test",
        runtime_code_revision="e774e9f",
    )

    expected_count = sum(
        observation.category is not EvidenceCategory.MISSING
        for episode in matrix.episodes
        for turn in episode.turns
        for observation in turn.observations
    )
    assert gc.isenabled() is collection_was_enabled
    assert report.measurement_count == len(TrackerId)
    assert report.non_missing_updates_per_tracker_repetition == expected_count
    assert report.measured_update_invocation_count == expected_count * len(TrackerId) * 20
    assert report.warmup_update_invocation_count == expected_count * len(TrackerId) * 3
    assert not report.exact_replay_expected
    assert not report.canonical_claim_allowed
    assert not report.human_learning_claim_supported
    assert not report.tutoring_efficacy_claim_supported
    assert tuple(row.tracker_id for row in report.measurements) == tuple(TrackerId)
    assert all(len(row.elapsed_nanoseconds) == 20 for row in report.measurements)
    assert all(row.median_nanoseconds_per_non_missing_update > 0 for row in report.measurements)
    assert report.platform.operating_system
    assert report.platform.machine
    assert report.platform.python_version
    assert (output_root / "development_runtime_plan.json").is_file()
    assert (output_root / "development_runtime_report.json").is_file()
