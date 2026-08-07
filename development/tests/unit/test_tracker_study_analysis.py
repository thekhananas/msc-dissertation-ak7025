from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean

import pytest

from socratic_tutor.tracker_study import (
    calculate_binary_metrics,
    load_tracker_study_analysis_specification,
    load_tracker_study_configuration,
    publish_development_matrix,
    run_development_analysis,
    simulate_verified_development_matrix,
)
from socratic_tutor.tracker_study.analysis import (
    TrackerStudyAnalysisError,
    load_verified_stress_matrix,
)
from socratic_tutor.tracker_study.config import StressCondition, TrackerId

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
SPECIFICATION = load_tracker_study_analysis_specification(
    ROOT / "configs" / "tracker-study" / "v1-analysis.yaml",
    CONFIGURATION,
)


def test_binary_metrics_match_hand_calculation() -> None:
    metrics = calculate_binary_metrics(
        probabilities=(0.2, 0.8, 0.6, 0.4),
        outcomes=(False, True, False, True),
        priors=(0.5, 0.5, 0.5, 0.5),
        probability_floor=0.000001,
        classification_threshold=0.5,
        ece_bin_count=10,
    )

    assert math.isclose(metrics.brier_score, 0.20, abs_tol=1e-12)
    assert math.isclose(
        metrics.negative_log_likelihood,
        -0.5 * (math.log(0.8) + math.log(0.4)),
        abs_tol=1e-12,
    )
    assert math.isclose(metrics.expected_calibration_error, 0.4, abs_tol=1e-12)
    assert metrics.classification_error == 0.5
    assert math.isclose(metrics.mean_absolute_posterior_update, 0.2, abs_tol=1e-12)


def test_development_analysis_uses_episode_level_pairs_and_is_retry_safe(
    tmp_path: Path,
) -> None:
    matrix, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=3,
    )
    simulation_root = tmp_path / "simulation"
    manifest = publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=simulation_root,
        run_id="tracker-analysis-source-test",
        code_revision="e58124e",
    )
    output_root = tmp_path / "analysis"
    first = run_development_analysis(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="tracker-development-analysis-test",
        analysis_code_revision="9a65a5c",
    )
    retry = run_development_analysis(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="tracker-development-analysis-test",
        analysis_code_revision="9a65a5c",
    )

    assert first == retry
    assert first.analysis_plan_hash
    assert first.condition_metric_count == len(StressCondition) * len(TrackerId)
    assert first.primary_adverse_brier.episode_count == 3
    assert first.clean_brier_guardrail.episode_count == 3
    assert first.primary_adverse_brier.test_repetitions == 100000
    assert first.primary_decision_status == "not_evaluated_on_development_split"
    assert not first.canonical_claim_allowed
    assert not first.human_learning_claim_supported
    assert not first.tutoring_efficacy_claim_supported
    assert manifest.trajectory_content_hash == replay_hash
    rows = {(row.condition, row.tracker_id): row for row in first.condition_metrics}
    expected_primary = fmean(
        rows[(condition, TrackerId.BOUNDED_CHANNEL_AWARE)].metrics.brier_score
        - rows[(condition, TrackerId.CHANNEL_AWARE)].metrics.brier_score
        for condition in SPECIFICATION.primary.adverse_conditions
    )
    expected_clean = (
        rows[(StressCondition.CLEAN, TrackerId.BOUNDED_CHANNEL_AWARE)].metrics.brier_score
        - rows[(StressCondition.CLEAN, TrackerId.CHANNEL_AWARE)].metrics.brier_score
    )
    assert math.isclose(
        first.primary_adverse_brier.mean_effect,
        expected_primary,
        abs_tol=1e-12,
    )
    assert math.isclose(
        first.clean_brier_guardrail.mean_effect,
        expected_clean,
        abs_tol=1e-12,
    )
    assert (output_root / "development_analysis_plan.json").is_file()
    assert (output_root / "development_analysis_report.json").is_file()


def test_verified_loader_rejects_changed_trajectory_bytes(tmp_path: Path) -> None:
    matrix, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=1,
    )
    publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=tmp_path,
        run_id="tracker-tamper-source-test",
        code_revision="e58124e",
    )
    trajectory_path = tmp_path / "trajectories.jsonl"
    trajectory_path.write_bytes(trajectory_path.read_bytes() + b"\n")

    with pytest.raises(TrackerStudyAnalysisError, match="file hash"):
        load_verified_stress_matrix(
            tmp_path / "stress_matrix_manifest.json",
            CONFIGURATION,
        )
