from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from socratic_tutor.tracker_study import (
    ConfiguredMasteryTracker,
    EvidenceObservation,
    SensitivityDimension,
    load_tracker_study_analysis_specification,
    load_tracker_study_configuration,
    publish_development_matrix,
    run_development_analysis,
    run_development_sensitivity,
    simulate_verified_development_matrix,
)
from socratic_tutor.tracker_study.config import (
    EvidenceCategory,
    EvidenceChannel,
    TrackerId,
)
from socratic_tutor.tracker_study.sensitivity import TrackerSensitivityError

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
SPECIFICATION = load_tracker_study_analysis_specification(
    ROOT / "configs" / "tracker-study" / "v1-analysis.yaml",
    CONFIGURATION,
)


def test_tracker_sensitivity_controls_have_neutral_limits() -> None:
    observation = EvidenceObservation(
        channel=EvidenceChannel.PUBLIC_ANSWER,
        category=EvidenceCategory.SUPPORTS_MASTERY,
        confidence=0.7,
    )
    no_trust = ConfiguredMasteryTracker(
        tracker_id=TrackerId.BOUNDED_CHANNEL_AWARE,
        configuration=CONFIGURATION,
        trust_weight_multiplier=0.0,
    )
    no_assumed_reliability = ConfiguredMasteryTracker(
        tracker_id=TrackerId.CHANNEL_AWARE,
        configuration=CONFIGURATION,
        assumed_channel_reliability_scale=0.0,
    )

    assert no_trust.update(0.35, observation).posterior_mastery_probability == 0.35
    assert no_assumed_reliability.update(0.35, observation).posterior_mastery_probability == 0.35


def test_development_sensitivity_reconciles_anchors_and_retries_exactly(
    tmp_path: Path,
) -> None:
    simulation_root, analysis_root = _publish_source_analysis(tmp_path)
    output_root = tmp_path / "sensitivity"
    first = _run_sensitivity(simulation_root, analysis_root, output_root)
    retry = _run_sensitivity(simulation_root, analysis_root, output_root)

    assert first == retry
    assert first.point_count == 12
    assert first.anchor_reconciliation.passed
    assert first.anchor_reconciliation.configured_anchor_count == 3
    assert first.anchor_reconciliation.maximum_absolute_difference <= 1e-12
    assert not first.parameter_selected
    assert first.configured_tracker_unchanged
    assert not first.canonical_claim_allowed
    assert not first.human_learning_claim_supported
    assert not first.tutoring_efficacy_claim_supported
    assert tuple(point.dimension for point in first.points) == tuple(
        dimension for dimension in SensitivityDimension for _ in range(4)
    )
    trust_points = tuple(
        point
        for point in first.points
        if point.dimension is SensitivityDimension.TRUST_WEIGHT_MULTIPLIER
    )
    assert not math.isclose(
        trust_points[0].candidate_minus_reference_adverse_brier,
        trust_points[-1].candidate_minus_reference_adverse_brier,
        abs_tol=1e-12,
    )
    assert (output_root / "development_sensitivity_plan.json").is_file()
    assert (output_root / "development_sensitivity_report.json").is_file()


def test_development_sensitivity_rejects_broken_analysis_lineage(tmp_path: Path) -> None:
    simulation_root, analysis_root = _publish_source_analysis(tmp_path)
    plan_path = analysis_root / "development_analysis_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["simulation_manifest_hash"] = "0" * 64
    from socratic_tutor.benchmark.hashing import canonical_sha256

    plan["plan_hash"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(TrackerSensitivityError, match="another simulation"):
        _run_sensitivity(simulation_root, analysis_root, tmp_path / "sensitivity")


def _publish_source_analysis(tmp_path: Path) -> tuple[Path, Path]:
    matrix, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=2,
    )
    simulation_root = tmp_path / "simulation"
    publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=simulation_root,
        run_id="tracker-sensitivity-source-test",
        code_revision="e58124e",
    )
    analysis_root = tmp_path / "analysis"
    run_development_analysis(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=analysis_root,
        run_id="tracker-sensitivity-analysis-test",
        analysis_code_revision="7caac33",
    )
    return simulation_root, analysis_root


def _run_sensitivity(
    simulation_root: Path,
    analysis_root: Path,
    output_root: Path,
):
    return run_development_sensitivity(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        development_analysis_plan_path=analysis_root / "development_analysis_plan.json",
        development_analysis_report_path=analysis_root / "development_analysis_report.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="tracker-development-sensitivity-test",
        sensitivity_code_revision="7caac33",
    )
