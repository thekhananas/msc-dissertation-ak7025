from __future__ import annotations

import math
from pathlib import Path

from socratic_tutor.acquisition_study import (
    PolicyId,
    analyse_development_budget_curve,
    load_acquisition_study_plan,
    load_verified_development_budget_curve,
    publish_development_budget_curve,
    publish_development_policy_matrix,
    run_verified_development_budget_curve,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_budget_analysis_preserves_pairing_and_descriptive_claim_boundary(
    tmp_path: Path,
) -> None:
    source_matrix, source_replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=2,
    )
    source_root = tmp_path / "source"
    publish_development_policy_matrix(
        source_matrix,
        replay_content_hash=source_replay_hash,
        output_root=source_root,
        run_id="acquisition-budget-analysis-source-test",
        code_revision="4e14544",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    curve, curve_replay_hash = run_verified_development_budget_curve(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    curve_root = tmp_path / "curve"
    curve_manifest = publish_development_budget_curve(
        curve,
        replay_content_hash=curve_replay_hash,
        output_root=curve_root,
        run_id="acquisition-budget-analysis-curve-test",
        code_revision="12028af",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    loaded_manifest, loaded_curve = load_verified_development_budget_curve(
        curve_root / "development_budget_curve_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )

    report = analyse_development_budget_curve(
        loaded_curve,
        analysis=ANALYSIS,
        source_manifest=loaded_manifest,
    )
    retry = analyse_development_budget_curve(
        loaded_curve,
        analysis=ANALYSIS,
        source_manifest=loaded_manifest,
    )

    assert report == retry
    assert report.source_budget_curve_manifest_hash == curve_manifest.manifest_hash
    assert len(report.environment_summaries) == 7 * 5 * 5
    assert len(report.held_out_macro_summaries) == 5 * 5
    assert len(report.held_out_paired_differences) == 5 * 4
    assert report.analysis_status == "descriptive_not_confirmatory"
    assert not report.canonical_claim_allowed
    assert not report.human_learning_claim_supported

    endpoint_differences = tuple(
        row for row in report.held_out_paired_differences if row.budget_fraction in (0.0, 1.0)
    )
    assert all(math.isclose(row.macro_mean_paired_difference, 0.0) for row in endpoint_differences)
    assert all(row.tied_episode_count == 12 for row in endpoint_differences)

    half_budget_candidate = next(
        row
        for row in report.held_out_macro_summaries
        if row.budget_fraction == 0.5 and row.policy_id is PolicyId.RELIABILITY_AWARE_BOUNDED
    )
    half_budget_random = next(
        row
        for row in report.held_out_macro_summaries
        if row.budget_fraction == 0.5 and row.policy_id is PolicyId.SEEDED_RANDOM_BOUNDED
    )
    half_budget_difference = next(
        row
        for row in report.held_out_paired_differences
        if row.budget_fraction == 0.5 and row.reference_policy_id is PolicyId.SEEDED_RANDOM_BOUNDED
    )
    assert math.isclose(
        half_budget_difference.macro_mean_paired_difference,
        half_budget_candidate.macro_mean_classification_error
        - half_budget_random.macro_mean_classification_error,
        abs_tol=1e-12,
    )
