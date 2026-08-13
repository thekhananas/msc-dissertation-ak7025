from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean

import numpy as np

from socratic_tutor.acquisition_study import (
    PolicyId,
    load_acquisition_study_plan,
    publish_development_policy_matrix,
    run_development_primary_analysis,
    run_verified_development_policy_matrix,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    stratified_paired_bootstrap_distributions,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_primary_analysis_preserves_pairs_strata_and_claim_boundary(tmp_path: Path) -> None:
    matrix, replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=2,
    )
    source_root = tmp_path / "matrix"
    source_manifest = publish_development_policy_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=source_root,
        run_id="acquisition-primary-source-test",
        code_revision="4e14544",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    output_root = tmp_path / "analysis"
    report = run_development_primary_analysis(
        source_manifest_path=source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-primary-analysis-test",
        analysis_code_revision="1234567",
    )
    retry = run_development_primary_analysis(
        source_manifest_path=source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-primary-analysis-test",
        analysis_code_revision="1234567",
    )

    assert report == retry
    assert report.source_manifest_hash == source_manifest.manifest_hash
    assert report.comparator_policy_ids == ANALYSIS.policies.matched_budget_comparators
    assert len(report.held_out_primary_intervals) == 3
    assert all(
        row.confidence_level == ANALYSIS.primary.simultaneous_confidence_level
        for row in report.held_out_primary_intervals
    )
    assert all(row.episodes_per_environment == 2 for row in report.held_out_primary_intervals)
    assert not report.canonical_claim_allowed
    assert report.primary_decision_status == "not_evaluated_on_development_split"
    assert not report.human_learning_claim_supported
    assert not report.tutoring_efficacy_claim_supported

    candidate = PolicyId.RELIABILITY_AWARE_BOUNDED
    comparator = PolicyId.PLUG_IN_EVSI_BOUNDED
    held_out = set(ANALYSIS.primary.held_out_environments)
    direct_effects: list[float] = []
    for comparison in matrix.comparisons:
        if comparison.environment_id not in held_out:
            continue
        by_policy = {result.policy_id: result for result in comparison.results}
        direct_effects.append(
            by_policy[candidate].final_classification_error
            - by_policy[comparator].final_classification_error
        )
    expected_macro = fmean(direct_effects)
    plug_in = next(
        row for row in report.held_out_primary_intervals if row.comparator_policy_id is comparator
    )
    assert math.isclose(plug_in.macro_mean_paired_effect, expected_macro, abs_tol=1e-12)
    assert (output_root / "development_primary_analysis_plan.json").is_file()
    assert (output_root / "development_primary_analysis_report.json").is_file()


def test_primary_bootstrap_keeps_held_out_environments_as_separate_strata() -> None:
    environment_effects = (-0.30, -0.20, -0.10, 0.00, 0.10, 0.20)
    effects = {
        (environment_id, comparator): (effect,) * 4
        for environment_id, effect in zip(
            ANALYSIS.primary.held_out_environments,
            environment_effects,
            strict=True,
        )
        for comparator in ANALYSIS.policies.matched_budget_comparators
    }

    distributions = stratified_paired_bootstrap_distributions(effects, analysis=ANALYSIS)

    expected_macro = fmean(environment_effects)
    assert all(
        np.allclose(distribution, expected_macro, atol=1e-12, rtol=0.0)
        for distribution in distributions.values()
    )
