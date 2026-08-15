from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean

from socratic_tutor.acquisition_study import (
    AblationId,
    AblationPolicyCellId,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
    publish_development_ablation_matrix,
    publish_development_policy_matrix,
    run_development_ablation_analysis,
    run_verified_development_ablation_matrix,
    run_verified_development_policy_matrix,
)

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def test_ablation_analysis_preserves_pairs_components_and_claim_boundary(
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
        run_id="acquisition-ablation-analysis-source-test",
        code_revision="4e14544",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    matrix, matrix_replay_hash = run_verified_development_ablation_matrix(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    matrix_root = tmp_path / "ablations"
    matrix_manifest = publish_development_ablation_matrix(
        matrix,
        replay_content_hash=matrix_replay_hash,
        output_root=matrix_root,
        run_id="acquisition-ablation-analysis-matrix-test",
        code_revision="89f4595",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    output_root = tmp_path / "analysis"
    report = run_development_ablation_analysis(
        source_manifest_path=matrix_root / "development_ablation_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-ablation-analysis-test",
        analysis_code_revision="1234567",
    )
    retry = run_development_ablation_analysis(
        source_manifest_path=matrix_root / "development_ablation_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        run_id="acquisition-ablation-analysis-test",
        analysis_code_revision="1234567",
    )

    assert retry == report
    assert report.source_ablation_manifest_hash == matrix_manifest.manifest_hash
    assert len(report.environment_summaries) == 7 * 4
    assert len(report.held_out_cell_summaries) == 4
    assert len(report.environment_effects) == 7 * 3
    assert len(report.held_out_effects) == 3
    assert len(report.dense_sparse_selection_effects) == 4
    assert report.bounded_source_reproduction_verified
    assert not report.canonical_claim_allowed
    assert not report.human_learning_claim_supported
    assert not report.tutoring_efficacy_claim_supported
    assert (output_root / "development_ablation_analysis_plan.json").is_file()
    assert (output_root / "development_ablation_analysis_report.json").is_file()

    held_out = set(ANALYSIS.primary.held_out_environments)
    direct_effects = tuple(
        row.metric.final_classification_error
        - next(
            candidate.metric.final_classification_error
            for candidate in matrix.rows
            if candidate.metric.environment_id is row.metric.environment_id
            and candidate.metric.episode_index == row.metric.episode_index
            and candidate.cell_id is AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED
        )
        for row in matrix.rows
        if row.metric.environment_id in held_out
        and row.cell_id is AblationPolicyCellId.FULL_CANDIDATE
    )
    reliability_effect = next(
        row
        for row in report.held_out_effects
        if row.ablation_id is AblationId.REPLACE_LOWER_QUANTILE_WITH_POSTERIOR_MEAN
    )
    assert math.isclose(
        reliability_effect.macro_mean_paired_difference,
        fmean(direct_effects),
        abs_tol=1e-12,
    )
    matched_summaries = tuple(
        row
        for row in report.environment_summaries
        if row.metrics.environment_id is EvaluationEnvironmentId.MATCHED
    )
    assert len(matched_summaries) == 4
