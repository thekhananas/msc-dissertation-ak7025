from __future__ import annotations

from pathlib import Path

from socratic_tutor.acquisition_study import (
    load_acquisition_study_plan,
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


def test_budget_curve_reproduces_primary_budget_and_publishes_compact_rows(
    tmp_path: Path,
) -> None:
    source_matrix, source_replay_hash = run_verified_development_policy_matrix(
        SPECIFICATION,
        ANALYSIS,
        episodes_per_environment=2,
    )
    source_root = tmp_path / "source"
    source_manifest = publish_development_policy_matrix(
        source_matrix,
        replay_content_hash=source_replay_hash,
        output_root=source_root,
        run_id="acquisition-budget-source-test",
        code_revision="1b7795d",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    matrix, replay_hash = run_verified_development_budget_curve(
        source_root / "development_policy_matrix_manifest.json",
        specification=SPECIFICATION,
        analysis=ANALYSIS,
    )
    output_root = tmp_path / "budget"
    manifest = publish_development_budget_curve(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id="acquisition-budget-curve-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )
    retry = publish_development_budget_curve(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id="acquisition-budget-curve-test",
        code_revision="1234567",
        pixi_lock_path=ROOT / "pixi.lock",
    )

    assert retry == manifest
    assert matrix.primary_budget_reproduction_verified
    assert matrix.endpoint_agreement_verified
    assert matrix.row_count == 7 * 2 * 5 * 5
    assert manifest.source_manifest_hash == source_manifest.manifest_hash
    assert manifest.row_count == matrix.row_count
    assert manifest.candidates_per_episode == 40
    assert manifest.case_prediction_count == matrix.row_count * 40
    assert manifest.deterministic_replay_verified
    assert not manifest.canonical_claim_allowed
    assert (output_root / "development_budget_curve.jsonl").is_file()
    assert (output_root / "development_budget_curve_manifest.json").is_file()

    selected_by_budget = {
        fraction: {
            row.selected_probe_count for row in matrix.rows if row.budget_fraction == fraction
        }
        for fraction in matrix.budget_fractions
    }
    assert selected_by_budget == {
        0.0: {0},
        0.25: {10},
        0.5: {20},
        0.75: {30},
        1.0: {40},
    }
