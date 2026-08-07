from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.tracker_study import (
    load_tracker_study_analysis_specification,
    load_tracker_study_configuration,
    publish_development_matrix,
    publish_development_results,
    run_development_analysis,
    run_development_runtime_profile,
    run_development_sensitivity,
    simulate_verified_development_matrix,
)

ROOT = Path(__file__).parents[2]
CONFIGURATION = load_tracker_study_configuration(ROOT / "configs" / "tracker-study" / "v1.yaml")
SPECIFICATION = load_tracker_study_analysis_specification(
    ROOT / "configs" / "tracker-study" / "v1-analysis.yaml",
    CONFIGURATION,
)


def test_publication_reconciles_sources_and_retries_exactly(tmp_path: Path) -> None:
    simulation_root, analysis_root, sensitivity_root, runtime_root = _publish_sources(tmp_path)
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 11, 0, tzinfo=UTC)
    first = publish_development_results(
        analysis_root=analysis_root,
        sensitivity_root=sensitivity_root,
        runtime_root=runtime_root,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        publication_code_revision="79599b2",
        generated_at_utc=generated_at,
    )
    retry = publish_development_results(
        analysis_root=analysis_root,
        sensitivity_root=sensitivity_root,
        runtime_root=runtime_root,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=output_root,
        publication_code_revision="79599b2",
    )

    assert first == retry
    assert first.generated_at_utc == generated_at
    assert not first.canonical_claim_allowed
    assert not first.human_learning_claim_supported
    assert not first.tutoring_efficacy_claim_supported
    assert (output_root / "tracker_study_development.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "tracker_study_development.svg").read_text(encoding="utf-8")
    assert "When does bounded evidence updating help?" in svg
    assert "Development diagnostics only" in svg
    assert "not human" in svg
    condition_csv = (output_root / "condition_metrics.csv").read_text(encoding="utf-8")
    assert "brier_score" in condition_csv
    assert "bounded_channel_aware" in condition_csv
    sensitivity_csv = (output_root / "sensitivity.csv").read_text(encoding="utf-8")
    assert "is_configured_anchor" in sensitivity_csv
    runtime_csv = (output_root / "runtime.csv").read_text(encoding="utf-8")
    assert "median_microseconds_per_update" in runtime_csv
    primary_csv = (output_root / "primary_summary.csv").read_text(encoding="utf-8")
    assert "development_only_no_decision" in primary_csv
    assert simulation_root.is_dir()


def _publish_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    matrix, replay_hash = simulate_verified_development_matrix(
        CONFIGURATION,
        episodes_per_condition=1,
    )
    simulation_root = tmp_path / "simulation"
    publish_development_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=simulation_root,
        run_id="tracker-publication-source-test",
        code_revision="e58124e",
    )
    analysis_root = tmp_path / "analysis"
    run_development_analysis(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=analysis_root,
        run_id="tracker-publication-analysis-test",
        analysis_code_revision="7caac33",
    )
    sensitivity_root = tmp_path / "sensitivity"
    run_development_sensitivity(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        development_analysis_plan_path=analysis_root / "development_analysis_plan.json",
        development_analysis_report_path=analysis_root / "development_analysis_report.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=sensitivity_root,
        run_id="tracker-publication-sensitivity-test",
        sensitivity_code_revision="e774e9f",
    )
    runtime_root = tmp_path / "runtime"
    run_development_runtime_profile(
        simulation_manifest_path=simulation_root / "stress_matrix_manifest.json",
        development_analysis_plan_path=analysis_root / "development_analysis_plan.json",
        development_analysis_report_path=analysis_root / "development_analysis_report.json",
        configuration=CONFIGURATION,
        specification=SPECIFICATION,
        pixi_lock_path=ROOT / "pixi.lock",
        output_root=runtime_root,
        run_id="tracker-publication-runtime-test",
        runtime_code_revision="79599b2",
    )
    return simulation_root, analysis_root, sensitivity_root, runtime_root
