from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.canonical_budget_curve import (
    write_verified_canonical_budget_curve_stream,
)
from socratic_tutor.acquisition_study.canonical_budget_curve_analysis import (
    BudgetStreamExpectation,
    CanonicalBudgetCurveAnalysisError,
    analyse_budget_stream,
    load_verified_budget_stream,
)
from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    analyse_canonical_primary_errors,
)
from socratic_tutor.acquisition_study.canonical_stream import project_canonical_episode
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.acquisition_study.runner import run_episode_policy_bundle
from socratic_tutor.benchmark.hashing import canonical_json_bytes, file_sha256

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)
POLICIES = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
    PolicyId.NEVER_PROBE,
    PolicyId.ALWAYS_PROBE_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)


def test_canonical_budget_analysis_preserves_order_pairing_and_primary_result(
    tmp_path: Path,
) -> None:
    calibration = estimate_probe_reliability(SPECIFICATION)
    source_content = bytearray()
    errors: dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]] = {}
    for environment_id in EvaluationEnvironmentId:
        run = run_episode_policy_bundle(
            SPECIFICATION,
            ANALYSIS,
            calibration,
            environment_id=environment_id,
            episode_index=1,
        )
        _, public_rows = project_canonical_episode(run, ANALYSIS)
        for row in public_rows:
            source_content.extend(canonical_json_bytes(row) + b"\n")
        for result in run.comparison.results:
            errors[(environment_id, result.policy_id)] = (result.final_classification_error,)

    source_path = tmp_path / "canonical_public.jsonl"
    source_path.write_bytes(source_content)
    curve_path = tmp_path / "canonical_budget_curve.jsonl"
    curve, retry = write_verified_canonical_budget_curve_stream(
        SPECIFICATION,
        ANALYSIS,
        source_public_path=source_path,
        expected_source_hash=file_sha256(bytes(source_content)),
        episode_indices=(1,),
        output_path=curve_path,
    )
    assert curve == retry

    expectation = BudgetStreamExpectation(
        file_sha256=curve.budget_hash,
        episode_index_start=1,
        episodes_per_environment=1,
        candidates_per_episode=40,
    )
    loaded = load_verified_budget_stream(curve_path, expectation)
    primary = analyse_canonical_primary_errors(
        errors,
        analysis=ANALYSIS,
        execution_plan_hash="1" * 64,
        source_plan_hash="2" * 64,
        canonical_run_manifest_hash="3" * 64,
        canonical_stream_report_hash="4" * 64,
        public_stream_sha256="5" * 64,
        source_public_row_count=7 * len(POLICIES),
    )
    tables = analyse_budget_stream(
        loaded,
        primary=primary,
        episodes_per_environment=1,
        candidates_per_episode=40,
    )

    assert len(tables.environments) == 175
    assert len(tables.held_out) == 25
    assert len(tables.paired) == 20
    assert all(
        row.macro_mean_paired_difference == 0.0
        for row in tables.paired
        if row.budget_fraction in (0.0, 1.0)
    )

    lines = curve_path.read_bytes().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    reordered = b"".join(lines)
    curve_path.write_bytes(reordered)
    with pytest.raises(CanonicalBudgetCurveAnalysisError, match="fixed order"):
        load_verified_budget_stream(
            curve_path,
            replace(expectation, file_sha256=file_sha256(reordered)),
        )
