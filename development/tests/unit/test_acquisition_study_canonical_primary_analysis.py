from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from socratic_tutor.acquisition_study import (
    PolicyId,
    PublicStreamExpectation,
    analyse_canonical_primary_errors,
    load_acquisition_study_plan,
    load_canonical_primary_source_plan,
    load_public_classification_errors,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    CanonicalPrimaryAnalysisError,
)
from socratic_tutor.acquisition_study.canonical_stream import project_canonical_episode
from socratic_tutor.acquisition_study.plan import EvaluationEnvironmentId
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


def test_source_plan_binds_the_completed_run_without_restricted_data() -> None:
    plan = load_canonical_primary_source_plan(
        ROOT / "configs" / "acquisition-study" / "v1-canonical-primary-analysis.yaml"
    )

    assert plan.canonical_code_revision == "63b0e523d0edabc3372dfc0c0df8ce02bcb74aeb"
    assert plan.environment_ids == tuple(EvaluationEnvironmentId)
    assert plan.policy_ids == POLICIES
    assert plan.expected_public_row_count == 98_000
    assert not plan.raw_policy_outcomes_inspected_before_freeze
    assert not plan.restricted_stream_access_allowed


def test_public_reader_enforces_pairing_and_order_on_development_fixture(
    tmp_path: Path,
) -> None:
    calibration = estimate_probe_reliability(SPECIFICATION)
    run = run_episode_policy_bundle(
        SPECIFICATION,
        ANALYSIS,
        calibration,
        environment_id=EvaluationEnvironmentId.MATCHED,
        episode_index=0,
    )
    _, rows = project_canonical_episode(run, ANALYSIS)
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path = tmp_path / "public.jsonl"
    path.write_bytes(content)
    expectation = PublicStreamExpectation(
        environment_ids=(EvaluationEnvironmentId.MATCHED,),
        policy_ids=POLICIES,
        episode_index_start=0,
        episodes_per_environment=1,
        candidates_per_episode=40,
        matched_probe_budget=20,
        environment_specification_hash=SPECIFICATION.specification_hash,
        analysis_plan_hash=ANALYSIS.plan_hash,
        calibration_hash=calibration.calibration_hash,
        public_stream_sha256=file_sha256(content),
    )

    loaded = load_public_classification_errors(path, expectation)
    assert loaded.row_count == 7
    assert loaded.errors[(EvaluationEnvironmentId.MATCHED, POLICIES[0])] == (
        rows[0].metric.final_classification_error,
    )

    reordered = (rows[1], rows[0], *rows[2:])
    reordered_content = b"".join(canonical_json_bytes(row) + b"\n" for row in reordered)
    path.write_bytes(reordered_content)
    with pytest.raises(CanonicalPrimaryAnalysisError, match="sealed order"):
        load_public_classification_errors(
            path,
            replace(expectation, public_stream_sha256=file_sha256(reordered_content)),
        )


def test_primary_rules_are_applied_to_paired_environment_errors() -> None:
    policy_error = {
        PolicyId.RELIABILITY_AWARE_BOUNDED: 0.10,
        PolicyId.SEEDED_RANDOM_BOUNDED: 0.30,
        PolicyId.UNCERTAINTY_ONLY_BOUNDED: 0.25,
        PolicyId.PLUG_IN_EVSI_BOUNDED: 0.20,
        PolicyId.NEVER_PROBE: 0.40,
        PolicyId.ALWAYS_PROBE_BOUNDED: 0.05,
        PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED: 0.00,
    }
    errors = {
        (environment_id, policy_id): (value,) * 4
        for environment_id in EvaluationEnvironmentId
        for policy_id, value in policy_error.items()
    }

    report = analyse_canonical_primary_errors(
        errors,
        analysis=ANALYSIS,
        execution_plan_hash="1" * 64,
        source_plan_hash="2" * 64,
        canonical_run_manifest_hash="3" * 64,
        canonical_stream_report_hash="4" * 64,
        public_stream_sha256="5" * 64,
        source_public_row_count=196,
    )

    assert report.all_simultaneous_interval_rules_met
    assert report.no_held_out_point_regression_against_plug_in_evsi
    assert report.robustness_rule_met
    assert all(row.interval_upper < 0.0 for row in report.held_out_primary_intervals)
    assert report.claim_assessments[0].status == "supported_within_declared_simulator"
    assert all(row.status == "prohibited_by_design" for row in report.claim_assessments[2:])
