from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.acquisition_study.canonical_execution import (
    CanonicalExecutionError,
    DevelopmentEvidenceRequirement,
    load_canonical_execution_plan,
    verify_development_evidence,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import EvaluationEnvironmentId
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.hashing import canonical_sha256

ROOT = Path(__file__).parents[2]


def test_canonical_plan_fixes_inputs_workload_and_one_run_rule() -> None:
    plan = load_canonical_execution_plan(
        ROOT / "configs" / "acquisition-study" / "v1-canonical.yaml"
    )

    assert plan.environment_ids == tuple(EvaluationEnvironmentId)
    assert plan.policy_ids == (
        PolicyId.RELIABILITY_AWARE_BOUNDED,
        PolicyId.SEEDED_RANDOM_BOUNDED,
        PolicyId.UNCERTAINTY_ONLY_BOUNDED,
        PolicyId.PLUG_IN_EVSI_BOUNDED,
        PolicyId.NEVER_PROBE,
        PolicyId.ALWAYS_PROBE_BOUNDED,
        PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
    )
    assert plan.evaluation_episode_index_start == 50
    assert plan.evaluation_episode_index_stop_exclusive == 2050
    assert plan.expected_episode_count == 14_000
    assert plan.expected_public_row_count == 98_000
    assert plan.expected_case_prediction_count == 3_920_000
    assert plan.canonical_evaluation_run_limit == 1
    assert plan.exact_seeded_retry_required
    assert not plan.known_blocking_defect_ids


def test_development_evidence_requires_matching_content_not_only_a_claimed_hash(
    tmp_path: Path,
) -> None:
    content = {
        "schema_id": "example.development_report.v1",
        "result_status": "development_complete",
        "source_reproduced": True,
        "canonical_claim_allowed": False,
    }
    report_hash = canonical_sha256(content)
    valid_path = tmp_path / "valid.json"
    write_immutable_json(valid_path, {**content, "report_hash": report_hash})
    requirement = DevelopmentEvidenceRequirement(
        evidence_id="example_report",
        path="valid.json",
        schema_id="example.development_report.v1",
        content_hash_field="report_hash",
        expected_content_hash=report_hash,
        required_result_status="development_complete",
        required_true_fields=("source_reproduced",),
        required_false_fields=("canonical_claim_allowed",),
    )

    sealed = verify_development_evidence(tmp_path, requirement)
    assert sealed.content_hash == report_hash

    tampered_path = tmp_path / "tampered.json"
    write_immutable_json(
        tampered_path,
        {
            **content,
            "canonical_claim_allowed": True,
            "report_hash": report_hash,
        },
    )
    tampered_requirement = requirement.model_copy(update={"path": "tampered.json"})
    with pytest.raises(CanonicalExecutionError, match="self-hash failed"):
        verify_development_evidence(tmp_path, tampered_requirement)
