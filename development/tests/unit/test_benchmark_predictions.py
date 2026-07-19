"""Decision-safe benchmark prediction contract and Arrow-schema tests."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.benchmark.public import (
    BenchmarkCondition,
    ControlFactory,
    DecisionPredictionRecord,
    PairedConditionRun,
    PairedConditionRunner,
    PredictionRecordError,
    PredictionRecordFactory,
    PublicBenchmarkManifest,
    TrackerSecondaryPrediction,
    find_public_payload_violations,
    prediction_arrow_snapshot,
)
from socratic_tutor.contracts import Evidence, EvidenceCategory, TrackerState
from socratic_tutor.policies import choose_action
from socratic_tutor.tracking import update_tracker

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
COMMITTED_AT = datetime(2026, 8, 9, 9, 30, tzinfo=UTC)


def paired_fixture() -> tuple[PublicBenchmarkManifest, PairedConditionRun]:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    controls = ControlFactory(BENCHMARK_ROOT, public)
    aliasing = controls.probe_summary(case_id="dev-aliasing-001", passed=0, failed=2)
    none_falsy = controls.probe_summary(case_id="dev-none-falsy-001", passed=2, failed=0)
    case = next(item for item in public.cases if item.case_id == aliasing.case_id)
    runner = PairedConditionRunner(
        public,
        tracker_update=update_tracker,
        policy_decision=choose_action,
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
    )
    paired = runner.run_case(
        case_id=case.case_id,
        initial_tracker_state=TrackerState(
            concept=case.target_concept,
            mastery_probability=0.5,
            observations=0,
        ),
        public_evidence=Evidence(
            category=EvidenceCategory.CORRECT,
            confidence=0.8,
            rationale="The public explanation appears correct.",
        ),
        probe_evidence=aliasing,
        unrelated_evidence=controls.unrelated(
            case_id=case.case_id,
            evidence_by_case={aliasing.case_id: aliasing, none_falsy.case_id: none_falsy},
        ),
        corrupted_evidence=controls.corrupted(case_id=case.case_id, evidence=aliasing),
    )
    return public, paired


def prediction_records() -> tuple[DecisionPredictionRecord, ...]:
    public, paired = paired_fixture()
    return PredictionRecordFactory(public).build(
        paired,
        run_id="dev-prediction-run-001",
        sample_id="sample-001",
        model_route_id="recorded_fixture/authored-deterministic",
        public_record_hash=canonical_sha256({"record": "dev-aliasing-public"}),
        committed_at_utc=COMMITTED_AT,
    )


def test_factory_builds_complete_decision_safe_prediction_set() -> None:
    records = prediction_records()

    assert tuple(record.condition for record in records) == tuple(BenchmarkCondition)
    assert len({record.record_hash for record in records}) == 4
    assert len({record.initial_state_hash for record in records}) == 1
    assert len({record.public_record_hash for record in records}) == 1
    assert {record.tracker_version for record in records} == {"simple-v1"}
    assert {record.policy_version for record in records} == {"heuristic-v1"}
    assert {record.policy_propensity for record in records} == {1.0}
    assert {record.tracker_uncertainty for record in records} == {None}
    assert {record.tracker_uncertainty_method for record in records} == {"not_estimated"}
    assert records[0].additional_evidence_hash is None
    assert all(record.additional_evidence_hash is not None for record in records[1:])
    assert all(record.committed_at_utc == COMMITTED_AT for record in records)
    assert all(
        find_public_payload_violations(record.model_dump(mode="json")) == () for record in records
    )


def test_prediction_record_rejects_tampering_and_evaluator_fields() -> None:
    record = prediction_records()[0]
    changed_probability = record.model_dump(mode="python")
    changed_probability["tracker_mastery_probability"] = 0.1
    with pytest.raises(ValidationError, match="hash does not match"):
        DecisionPredictionRecord.model_validate(changed_probability)

    evaluator_injection = record.model_dump(mode="python")
    evaluator_injection["criterion_label"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DecisionPredictionRecord.model_validate(evaluator_injection)


def test_prediction_record_requires_utc_commit_time() -> None:
    record = prediction_records()[0]
    non_utc = record.model_dump(mode="python")
    non_utc["committed_at_utc"] = COMMITTED_AT.astimezone(timezone(timedelta(hours=1)))

    with pytest.raises(ValidationError, match="must use UTC"):
        DecisionPredictionRecord.model_validate(non_utc)


def test_secondary_predictions_require_honest_uncertainty_provenance() -> None:
    with pytest.raises(ValidationError, match="not_estimated"):
        TrackerSecondaryPrediction(
            uncertainty=None,
            uncertainty_method="posterior_entropy",
        )

    estimate = TrackerSecondaryPrediction(
        uncertainty=0.25,
        uncertainty_method="posterior_entropy-v1",
        misconception_probability=0.1,
    )
    assert estimate.uncertainty == 0.25


def test_factory_rejects_incomplete_condition_metadata() -> None:
    public, paired = paired_fixture()
    incomplete = {
        condition: TrackerSecondaryPrediction(uncertainty_method="not_estimated")
        for condition in tuple(BenchmarkCondition)[:-1]
    }

    with pytest.raises(PredictionRecordError, match="condition set"):
        PredictionRecordFactory(public).build(
            paired,
            run_id="run",
            sample_id="sample",
            model_route_id="route",
            public_record_hash="0" * 64,
            committed_at_utc=COMMITTED_AT,
            secondary_predictions=incomplete,
        )


def test_prediction_rows_match_exact_arrow_schema() -> None:
    records = prediction_records()
    snapshot = prediction_arrow_snapshot(records)

    assert snapshot.schema_id == "benchmark.condition_prediction.v1"
    assert snapshot.row_count == 4
    assert snapshot.conditions == tuple(condition.value for condition in BenchmarkCondition)
    assert snapshot.uncertainty_null_count == 4
    assert "condition: dictionary<values=string, indices=int8, ordered=0>" in (snapshot.field_types)
    assert "committed_at_utc: timestamp[us, tz=UTC]" in snapshot.field_types
    assert snapshot.nullable_fields == (
        "additional_evidence_hash",
        "tracker_uncertainty",
        "tracker_misconception_probability",
    )


def test_prediction_schema_cannot_represent_private_outcomes() -> None:
    serialized_schema = str(DecisionPredictionRecord.model_json_schema()).casefold()

    assert "criterion" not in serialized_schema
    assert "simulator" not in serialized_schema
    assert "reward" not in serialized_schema
    assert "true_mastery" not in serialized_schema
