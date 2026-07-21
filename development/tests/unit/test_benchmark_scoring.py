"""Criterion outcome and hand-calculated scoring-layer tests."""

import math
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.evaluator import (
    BaselineFitRole,
    BaselineResult,
    BenchmarkScorer,
    CalibrationDecision,
    CalibrationStatus,
    CriterionChannelRequestBuilder,
    CriterionExecutionStatus,
    CriterionMissingReason,
    CriterionRecord,
    CriterionRecordConflictError,
    CriterionRecordError,
    CriterionRecordFactory,
    CriterionRecordIndex,
    PrimaryMetric,
    RepeatEstimator,
    RepeatLossName,
    ScoringBaseline,
    ScoringError,
    brier_loss,
    classification_error,
    create_calibration_decision,
    create_constant_baseline_result,
    create_normalized_criterion_execution,
    create_probe_only_baseline_result,
    fit_constant_prevalence,
    repeat_comparisons,
)
from socratic_tutor.benchmark.evaluator.gate import VerifiedCriterionAccess
from socratic_tutor.benchmark.generation import create_student_generation_request
from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.benchmark.public import ConditionCommitStore, ControlFactory
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    create_recorded_response,
)
from socratic_tutor.tracking import update_tracker
from tests.unit.test_benchmark_criterion_gate import (
    BENCHMARK_ROOT,
    complete_gate_fixture,
    evaluator_manifest_fixture,
    generation_spec_fixture,
)
from tests.unit.test_benchmark_global_seal import benchmark_key
from tests.unit.test_benchmark_predictions import COMMITTED_AT, paired_fixture


def _marker(name: str) -> str:
    return canonical_sha256({"fixture": name})


def _assert_close(
    actual: tuple[float | None, ...],
    expected: tuple[float, ...],
) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value is not None
        assert math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12)


def _criterion_fixture(
    tmp_path: Path,
    *,
    status: CriterionExecutionStatus = CriterionExecutionStatus.COMPLETED,
    passed: int = 0,
    failed: int = 2,
) -> tuple[CriterionRecord, ConditionCommitStore, VerifiedCriterionAccess]:
    gate, condition_store = complete_gate_fixture(tmp_path)
    access = gate.open(
        gate.issue(benchmark_key("sample-001")),
        manifest_loader=evaluator_manifest_fixture,
    )
    request = generation_spec_fixture()
    criterion_request = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access).build_criterion(
        case_id=access.sample_key.case_id,
        spec=request,
    )
    requested_at = access.global_sealed_at_utc + timedelta(seconds=1)
    response = create_recorded_response(
        request=criterion_request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response="def held_out_solution(value):\n    return value\n",
        captured_at_utc=requested_at,
    )
    criterion_spec = next(
        item for item in access.manifest.criteria if item.case_id == access.sample_key.case_id
    )
    attempted = status not in {
        CriterionExecutionStatus.PROVIDER_UNAVAILABLE,
        CriterionExecutionStatus.SANDBOX_UNAVAILABLE,
    }
    execution = create_normalized_criterion_execution(
        status=status,
        test_bundle_sha256=criterion_spec.test_bundle_sha256,
        requested_at_utc=requested_at,
        operation_id="sandbox-op-001" if attempted else None,
        passed=passed if status is CriterionExecutionStatus.COMPLETED else 0,
        failed=failed if status is CriterionExecutionStatus.COMPLETED else 0,
        exit_code=(0 if failed == 0 else 1)
        if status is CriterionExecutionStatus.COMPLETED
        else None,
        timed_out=status is CriterionExecutionStatus.SANDBOX_TIMEOUT,
        resource_limited=status is CriterionExecutionStatus.SANDBOX_RESOURCE_LIMIT,
        stdout_sha256=_marker("stdout") if attempted else None,
        stderr_sha256=_marker("stderr") if attempted else None,
        executed_at_utc=requested_at + timedelta(seconds=1) if attempted else None,
    )
    criterion_response = (
        None if status is CriterionExecutionStatus.PROVIDER_UNAVAILABLE else response
    )
    record = CriterionRecordFactory().build(
        access,
        request=criterion_request,
        response=criterion_response,
        execution=execution,
        revealed_at_utc=requested_at + timedelta(seconds=2),
    )
    return record, condition_store, access


def _calibration(
    status: CalibrationStatus,
    *,
    tracker_version: str = "simple-v1",
) -> CalibrationDecision:
    return create_calibration_decision(
        status=status,
        policy_threshold=0.6,
        calibration_task_family_hash=_marker("calibration-families"),
        calibration_data_hash=_marker("calibration-data"),
        tracker_version=tracker_version,
        diagnostics_hash=_marker("calibration-diagnostics"),
        decision_owner="researcher",
        decided_at_utc=COMMITTED_AT - timedelta(days=1),
    )


def _baselines(
    record: CriterionRecord,
    calibration: CalibrationDecision,
) -> tuple[BaselineResult, BaselineResult]:
    fit = fit_constant_prevalence(
        (True, True, False, False),
        fit_role=BaselineFitRole.CALIBRATION,
        fit_split_hash=_marker("calibration-split"),
        baseline_version="constant-prevalence-v1",
        fitted_at_utc=COMMITTED_AT - timedelta(days=2),
    )
    public, paired = paired_fixture()
    controls = ControlFactory(BENCHMARK_ROOT, public)
    probe = controls.probe_summary(case_id=record.key.case_id, passed=0, failed=2)
    created_at = record.revealed_at_utc + timedelta(seconds=1)
    return (
        create_constant_baseline_result(
            record.key,
            fit=fit,
            calibration=calibration,
            created_at_utc=created_at,
        ),
        create_probe_only_baseline_result(
            record.key,
            fit_role=BaselineFitRole.CALIBRATION,
            fit_split_hash=_marker("calibration-split"),
            baseline_version="probe-only-simple-v1",
            initial_tracker_state=paired.initial_tracker_state,
            probe_evidence=probe,
            tracker_update=update_tracker,
            calibration=calibration,
            created_at_utc=created_at,
        ),
    )


criterion_fixture = _criterion_fixture
calibration_fixture = _calibration
baselines_fixture = _baselines


def test_completed_criterion_record_derives_outcome_and_is_exactly_once(
    tmp_path: Path,
) -> None:
    record, _, access = _criterion_fixture(tmp_path)

    assert record.demonstrated_performance is False
    assert record.missing_reason is None
    assert record.execution.passed == 0
    assert record.execution.failed == 2
    assert record.revealed_at_utc > access.global_sealed_at_utc
    assert record.revealed_at_utc > access.condition_sealed_at_utc

    index = CriterionRecordIndex()
    assert index.add(record) == record
    assert index.add(record) == record
    assert index.exists(record.key)

    request = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access).build_criterion(
        case_id=access.sample_key.case_id,
        spec=generation_spec_fixture(),
    )
    successful_execution = create_normalized_criterion_execution(
        status=CriterionExecutionStatus.COMPLETED,
        test_bundle_sha256=record.test_bundle_sha256,
        requested_at_utc=record.execution.requested_at_utc,
        operation_id="sandbox-op-002",
        passed=2,
        failed=0,
        exit_code=0,
        stdout_sha256=_marker("successful-stdout"),
        stderr_sha256=_marker("successful-stderr"),
        executed_at_utc=record.execution.requested_at_utc + timedelta(seconds=1),
    )
    replacement_response = create_recorded_response(
        request=request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response="def held_out_solution(value):\n    return value\n",
        captured_at_utc=record.execution.requested_at_utc,
    )
    different = CriterionRecordFactory().build(
        access,
        request=request,
        response=replacement_response,
        execution=successful_execution,
        revealed_at_utc=record.revealed_at_utc,
    )
    with pytest.raises(CriterionRecordConflictError, match="another criterion"):
        index.add(different)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            CriterionExecutionStatus.PROVIDER_UNAVAILABLE,
            CriterionMissingReason.PROVIDER_UNAVAILABLE,
        ),
        (
            CriterionExecutionStatus.SANDBOX_UNAVAILABLE,
            CriterionMissingReason.SANDBOX_UNAVAILABLE,
        ),
    ],
)
def test_unavailable_criterion_is_missing_not_failed(
    tmp_path: Path,
    status: CriterionExecutionStatus,
    reason: CriterionMissingReason,
) -> None:
    record, _, _ = _criterion_fixture(tmp_path, status=status)

    assert record.demonstrated_performance is None
    assert record.missing_reason is reason


def test_criterion_request_must_follow_both_decision_seals(tmp_path: Path) -> None:
    gate, _ = complete_gate_fixture(tmp_path)
    access = gate.open(
        gate.issue(benchmark_key("sample-001")),
        manifest_loader=evaluator_manifest_fixture,
    )
    request = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access).build_criterion(
        case_id=access.sample_key.case_id,
        spec=generation_spec_fixture(),
    )
    criterion_spec = next(
        item for item in access.manifest.criteria if item.case_id == access.sample_key.case_id
    )
    execution = create_normalized_criterion_execution(
        status=CriterionExecutionStatus.PROVIDER_UNAVAILABLE,
        test_bundle_sha256=criterion_spec.test_bundle_sha256,
        requested_at_utc=access.global_sealed_at_utc,
    )

    with pytest.raises(CriterionRecordError, match="request must follow"):
        CriterionRecordFactory().build(
            access,
            request=request,
            response=None,
            execution=execution,
            revealed_at_utc=access.global_sealed_at_utc + timedelta(seconds=1),
        )


def test_criterion_response_must_match_the_exact_request(tmp_path: Path) -> None:
    gate, _ = complete_gate_fixture(tmp_path)
    access = gate.open(
        gate.issue(benchmark_key("sample-001")),
        manifest_loader=evaluator_manifest_fixture,
    )
    request = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access).build_criterion(
        case_id=access.sample_key.case_id,
        spec=generation_spec_fixture(),
    )
    wrong_spec = generation_spec_fixture().model_copy(update={"sample_id": "another-sample"})
    wrong_request = create_student_generation_request(
        spec=wrong_spec,
        payload=request.task_payload,
    )
    wrong_response = create_recorded_response(
        request=wrong_request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response="def held_out_solution(value):\n    return value\n",
        captured_at_utc=access.global_sealed_at_utc + timedelta(seconds=1),
    )
    execution = create_normalized_criterion_execution(
        status=CriterionExecutionStatus.COMPLETED,
        test_bundle_sha256=next(
            item.test_bundle_sha256
            for item in access.manifest.criteria
            if item.case_id == access.sample_key.case_id
        ),
        requested_at_utc=access.global_sealed_at_utc + timedelta(seconds=1),
        operation_id="sandbox-op-wrong-response",
        passed=1,
        failed=0,
        exit_code=0,
        stdout_sha256=_marker("wrong-response-stdout"),
        stderr_sha256=_marker("wrong-response-stderr"),
        executed_at_utc=access.global_sealed_at_utc + timedelta(seconds=2),
    )

    with pytest.raises(CriterionRecordError, match="another request"):
        CriterionRecordFactory().build(
            access,
            request=request,
            response=wrong_response,
            execution=execution,
            revealed_at_utc=access.global_sealed_at_utc + timedelta(seconds=3),
        )


@pytest.mark.parametrize(
    ("status", "selected_loss", "expected_effects"),
    [
        (
            CalibrationStatus.CALIBRATED,
            RepeatLossName.BRIER,
            (0.24, -0.32, -0.32, 0.24, 0.40),
        ),
        (
            CalibrationStatus.UNCALIBRATED_SCORE,
            RepeatLossName.CLASSIFICATION_ERROR,
            (1.0, 0.0, 0.0, 1.0, 1.0),
        ),
    ],
)
def test_scorer_matches_hand_calculated_losses_controls_and_safety(
    tmp_path: Path,
    status: CalibrationStatus,
    selected_loss: RepeatLossName,
    expected_effects: tuple[float, float, float, float, float],
) -> None:
    criterion, condition_store, _ = _criterion_fixture(tmp_path)
    calibration = _calibration(status)
    baselines = _baselines(criterion, calibration)
    commit_set = condition_store.get(criterion.key)
    assert commit_set is not None
    predictions = condition_store.get_records(criterion.key)

    metrics = BenchmarkScorer(calibration, scorer_version="scorer-v1").score(
        commit_set=commit_set,
        predictions=predictions,
        criterion=criterion,
        baselines=baselines,
        created_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
    )

    assert tuple(metric.estimator for metric in metrics) == tuple(RepeatEstimator)
    assert all(metric.selected_loss_name is selected_loss for metric in metrics)
    _assert_close(
        tuple(metric.score for metric in metrics),
        (0.7, 0.5, 0.9, 0.9, 0.5, 0.3),
    )
    _assert_close(
        tuple(metric.brier_loss for metric in metrics),
        (0.49, 0.25, 0.81, 0.81, 0.25, 0.09),
    )
    assert tuple(metric.classification_error for metric in metrics) == (
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
    )
    assert tuple(metric.false_confidence_acceptance for metric in metrics) == (
        True,
        False,
        True,
        True,
        False,
        False,
    )
    assert tuple(metric.unsafe_advancement for metric in metrics) == (
        True,
        False,
        True,
        True,
        None,
        None,
    )
    assert tuple(metric.action_disagreement_with_dialogue for metric in metrics) == (
        False,
        True,
        False,
        False,
        None,
        None,
    )
    effects = repeat_comparisons(metrics)
    _assert_close(
        (
            effects.primary_valid_effect,
            effects.unrelated_control_effect,
            effects.corrupted_control_effect,
            effects.dialogue_vs_constant_effect,
            effects.dialogue_vs_probe_only_effect,
        ),
        expected_effects,
    )


def test_missing_criterion_produces_no_invented_losses(tmp_path: Path) -> None:
    criterion, condition_store, _ = _criterion_fixture(
        tmp_path,
        status=CriterionExecutionStatus.SANDBOX_TIMEOUT,
    )
    calibration = _calibration(CalibrationStatus.UNCALIBRATED_SCORE)
    commit_set = condition_store.get(criterion.key)
    assert commit_set is not None
    metrics = BenchmarkScorer(calibration, scorer_version="scorer-v1").score(
        commit_set=commit_set,
        predictions=condition_store.get_records(criterion.key),
        criterion=criterion,
        baselines=_baselines(criterion, calibration),
        created_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
    )

    assert all(not metric.eligible for metric in metrics)
    assert all(metric.loss_value is None for metric in metrics)
    assert all(metric.brier_loss is None for metric in metrics)
    assert all(metric.classification_error is None for metric in metrics)
    assert all(metric.execution_failure for metric in metrics)
    assert repeat_comparisons(metrics).primary_valid_effect is None


def test_scorer_rejects_calibration_from_another_tracker(tmp_path: Path) -> None:
    criterion, condition_store, _ = _criterion_fixture(tmp_path)
    calibration = _calibration(
        CalibrationStatus.UNCALIBRATED_SCORE,
        tracker_version="another-tracker-v1",
    )
    commit_set = condition_store.get(criterion.key)
    assert commit_set is not None

    with pytest.raises(ScoringError, match="another tracker"):
        BenchmarkScorer(calibration, scorer_version="scorer-v1").score(
            commit_set=commit_set,
            predictions=condition_store.get_records(criterion.key),
            criterion=criterion,
            baselines=_baselines(criterion, calibration),
            created_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
        )


def test_baselines_and_metric_helpers_are_frozen_and_hand_checkable(tmp_path: Path) -> None:
    criterion, _, _ = _criterion_fixture(tmp_path)
    calibration = _calibration(CalibrationStatus.CALIBRATED)
    constant, probe_only = _baselines(criterion, calibration)

    assert calibration.primary_metric is PrimaryMetric.PAIRED_BRIER_DIFFERENCE
    assert constant.baseline is ScoringBaseline.CONSTANT_PREVALENCE
    assert constant.score == 0.5
    assert constant.binary_decision is False
    assert probe_only.baseline is ScoringBaseline.PROBE_ONLY
    assert math.isclose(probe_only.score, 0.3)
    assert probe_only.binary_decision is False
    assert math.isclose(brier_loss(0.7, False), 0.49)
    assert classification_error(0.7, False, threshold=0.6) == 1.0

    with pytest.raises(ValidationError, match="status and primary metric"):
        CalibrationDecision.model_validate(
            {
                **calibration.model_dump(mode="python"),
                "primary_metric": PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE,
            }
        )
