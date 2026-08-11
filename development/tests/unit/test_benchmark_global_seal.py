"""Run-plan accounting and atomic global decision-seal tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.benchmark.public import (
    BenchmarkSampleKey,
    CommitStorageError,
    DecisionRunPlan,
    DecisionRunProvenance,
    FilesystemConditionCommitStore,
    FilesystemGlobalDecisionSealStore,
    GlobalDecisionSeal,
    GlobalSealAccountingError,
    GlobalSealConflictError,
    GlobalSealStorageError,
    SampleDecisionFailure,
    SampleDecisionReason,
    SampleDecisionStatus,
    create_decision_run_plan,
    decision_record_root_hash,
    decision_run_plan_hash,
    global_decision_seal_hash,
)
from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    PredictionRecordFactory,
)
from tests.unit.test_benchmark_predictions import COMMITTED_AT, paired_fixture

RUN_ID = "dev-global-seal-001"
MODEL_ROUTE_ID = "recorded_fixture/authored-deterministic"
PLAN_CREATED_AT = datetime(2026, 8, 9, 8, 0, tzinfo=UTC)


def _records(sample_id: str) -> tuple[DecisionPredictionRecord, ...]:
    public, paired = paired_fixture()
    return PredictionRecordFactory(public).build(
        paired,
        run_id=RUN_ID,
        sample_id=sample_id,
        model_route_id=MODEL_ROUTE_ID,
        public_record_hash=canonical_sha256({"record": "dev-aliasing-public"}),
        committed_at_utc=COMMITTED_AT,
    )


def _key(sample_id: str) -> BenchmarkSampleKey:
    return BenchmarkSampleKey(
        benchmark_version="dev-v0",
        run_id=RUN_ID,
        case_id="dev-aliasing-001",
        sample_id=sample_id,
        model_route_id=MODEL_ROUTE_ID,
    )


def _plan(
    *sample_ids: str,
    public_manifest_hash: str | None = None,
) -> DecisionRunPlan:
    def marker(name: str) -> str:
        return canonical_sha256({"fixture": name})

    return create_decision_run_plan(
        benchmark_version="dev-v0",
        run_id=RUN_ID,
        provenance=DecisionRunProvenance(
            code_revision="4344529",
            dirty_worktree=False,
            pixi_lock_hash=marker("pixi-lock"),
            resolved_config_hash=marker("config"),
            public_manifest_hash=public_manifest_hash or marker("public-manifest"),
            split_hash=marker("split"),
            prompt_version="benchmark-student-v1",
            prompt_hash=marker("prompt"),
            tracker_version="simple-v1",
            policy_version="heuristic-v1",
            observation_schema="benchmark.condition_outcome.v1",
            action_mapping_hash=marker("actions"),
            calibration_decision_hash=None,
            analysis_plan_hash=marker("analysis"),
            review_gate_hash=marker("review"),
            root_seed=20260811,
        ),
        planned_samples=tuple(_key(sample_id) for sample_id in sample_ids),
        created_at_utc=PLAN_CREATED_AT,
    )


def _seal_records(
    store: FilesystemConditionCommitStore,
    sample_id: str,
) -> datetime:
    records = _records(sample_id)
    for record in records:
        store.append(record)
    return store.seal(_key(sample_id)).sealed_at_utc


benchmark_key = _key
decision_plan = _plan
prediction_records_for_sample = _records
seal_sample_records = _seal_records


def test_global_seal_accounts_for_complete_missing_and_invalid_samples(
    tmp_path: Path,
) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    condition_store.append(_records("sample-002")[0])
    plan = _plan("sample-003", "sample-001", "sample-002")
    global_time = condition_sealed_at + timedelta(seconds=1)
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: global_time,
    )

    seal = store.publish(
        plan,
        failures=(
            SampleDecisionFailure(
                key=_key("sample-002"),
                status=SampleDecisionStatus.PRECRITERION_MISSING,
                reason_code=SampleDecisionReason.INCOMPLETE_CONDITION_SET,
            ),
            SampleDecisionFailure(
                key=_key("sample-003"),
                status=SampleDecisionStatus.INVALID,
                reason_code=SampleDecisionReason.DECISION_FAILURE,
                failure_record_hash=canonical_sha256({"failure": "tracker"}),
            ),
        ),
    )

    assert tuple(record.key for record in seal.sample_statuses) == plan.planned_samples
    assert tuple(record.status for record in seal.sample_statuses) == (
        SampleDecisionStatus.COMPLETE,
        SampleDecisionStatus.PRECRITERION_MISSING,
        SampleDecisionStatus.INVALID,
    )
    assert seal.planned_sample_count == 3
    assert len(seal.complete_set_hashes) == 1
    assert len(seal.sample_statuses[0].prediction_hashes) == 4
    assert len(seal.sample_statuses[1].prediction_hashes) == 1
    assert seal.sample_statuses[2].prediction_hashes == ()
    assert seal.decision_record_root_hash == decision_record_root_hash(seal.sample_statuses)
    assert seal.seal_hash == global_decision_seal_hash(seal)
    assert store.publish(plan) == seal


def test_global_seal_survives_restart_and_revalidates_underlying_sets(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    plan = _plan("sample-001")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=1),
    )
    published = store.publish(plan)

    restarted_conditions = FilesystemConditionCommitStore(tmp_path)
    restarted = FilesystemGlobalDecisionSealStore(tmp_path, restarted_conditions)
    assert restarted.load() == published
    durable = GlobalDecisionSeal.model_validate_json(
        (tmp_path / "commitment_manifest.json").read_bytes()
    )
    assert durable == published


def test_global_seal_detects_decision_writes_after_publication(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    plan = _plan("sample-001")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: COMMITTED_AT + timedelta(days=1),
    )
    store.publish(
        plan,
        failures=(
            SampleDecisionFailure(
                key=_key("sample-001"),
                status=SampleDecisionStatus.PRECRITERION_MISSING,
                reason_code=SampleDecisionReason.GENERATION_MISSING,
            ),
        ),
    )

    condition_store.append(_records("sample-001")[0])
    with pytest.raises(
        GlobalSealStorageError,
        match="predictions changed",
    ):
        store.load()


def test_run_plan_must_precede_prediction_commits(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    original = _plan("sample-001")
    late_plan = create_decision_run_plan(
        benchmark_version=original.benchmark_version,
        run_id=original.run_id,
        provenance=original.provenance,
        planned_samples=original.planned_samples,
        created_at_utc=condition_sealed_at + timedelta(seconds=1),
    )
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=2),
    )

    with pytest.raises(GlobalSealAccountingError, match="plan must precede"):
        store.publish(late_plan)


def test_global_seal_rejects_missing_duplicate_and_conflicting_accounting(
    tmp_path: Path,
) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    plan = _plan("sample-001", "sample-002")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=1),
    )

    with pytest.raises(GlobalSealAccountingError, match="no terminal outcome"):
        store.publish(plan)

    missing = SampleDecisionFailure(
        key=_key("sample-002"),
        status=SampleDecisionStatus.PRECRITERION_MISSING,
        reason_code=SampleDecisionReason.GENERATION_MISSING,
    )
    with pytest.raises(GlobalSealAccountingError, match="duplicate"):
        store.publish(plan, failures=(missing, missing))

    with pytest.raises(GlobalSealAccountingError, match="also has a failure"):
        store.publish(
            plan,
            failures=(
                missing,
                SampleDecisionFailure(
                    key=_key("sample-001"),
                    status=SampleDecisionStatus.INVALID,
                    reason_code=SampleDecisionReason.DECISION_FAILURE,
                ),
            ),
        )


def test_global_seal_rejects_unplanned_condition_set(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    first_time = _seal_records(condition_store, "sample-001")
    second_time = _seal_records(condition_store, "sample-002")
    plan = _plan("sample-001")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: max(first_time, second_time) + timedelta(seconds=1),
    )

    with pytest.raises(GlobalSealAccountingError, match="unplanned"):
        store.publish(plan)


def test_corrupt_condition_set_blocks_global_publication(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    set_path = next((tmp_path / "sets").glob("*.json"))
    set_path.write_text("{}\n", encoding="utf-8")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=1),
    )

    with pytest.raises(CommitStorageError):
        store.publish(
            _plan("sample-001"),
            failures=(
                SampleDecisionFailure(
                    key=_key("sample-001"),
                    status=SampleDecisionStatus.INVALID,
                    reason_code=SampleDecisionReason.INTEGRITY_FAILURE,
                ),
            ),
        )

    assert not (tmp_path / "commitment_manifest.json").exists()


def test_atomic_replace_failure_leaves_no_global_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    plan = _plan("sample-001")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=1),
    )

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated crash before replacement")

    monkeypatch.setattr("socratic_tutor.benchmark.public.global_seal.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        store.publish(plan)

    assert not (tmp_path / "commitment_manifest.json").exists()
    assert tuple(tmp_path.glob(".commitment_manifest.json.*.tmp")) == ()


def test_run_plan_and_failure_contracts_reject_invalid_inputs() -> None:
    plan = _plan("sample-001", "sample-002")
    tampered = plan.model_dump(mode="python")
    tampered["run_id"] = "another-run"
    with pytest.raises(ValidationError, match=r"sample|hash"):
        DecisionRunPlan.model_validate(tampered)

    with pytest.raises(ValidationError, match="not valid"):
        SampleDecisionFailure(
            key=_key("sample-001"),
            status=SampleDecisionStatus.INVALID,
            reason_code=SampleDecisionReason.GENERATION_MISSING,
        )


def test_existing_global_seal_rejects_another_plan(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = _seal_records(condition_store, "sample-001")
    first_plan = _plan("sample-001")
    store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=1),
    )
    store.publish(first_plan)

    draft = first_plan.model_copy(update={"created_at_utc": PLAN_CREATED_AT + timedelta(seconds=1)})
    changed = draft.model_dump(mode="python")
    changed["plan_hash"] = decision_run_plan_hash(draft)
    second_plan = DecisionRunPlan.model_validate(changed)
    with pytest.raises(GlobalSealConflictError, match="Another run plan"):
        store.publish(second_plan)
