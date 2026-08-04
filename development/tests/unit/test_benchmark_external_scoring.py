# pyright: reportPrivateUsage=false
"""End-to-end deterministic scoring tests for a sealed external run."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.calibration import (
    CalibrationManifestInventory,
    UncalibratedDecisionReport,
)
from socratic_tutor.benchmark.evaluator.scoring import (
    CalibrationDecision,
    CalibrationStatus,
    create_calibration_decision,
)
from socratic_tutor.benchmark.external_criterion import run_external_criterion
from socratic_tutor.benchmark.external_replay import (
    ExternalReplayPlan,
    ExternalReplayReport,
    replay_published_datasets,
)
from socratic_tutor.benchmark.external_scoring import run_external_scoring
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.global_seal import FilesystemGlobalDecisionSealStore
from tests.unit.test_benchmark_external_criterion import (
    BENCHMARK_ROOT,
    CRITERION_AT,
    PROMPT,
    FakeExecutor,
    FakeGateway,
    IncrementingClock,
    _no_sleep,
    _sealed_fixture,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SPECIFICATION = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
PIXI_LOCK = WORKSPACE_ROOT / "pixi.lock"
SCORED_AT = datetime(2026, 9, 1, 13, 0, tzinfo=UTC)


def test_scores_sealed_run_once_and_preserves_missingness(tmp_path: Path) -> None:
    specification = load_analysis_specification(ANALYSIS_SPECIFICATION)
    authored_hash = file_sha256((BENCHMARK_ROOT / "manifest.yaml").read_bytes())
    calibration = create_calibration_decision(
        status=CalibrationStatus.UNCALIBRATED_SCORE,
        policy_threshold=specification.policy_threshold,
        calibration_task_family_hash=canonical_sha256(()),
        calibration_data_hash=canonical_sha256(()),
        tracker_version="simple-v1",
        diagnostics_hash=canonical_sha256({"status": "no_independent_calibration_cases"}),
        decision_owner="researcher",
        decided_at_utc=datetime(2026, 8, 30, 9, 0, tzinfo=UTC),
    )
    calibration_report = _calibration_report(
        calibration=calibration,
        manifest_hash=authored_hash,
        specification_hash=analysis_specification_hash(specification),
    )
    public, evaluator, criterion_plan = _sealed_fixture(
        tmp_path,
        calibration_decision_hash=calibration.decision_hash,
    )
    failed_case = public.cases[0].case_id
    asyncio.run(
        run_external_criterion(
            plan=criterion_plan,
            public_manifest=public,
            evaluator_manifest=evaluator,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            system_prompt=PROMPT.read_text(encoding="utf-8"),
            gateway=FakeGateway(fail_case_id=failed_case),
            executor=FakeExecutor(),
            clock=IncrementingClock(),
            sleeper=_no_sleep,
        )
    )
    replay_root = tmp_path / "replay" / "preanalysis-v1"
    replay_plan, replay_report = _replay_evidence(tmp_path, replay_root)
    write_immutable_json(replay_root / "external_replay_plan.json", replay_plan)
    write_immutable_json(replay_root / "external_replay_report.json", replay_report)
    public_path = tmp_path / "public_manifest.json"
    calibration_path = tmp_path / "calibration_report.json"
    write_immutable_json(public_path, public)
    write_immutable_json(calibration_path, calibration_report)

    summary = run_external_scoring(
        replay_report_path=replay_root / "external_replay_report.json",
        analysis_specification_path=ANALYSIS_SPECIFICATION,
        calibration_report_path=calibration_path,
        public_manifest_path=public_path,
        benchmark_root=BENCHMARK_ROOT,
        seal_root=tmp_path,
        pixi_lock_path=PIXI_LOCK,
        scoring_code_revision="external-scoring-unit",
        created_at_utc=SCORED_AT,
    )

    assert summary.gate_passed
    assert summary.criterion_count == 24
    assert summary.eligible_criterion_count == 23
    assert summary.missing_case_ids == (failed_case,)
    assert summary.baseline_count == 48
    assert summary.repeat_metric_count == 144
    assert summary.eligible_repeat_metric_count == 138
    assert summary.case_aggregate_count == 120
    assert summary.eligible_case_aggregate_count == 115
    assert summary.effect_interpretation_status == "not_computed"
    assert summary.network_calls_made == 0
    assert summary.sandbox_calls_made == 0
    assert (
        run_external_scoring(
            replay_report_path=replay_root / "external_replay_report.json",
            analysis_specification_path=ANALYSIS_SPECIFICATION,
            calibration_report_path=calibration_path,
            public_manifest_path=public_path,
            benchmark_root=BENCHMARK_ROOT,
            seal_root=tmp_path,
            pixi_lock_path=PIXI_LOCK,
            scoring_code_revision="external-scoring-unit",
            created_at_utc=SCORED_AT,
        )
        == summary
    )


def _calibration_report(
    *,
    calibration: CalibrationDecision,
    manifest_hash: str,
    specification_hash: str,
) -> UncalibratedDecisionReport:
    inventory = CalibrationManifestInventory(
        manifest_ref="v1/manifest.yaml",
        benchmark_version="v1",
        manifest_hash=manifest_hash,
        calibration_case_count=0,
        calibration_task_families=(),
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.uncalibrated_decision_report.v1",
        "benchmark_version": "v1",
        "heldout_manifest_hash": manifest_hash,
        "analysis_specification_hash": specification_hash,
        "source_manifests": (inventory,),
        "calibration_case_count": 0,
        "reason": "no_independent_calibration_cases",
        "decision": calibration,
    }
    draft = UncalibratedDecisionReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    return UncalibratedDecisionReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )


def _replay_evidence(
    seal_root: Path,
    replay_root: Path,
) -> tuple[ExternalReplayPlan, ExternalReplayReport]:
    conditions = FilesystemConditionCommitStore(seal_root / "decision")
    seal = FilesystemGlobalDecisionSealStore(seal_root / "decision", conditions).load()
    assert seal is not None
    comparison = replay_published_datasets(
        seal_root=seal_root,
        output_dataset_root=replay_root / "datasets",
    )
    replay_plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_replay_plan.v1",
        "run_id": seal.run_id,
        "postcriterion_integrity_report_hash": "a" * 64,
        "global_seal_hash": seal.seal_hash,
        "decision_response_root_hash": "b" * 64,
        "criterion_response_root_hash": "c" * 64,
        "replay_code_revision": "external-replay-unit",
        "replay_pixi_lock_hash": file_sha256(PIXI_LOCK.read_bytes()),
        "created_at_utc": CRITERION_AT,
    }
    plan_draft = ExternalReplayPlan.model_construct(
        _fields_set=set(replay_plan_content), **replay_plan_content, plan_hash="0" * 64
    )
    plan = ExternalReplayPlan.model_validate(
        {
            **replay_plan_content,
            "plan_hash": model_content_hash(plan_draft, exclude={"plan_hash"}),
        }
    )
    report_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_replay_report.v1",
        "run_id": seal.run_id,
        "replay_plan_hash": plan.plan_hash,
        "postcriterion_integrity_report_hash": "a" * 64,
        "replayed_decision_response_count": 48,
        "replayed_criterion_response_count": 23,
        "replayed_request_count": 71,
        "decision_response_root_hash": "b" * 64,
        "criterion_response_root_hash": "c" * 64,
        **comparison,
        "scored_dataset_status": "not_yet_created",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": CRITERION_AT,
        "gate_passed": True,
    }
    report_draft = ExternalReplayReport.model_construct(
        _fields_set=set(report_content), **report_content, report_hash="0" * 64
    )
    report = ExternalReplayReport.model_validate(
        {
            **report_content,
            "report_hash": model_content_hash(report_draft, exclude={"report_hash"}),
        }
    )
    return plan, report
