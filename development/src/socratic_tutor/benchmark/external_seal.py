"""Seal held-out external decisions before any criterion access."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.evidence_specificity import (
    ExternalRunPreflight,
    load_external_run_preflight,
)
from socratic_tutor.benchmark.external_decision import ExternalDecisionGenerationReport
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    load_external_model_execution_protocol,
    validate_external_model_execution_protocol,
)
from socratic_tutor.benchmark.generation import GenerationChannel, ModelRoute
from socratic_tutor.benchmark.hashing import (
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.benchmark.inferential_hierarchy import InferentialHierarchy
from socratic_tutor.benchmark.methodology_clarification import MethodologyClarification
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import (
    BenchmarkSampleKey,
    FilesystemConditionCommitStore,
)
from socratic_tutor.benchmark.public.controls import ControlFactory, ProbeExecutionSummary
from socratic_tutor.benchmark.public.datasets import publish_condition_predictions
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunProvenance,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionFailure,
    SampleDecisionReason,
    SampleDecisionStatus,
    create_decision_run_plan,
)
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.public.offline import (
    ProbeSummarySet,
    load_initial_tracker_state,
)
from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    PredictionRecordFactory,
)
from socratic_tutor.benchmark.public.runner import PairedConditionRunner
from socratic_tutor.benchmark.public_rating import (
    FinalPublicAnswerRating,
    PublicAnswerRatingReport,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
    replay_recorded_response,
)
from socratic_tutor.contracts import ContractModel, Evidence, TutorAction
from socratic_tutor.policies import choose_action
from socratic_tutor.sandbox import (
    ModalSandboxExecutor,
    SandboxExecutor,
    execute_authored_python_tests,
)
from socratic_tutor.tracking import update_tracker


class ExternalDecisionSealError(ValueError):
    """A held-out decision seal is incomplete, inconsistent, or unsafe."""


class ExternalDecisionSealPlan(ContractModel):
    """Content-addressed public-side inputs fixed before decision commits."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_seal_plan.v1"] = (
        "benchmark.external_decision_seal_plan.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preflight_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recorded_responses_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_rating_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology_clarification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inferential_hierarchy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_specification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_route: ModelRoute
    tracker_version: Literal["simple-v1"] = "simple-v1"
    policy_version: Literal["heuristic-v1"] = "heuristic-v1"
    system_prompt_version: str = Field(min_length=1)
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_seed: int = Field(ge=0, le=2**32 - 1)
    code_revision: str = Field(min_length=1)
    dirty_worktree: bool
    pixi_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: datetime
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_plan(self) -> ExternalDecisionSealPlan:
        _require_utc(self.created_at_utc, "External decision-seal plan time")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("External decision-seal plan hash does not match its content")
        return self


class ExternalEvidenceExecutionArtifact(ContractModel):
    """One resumable, criterion-free execution of a recorded evidence response."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_evidence_execution.v1"] = (
        "benchmark.external_evidence_execution.v1"
    )
    case_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_test_ref: str = Field(min_length=1)
    evidence_test_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_status: Literal["completed", "failed"]
    outcome_status: Literal["completed", "invalid_submission", "execution_error"] | None = None
    passed: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    sandbox_id: str | None = None
    exit_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    executed_at_utc: datetime
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact(self) -> ExternalEvidenceExecutionArtifact:
        _require_utc(self.executed_at_utc, "Evidence execution time")
        if self.outcome_status == "completed":
            if self.passed is None or self.failed is None or self.passed + self.failed == 0:
                raise ValueError("Completed evidence execution requires observable test counts")
        elif self.passed not in {None, 0} or self.failed not in {None, 0}:
            raise ValueError("Incomplete evidence execution cannot claim passed or failed checks")
        if self.artifact_hash != model_content_hash(self, exclude={"artifact_hash"}):
            raise ValueError("Evidence execution artifact hash does not match its content")
        return self

    @property
    def usable(self) -> bool:
        return (
            self.execution_status == "completed"
            and self.outcome_status == "completed"
            and self.passed is not None
            and self.failed is not None
            and self.passed + self.failed > 0
        )


class ExternalDecisionSealReport(ContractModel):
    """Accounting summary published after all decisions become immutable."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_decision_seal_report.v1"] = (
        "benchmark.external_decision_seal_report.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    seal_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_run_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_rating_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    methodology_clarification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inferential_hierarchy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    planned_case_count: int = Field(ge=1)
    complete_case_count: int = Field(ge=0)
    precriterion_missing_case_count: int = Field(ge=0)
    invalid_case_count: int = Field(ge=0)
    evidence_execution_count: int = Field(ge=0)
    usable_evidence_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    global_seal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sealed_at_utc: datetime
    all_cases_complete: bool
    gate_passed: bool
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self) -> ExternalDecisionSealReport:
        _require_utc(self.sealed_at_utc, "External decision seal time")
        if self.planned_case_count != (
            self.complete_case_count
            + self.precriterion_missing_case_count
            + self.invalid_case_count
        ):
            raise ValueError("External decision-seal case counts do not reconcile")
        if self.prediction_count != self.complete_case_count * 4:
            raise ValueError("Every complete case must publish four predictions")
        if self.all_cases_complete != (self.complete_case_count == self.planned_case_count):
            raise ValueError("All-complete flag does not match case accounting")
        if not self.gate_passed:
            raise ValueError("A persisted external decision-seal report must pass accounting")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("External decision-seal report hash does not match its content")
        return self


type Clock = Callable[[], datetime]


def run_external_decision_seal_from_files(
    *,
    protocol_path: Path,
    preflight_path: Path,
    generation_report_path: Path,
    recorded_responses_path: Path,
    public_rating_report_path: Path,
    methodology_clarification_path: Path,
    inferential_hierarchy_path: Path,
    manifest_path: Path,
    analysis_specification_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
    dirty_worktree: bool,
    created_at_utc: datetime,
    executor: SandboxExecutor,
    clock: Clock | None = None,
) -> ExternalDecisionSealReport:
    """Validate frozen sources, execute evidence only, and seal all decisions."""

    protocol = load_external_model_execution_protocol(protocol_path)
    preflight = load_external_run_preflight(preflight_path)
    generation = _load_json_model(generation_report_path, ExternalDecisionGenerationReport)
    ratings = _load_json_model(public_rating_report_path, PublicAnswerRatingReport)
    methodology = _load_json_model(methodology_clarification_path, MethodologyClarification)
    hierarchy = _load_json_model(inferential_hierarchy_path, InferentialHierarchy)
    authored = load_and_verify_manifest(manifest_path)
    if authored.status is not ManifestStatus.FROZEN or authored.benchmark_version != "v1":
        raise ExternalDecisionSealError("External decision sealing requires frozen benchmark v1")
    public = project_public_manifest(authored, projected_at_utc=protocol.created_at_utc)
    analysis = load_analysis_specification(analysis_specification_path)
    validate_external_model_execution_protocol(
        protocol,
        manifest=authored,
        public_manifest=public,
        analysis_specification=analysis,
    )
    records = _load_records(recorded_responses_path)
    _validate_frozen_sources(
        protocol=protocol,
        preflight=preflight,
        generation=generation,
        ratings=ratings,
        methodology=methodology,
        hierarchy=hierarchy,
        public=public,
        analysis_hash=analysis_specification_hash(analysis),
        records=records,
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ExternalDecisionSealError(
            f"Could not read Pixi lock file: {pixi_lock_path}"
        ) from error
    plan = _create_seal_plan(
        protocol=protocol,
        preflight=preflight,
        generation=generation,
        ratings=ratings,
        methodology=methodology,
        hierarchy=hierarchy,
        public=public,
        analysis_hash=analysis_specification_hash(analysis),
        code_revision=code_revision,
        dirty_worktree=dirty_worktree,
        pixi_lock_hash=pixi_lock_hash,
        created_at_utc=created_at_utc,
    )
    return run_external_decision_seal(
        plan=plan,
        manifest=public,
        benchmark_root=manifest_path.resolve().parent,
        records=records,
        final_ratings=ratings.final_ratings,
        output_root=output_root,
        executor=executor,
        clock=clock,
    )


def run_external_decision_seal_with_modal(
    *,
    protocol_path: Path,
    preflight_path: Path,
    generation_report_path: Path,
    recorded_responses_path: Path,
    public_rating_report_path: Path,
    methodology_clarification_path: Path,
    inferential_hierarchy_path: Path,
    manifest_path: Path,
    analysis_specification_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
    dirty_worktree: bool,
    created_at_utc: datetime,
) -> ExternalDecisionSealReport:
    """Run the held-out decision seal using the only permitted external executor."""

    return run_external_decision_seal_from_files(
        protocol_path=protocol_path,
        preflight_path=preflight_path,
        generation_report_path=generation_report_path,
        recorded_responses_path=recorded_responses_path,
        public_rating_report_path=public_rating_report_path,
        methodology_clarification_path=methodology_clarification_path,
        inferential_hierarchy_path=inferential_hierarchy_path,
        manifest_path=manifest_path,
        analysis_specification_path=analysis_specification_path,
        pixi_lock_path=pixi_lock_path,
        output_root=output_root,
        code_revision=code_revision,
        dirty_worktree=dirty_worktree,
        created_at_utc=created_at_utc,
        executor=ModalSandboxExecutor(),
    )


def run_external_decision_seal(
    *,
    plan: ExternalDecisionSealPlan,
    manifest: PublicBenchmarkManifest,
    benchmark_root: Path,
    records: tuple[RecordedGenerationResponse, ...],
    final_ratings: tuple[FinalPublicAnswerRating, ...],
    output_root: Path,
    executor: SandboxExecutor,
    clock: Clock | None = None,
) -> ExternalDecisionSealReport:
    """Create four decisions per usable case and publish one global barrier."""

    now = clock or (lambda: datetime.now(UTC))
    root = output_root.resolve()
    decision_root = root / "decision"
    datasets_root = root / "datasets"
    _validate_core_inputs(plan, manifest, records, final_ratings)
    write_immutable_json(root / "external_decision_seal_plan.json", plan)
    completed_report_path = root / "external_decision_seal_report.json"
    if completed_report_path.exists():
        completed = _load_json_model(completed_report_path, ExternalDecisionSealReport)
        if (
            completed.run_id != plan.run_id
            or completed.seal_plan_hash != plan.plan_hash
            or completed.generation_report_hash != plan.generation_report_hash
        ):
            raise ExternalDecisionSealError("Existing seal report belongs to another run")
        return completed

    indexed_records = {
        (record.request.case_id, record.request.channel): record for record in records
    }
    ratings = {rating.case_id: rating for rating in final_ratings}
    route_id = f"{plan.model_route.provider}/{plan.model_route.model}"
    keys = tuple(
        BenchmarkSampleKey(
            benchmark_version=manifest.benchmark_version,
            run_id=plan.run_id,
            case_id=case.case_id,
            sample_id=_normalised_sample_id(
                indexed_records[(case.case_id, GenerationChannel.PUBLIC)],
                run_id=plan.run_id,
                case_id=case.case_id,
            ),
            model_route_id=route_id,
        )
        for case in manifest.cases
    )
    provenance = DecisionRunProvenance(
        code_revision=plan.code_revision,
        dirty_worktree=plan.dirty_worktree,
        pixi_lock_hash=plan.pixi_lock_hash,
        resolved_config_hash=plan.plan_hash,
        public_manifest_hash=manifest.projection_hash,
        split_hash=canonical_sha256(manifest.splits),
        prompt_version=plan.system_prompt_version,
        prompt_hash=plan.system_prompt_sha256,
        tracker_version=plan.tracker_version,
        policy_version=plan.policy_version,
        observation_schema="benchmark.condition_outcome.v1",
        action_mapping_hash=canonical_sha256(tuple(TutorAction)),
        calibration_decision_hash=plan.calibration_decision_hash,
        analysis_plan_hash=plan.inferential_hierarchy_hash,
        review_gate_hash=plan.public_rating_report_hash,
        root_seed=plan.root_seed,
    )
    run_plan = create_decision_run_plan(
        benchmark_version=manifest.benchmark_version,
        run_id=plan.run_id,
        provenance=provenance,
        planned_samples=keys,
        created_at_utc=plan.created_at_utc,
    )
    write_immutable_json(root / "run_plan.json", run_plan)

    controls = ControlFactory(benchmark_root, manifest)
    executions: dict[str, ExternalEvidenceExecutionArtifact] = {}
    probes: dict[str, ProbeExecutionSummary] = {}
    for case in manifest.cases:
        evidence_record = indexed_records[(case.case_id, GenerationChannel.EVIDENCE)]
        execution = _load_or_execute_evidence(
            root=decision_root,
            case_id=case.case_id,
            record=evidence_record,
            evidence_test_ref=case.evidence_test_ref,
            benchmark_root=benchmark_root,
            executor=executor,
            clock=now,
        )
        executions[case.case_id] = execution
        if execution.usable:
            assert execution.passed is not None and execution.failed is not None
            probes[case.case_id] = controls.probe_summary(
                case_id=case.case_id,
                passed=execution.passed,
                failed=execution.failed,
            )
    if probes:
        _write_probe_set(root, manifest, probes)

    condition_store = FilesystemConditionCommitStore(decision_root, clock=now)
    runner = PairedConditionRunner(
        manifest,
        tracker_update=update_tracker,
        policy_decision=choose_action,
        tracker_version=plan.tracker_version,
        policy_version=plan.policy_version,
    )
    prediction_factory = PredictionRecordFactory(manifest)
    predictions: list[DecisionPredictionRecord] = []
    failures: list[SampleDecisionFailure] = []
    for key in run_plan.planned_samples:
        execution = executions[key.case_id]
        if not execution.usable:
            status = (
                SampleDecisionStatus.INVALID
                if execution.outcome_status in {"invalid_submission", "execution_error"}
                else SampleDecisionStatus.PRECRITERION_MISSING
            )
            reason = (
                SampleDecisionReason.DECISION_FAILURE
                if status is SampleDecisionStatus.INVALID
                else SampleDecisionReason.INCOMPLETE_CONDITION_SET
            )
            failures.append(
                SampleDecisionFailure(
                    key=key,
                    status=status,
                    reason_code=reason,
                    failure_record_hash=(
                        execution.artifact_hash if status is SampleDecisionStatus.INVALID else None
                    ),
                )
            )
            continue
        try:
            unrelated = controls.unrelated(case_id=key.case_id, evidence_by_case=probes)
        except ValueError:
            failures.append(
                SampleDecisionFailure(
                    key=key,
                    status=SampleDecisionStatus.PRECRITERION_MISSING,
                    reason_code=SampleDecisionReason.INCOMPLETE_CONDITION_SET,
                )
            )
            continue
        existing = condition_store.get(key)
        if existing is not None:
            predictions.extend(condition_store.get_records(key))
            continue
        public_record = indexed_records[(key.case_id, GenerationChannel.PUBLIC)]
        rating = ratings[key.case_id]
        public_evidence = Evidence(
            category=rating.category,
            confidence=rating.confidence,
            rationale=f"Independent public-answer rating resolved by {rating.resolution}.",
            observed_signals=(f"rating_resolution:{rating.resolution}",),
        )
        probe = probes[key.case_id]
        paired = runner.run_case(
            case_id=key.case_id,
            initial_tracker_state=load_initial_tracker_state(
                benchmark_root,
                manifest,
                case_id=key.case_id,
            ),
            public_evidence=public_evidence,
            probe_evidence=probe,
            unrelated_evidence=unrelated,
            corrupted_evidence=controls.corrupted(case_id=key.case_id, evidence=probe),
        )
        staged = condition_store.get_staged_records(key)
        committed_at = staged[0].committed_at_utc if staged else now()
        _require_after(committed_at, plan.created_at_utc, "Prediction commit")
        case_records = prediction_factory.build(
            paired,
            run_id=plan.run_id,
            sample_id=key.sample_id,
            model_route_id=route_id,
            public_record_hash=public_record.response_hash,
            committed_at_utc=committed_at,
        )
        for prediction in case_records:
            condition_store.append(prediction)
        condition_store.seal(key)
        predictions.extend(case_records)

    global_store = FilesystemGlobalDecisionSealStore(decision_root, condition_store, clock=now)
    global_seal = global_store.publish(run_plan, failures=tuple(failures))
    publication = publish_condition_predictions(
        AtomicParquetDatasetStore(datasets_root),
        tuple(predictions),
        global_seal=global_seal,
        published_at_utc=global_seal.sealed_at_utc + timedelta(microseconds=1),
    )
    status_counts = {
        status: sum(item.status is status for item in global_seal.sample_statuses)
        for status in SampleDecisionStatus
    }
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_seal_report.v1",
        "run_id": plan.run_id,
        "benchmark_version": "v1",
        "seal_plan_hash": plan.plan_hash,
        "decision_run_plan_hash": run_plan.plan_hash,
        "generation_report_hash": plan.generation_report_hash,
        "public_rating_report_hash": plan.public_rating_report_hash,
        "methodology_clarification_hash": plan.methodology_clarification_hash,
        "inferential_hierarchy_hash": plan.inferential_hierarchy_hash,
        "planned_case_count": len(run_plan.planned_samples),
        "complete_case_count": status_counts[SampleDecisionStatus.COMPLETE],
        "precriterion_missing_case_count": status_counts[SampleDecisionStatus.PRECRITERION_MISSING],
        "invalid_case_count": status_counts[SampleDecisionStatus.INVALID],
        "evidence_execution_count": len(executions),
        "usable_evidence_count": sum(item.usable for item in executions.values()),
        "prediction_count": len(predictions),
        "global_seal_hash": global_seal.seal_hash,
        "decision_publication_hash": publication.publication_hash,
        "sealed_at_utc": global_seal.sealed_at_utc,
        "all_cases_complete": status_counts[SampleDecisionStatus.COMPLETE]
        == len(run_plan.planned_samples),
        "gate_passed": True,
    }
    draft = ExternalDecisionSealReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ExternalDecisionSealReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(completed_report_path, report)
    return report


def _create_seal_plan(
    *,
    protocol: ExternalModelExecutionProtocol,
    preflight: ExternalRunPreflight,
    generation: ExternalDecisionGenerationReport,
    ratings: PublicAnswerRatingReport,
    methodology: MethodologyClarification,
    hierarchy: InferentialHierarchy,
    public: PublicBenchmarkManifest,
    analysis_hash: str,
    code_revision: str,
    dirty_worktree: bool,
    pixi_lock_hash: str,
    created_at_utc: datetime,
) -> ExternalDecisionSealPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_seal_plan.v1",
        "run_id": generation.run_id,
        "benchmark_version": "v1",
        "source_manifest_hash": generation.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "protocol_hash": protocol.protocol_hash,
        "preflight_hash": preflight.preflight_hash,
        "generation_report_hash": generation.report_hash,
        "recorded_responses_hash": generation.recorded_responses_hash,
        "public_rating_report_hash": ratings.report_hash,
        "methodology_clarification_hash": methodology.clarification_hash,
        "inferential_hierarchy_hash": hierarchy.hierarchy_hash,
        "analysis_specification_hash": analysis_hash,
        "calibration_decision_hash": methodology.calibration_decision_hash,
        "model_route": ModelRoute(
            provider=protocol.provider_id,
            model=protocol.requested_model_id,
        ),
        "tracker_version": methodology.tracker_version,
        "policy_version": "heuristic-v1",
        "system_prompt_version": protocol.system_prompt_version,
        "system_prompt_sha256": protocol.system_prompt_sha256,
        "root_seed": protocol.run_seed,
        "code_revision": code_revision,
        "dirty_worktree": dirty_worktree,
        "pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ExternalDecisionSealPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ExternalDecisionSealPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _validate_frozen_sources(
    *,
    protocol: ExternalModelExecutionProtocol,
    preflight: ExternalRunPreflight,
    generation: ExternalDecisionGenerationReport,
    ratings: PublicAnswerRatingReport,
    methodology: MethodologyClarification,
    hierarchy: InferentialHierarchy,
    public: PublicBenchmarkManifest,
    analysis_hash: str,
    records: tuple[RecordedGenerationResponse, ...],
) -> None:
    expected = {
        "preflight protocol": (preflight.external_protocol_hash, protocol.protocol_hash),
        "preflight analysis": (preflight.analysis_specification_hash, analysis_hash),
        "generation protocol": (generation.protocol_hash, protocol.protocol_hash),
        "generation preflight": (generation.preflight_hash, preflight.preflight_hash),
        "generation manifest": (generation.source_manifest_hash, protocol.source_manifest_hash),
        "generation projection": (generation.public_projection_hash, public.projection_hash),
        "rating report": (methodology.public_rating_report_hash, ratings.report_hash),
        "methodology protocol": (methodology.external_protocol_hash, protocol.protocol_hash),
        "methodology analysis": (methodology.analysis_specification_hash, analysis_hash),
        "hierarchy methodology": (
            hierarchy.methodology_clarification_hash,
            methodology.clarification_hash,
        ),
        "hierarchy protocol": (hierarchy.external_protocol_hash, protocol.protocol_hash),
        "hierarchy analysis": (hierarchy.analysis_specification_hash, analysis_hash),
    }
    for name, (declared, actual) in expected.items():
        if declared != actual:
            raise ExternalDecisionSealError(f"External decision seal has mismatched {name}")
    if not preflight.gate_passed:
        raise ExternalDecisionSealError("External run preflight did not pass")
    if not methodology.required_before_criterion_access:
        raise ExternalDecisionSealError("Methodology clarification is not criterion-gating")
    if not hierarchy.all_precriterion_clarifications_complete:
        raise ExternalDecisionSealError("Inferential hierarchy is incomplete")
    collection_hash = model_content_hash(_ResponseCollection(responses=records), exclude=set())
    if collection_hash != generation.recorded_responses_hash:
        raise ExternalDecisionSealError("Recorded responses differ from the generation report")
    if ratings.item_count != len(public.cases):
        raise ExternalDecisionSealError("Public ratings do not cover the public manifest")


def _validate_core_inputs(
    plan: ExternalDecisionSealPlan,
    manifest: PublicBenchmarkManifest,
    records: tuple[RecordedGenerationResponse, ...],
    ratings: tuple[FinalPublicAnswerRating, ...],
) -> None:
    if manifest.benchmark_version != plan.benchmark_version:
        raise ExternalDecisionSealError("Seal plan belongs to another benchmark")
    if manifest.source_manifest_hash != plan.source_manifest_hash:
        raise ExternalDecisionSealError("Seal plan belongs to another source manifest")
    if manifest.projection_hash != plan.public_projection_hash:
        raise ExternalDecisionSealError("Seal plan belongs to another public projection")
    expected = {
        (case.case_id, channel)
        for case in manifest.cases
        for channel in (GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE)
    }
    observed = {(record.request.case_id, record.request.channel) for record in records}
    if observed != expected or len(records) != len(expected):
        raise ExternalDecisionSealError(
            "Recorded responses require exactly one public and evidence response per case"
        )
    for record in records:
        if record.request.channel is GenerationChannel.CRITERION:
            raise ExternalDecisionSealError("Criterion recordings are forbidden before sealing")
        if record.original_source is not OriginalGenerationSource.CEREBRAS:
            raise ExternalDecisionSealError(
                "Decision recordings must use the frozen external route"
            )
        if record.request.model_route != plan.model_route:
            raise ExternalDecisionSealError("Recorded response uses another model route")
        if record.request.run_id != plan.run_id:
            raise ExternalDecisionSealError("Recorded response belongs to another run")
        replay_recorded_response(record.request, record)
    by_case: dict[str, set[str]] = {}
    for record in records:
        by_case.setdefault(record.request.case_id, set()).add(record.request.sample_id)
    if any(len(sample_ids) != 1 for sample_ids in by_case.values()):
        raise ExternalDecisionSealError("Public and evidence responses must share one sample ID")
    rating_ids = tuple(rating.case_id for rating in ratings)
    if len(set(rating_ids)) != len(rating_ids) or set(rating_ids) != {
        case.case_id for case in manifest.cases
    }:
        raise ExternalDecisionSealError("Final public ratings must cover every case exactly once")
    public_hashes = {
        record.request.case_id: record.response_hash
        for record in records
        if record.request.channel is GenerationChannel.PUBLIC
    }
    if any(rating.response_hash != public_hashes[rating.case_id] for rating in ratings):
        raise ExternalDecisionSealError("Public rating response hash does not match its recording")


def _load_or_execute_evidence(
    *,
    root: Path,
    case_id: str,
    record: RecordedGenerationResponse,
    evidence_test_ref: str,
    benchmark_root: Path,
    executor: SandboxExecutor,
    clock: Clock,
) -> ExternalEvidenceExecutionArtifact:
    bundle_path = benchmark_root / evidence_test_ref
    try:
        test_hash = file_sha256(bundle_path.read_bytes())
    except OSError as error:
        raise ExternalDecisionSealError(
            f"Could not read evidence tests: {evidence_test_ref}"
        ) from error
    path = root / "evidence_executions" / f"{record.response_hash}.json"
    if path.exists():
        artifact = _load_json_model(path, ExternalEvidenceExecutionArtifact)
        if (
            artifact.case_id != case_id
            or artifact.sample_id != record.request.sample_id
            or artifact.response_hash != record.response_hash
            or artifact.evidence_test_ref != evidence_test_ref
            or artifact.evidence_test_sha256 != test_hash
        ):
            raise ExternalDecisionSealError(
                f"Existing evidence execution belongs to another input: {case_id}"
            )
        return artifact
    result = execute_authored_python_tests(
        executor,
        response=record.final_response,
        bundle_path=bundle_path,
    )
    outcome = result.outcome
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_evidence_execution.v1",
        "case_id": case_id,
        "sample_id": record.request.sample_id,
        "response_hash": record.response_hash,
        "evidence_test_ref": evidence_test_ref,
        "evidence_test_sha256": test_hash,
        "execution_status": result.execution.status,
        "outcome_status": outcome.status if outcome is not None else None,
        "passed": outcome.passed if outcome is not None else None,
        "failed": outcome.failed if outcome is not None else None,
        "sandbox_id": result.execution.sandbox_id,
        "exit_code": result.execution.exit_code,
        "error_type": result.execution.error_type,
        "error_message": result.execution.error_message,
        "executed_at_utc": clock(),
    }
    draft = ExternalEvidenceExecutionArtifact.model_construct(
        _fields_set=set(content), **content, artifact_hash="0" * 64
    )
    artifact = ExternalEvidenceExecutionArtifact.model_validate(
        {**content, "artifact_hash": model_content_hash(draft, exclude={"artifact_hash"})}
    )
    write_immutable_json(path, artifact)
    return artifact


def _normalised_sample_id(
    record: RecordedGenerationResponse,
    *,
    run_id: str,
    case_id: str,
) -> str:
    """Map the provider request identity to the cross-case repeat identity."""

    prefix = f"{run_id}:{case_id}:"
    if not record.request.sample_id.startswith(prefix):
        raise ExternalDecisionSealError(
            f"Recorded sample ID does not match its run and case: {case_id}"
        )
    sample_id = record.request.sample_id.removeprefix(prefix)
    if not sample_id:
        raise ExternalDecisionSealError(f"Recorded sample ID has no repeat suffix: {case_id}")
    return sample_id


def _write_probe_set(
    root: Path,
    manifest: PublicBenchmarkManifest,
    probes: dict[str, ProbeExecutionSummary],
) -> None:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.probe_summary_set.v1",
        "benchmark_version": manifest.benchmark_version,
        "records": tuple(probes[case.case_id] for case in manifest.cases if case.case_id in probes),
    }
    draft = ProbeSummarySet.model_construct(_fields_set=set(content), **content, set_hash="0" * 64)
    probe_set = ProbeSummarySet.model_validate(
        {**content, "set_hash": model_content_hash(draft, exclude={"set_hash"})}
    )
    write_immutable_json(root / "probe_summaries.json", probe_set)


class _ResponseCollection(ContractModel):
    responses: tuple[RecordedGenerationResponse, ...]


def _load_records(path: Path) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExternalDecisionSealError(f"Could not read recorded responses: {path}") from error
    records: list[RecordedGenerationResponse] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(RecordedGenerationResponse.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ExternalDecisionSealError(
                f"Invalid recorded response at line {line_number}"
            ) from error
    if not records:
        raise ExternalDecisionSealError("Recorded response file is empty")
    return tuple(records)


def _load_json_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ExternalDecisionSealError(f"Could not load immutable artifact: {path}") from error


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must use UTC")


def _require_after(value: datetime, earlier: datetime, name: str) -> None:
    _require_utc(value, name)
    if value <= earlier:
        raise ExternalDecisionSealError(f"{name} must follow the decision-seal plan")
