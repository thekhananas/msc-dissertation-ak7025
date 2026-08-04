"""Deterministic scoring of one audited and replayed external benchmark run."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.calibration import UncalibratedDecisionReport
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.criterion import CriterionRecord
from socratic_tutor.benchmark.evaluator.datasets import (
    publish_baseline_results,
    publish_case_aggregates,
    publish_repeat_metrics,
    read_criterion_records,
)
from socratic_tutor.benchmark.evaluator.offline import aggregate_case_metrics
from socratic_tutor.benchmark.evaluator.scoring import (
    BaselineFitRole,
    BaselineResult,
    BenchmarkScorer,
    PrimaryMetric,
    RepeatMetric,
    create_fixed_constant_baseline_result,
    create_probe_only_baseline_result,
)
from socratic_tutor.benchmark.external_replay import ExternalReplayPlan, ExternalReplayReport
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.datasets import read_condition_predictions
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunPlan,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionStatus,
    SampleDecisionStatusRecord,
)
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.public.offline import (
    ProbeSummarySet,
    load_initial_tracker_state,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracking import update_tracker


class ExternalScoringError(ValueError):
    """Scoring sources differ from the frozen, audited external run."""


class ExternalScoringPlan(ContractModel):
    """Frozen identities and versions used for deterministic external scoring."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_scoring_plan.v1"] = "benchmark.external_scoring_plan.v1"
    run_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    replay_report_hash: Sha256
    global_seal_hash: Sha256
    public_projection_hash: Sha256
    analysis_specification_hash: Sha256
    calibration_report_hash: Sha256
    calibration_decision_hash: Sha256
    probe_summary_set_hash: Sha256
    scorer_version: str = Field(min_length=1)
    constant_baseline_version: str = Field(min_length=1)
    constant_baseline_score: float = Field(default=0.5, ge=0.0, le=1.0)
    probe_baseline_version: str = Field(min_length=1)
    aggregation_version: str = Field(min_length=1)
    scoring_code_revision: str = Field(min_length=1)
    scoring_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> ExternalScoringPlan:
        _require_utc(self.created_at_utc)
        if self.constant_baseline_score != 0.5:
            raise ValueError("External v1 scoring requires the preregistered 0.5 baseline")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("External scoring-plan hash does not match its content")
        return self


class ExternalScoringSummary(ContractModel):
    """Publication identities and denominators, without effect interpretation."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_scoring_summary.v1"] = (
        "benchmark.external_scoring_summary.v1"
    )
    run_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    scoring_plan_hash: Sha256
    criterion_count: int = Field(ge=1)
    eligible_criterion_count: int = Field(ge=0)
    missing_criterion_count: int = Field(ge=0)
    missing_case_ids: tuple[str, ...]
    baseline_count: int = Field(ge=1)
    repeat_metric_count: int = Field(ge=1)
    eligible_repeat_metric_count: int = Field(ge=0)
    case_aggregate_count: int = Field(ge=1)
    eligible_case_aggregate_count: int = Field(ge=0)
    baseline_publication_hash: Sha256
    repeat_publication_hash: Sha256
    aggregate_publication_hash: Sha256
    selected_metric: Literal["paired_classification_error_difference"]
    effect_interpretation_status: Literal["not_computed"] = "not_computed"
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    gate_passed: Literal[True] = True
    summary_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> ExternalScoringSummary:
        _require_utc(self.completed_at_utc)
        if self.criterion_count != (self.eligible_criterion_count + self.missing_criterion_count):
            raise ValueError("External scoring criterion counts do not reconcile")
        if len(self.missing_case_ids) != self.missing_criterion_count or len(
            set(self.missing_case_ids)
        ) != len(self.missing_case_ids):
            raise ValueError("External scoring missing-case identities do not reconcile")
        expected_counts = (
            self.baseline_count == self.criterion_count * 2
            and self.repeat_metric_count == self.criterion_count * 6
            and self.eligible_repeat_metric_count == self.eligible_criterion_count * 6
            and self.case_aggregate_count == self.criterion_count * 5
            and self.eligible_case_aggregate_count == self.eligible_criterion_count * 5
        )
        if not expected_counts:
            raise ValueError("External scoring dataset counts do not reconcile")
        if self.summary_hash != model_content_hash(self, exclude={"summary_hash"}):
            raise ValueError("External scoring-summary hash does not match its content")
        return self


def run_external_scoring(
    *,
    replay_report_path: Path,
    analysis_specification_path: Path,
    calibration_report_path: Path,
    public_manifest_path: Path,
    benchmark_root: Path,
    seal_root: Path,
    pixi_lock_path: Path,
    scoring_code_revision: str,
    created_at_utc: datetime,
) -> ExternalScoringSummary:
    """Score immutable external records without provider or sandbox access."""

    _require_utc(created_at_utc)
    replay = _load_json(replay_report_path, ExternalReplayReport)
    replay_plan = _load_json(
        replay_report_path.resolve().parent / "external_replay_plan.json",
        ExternalReplayPlan,
    )
    specification = load_analysis_specification(analysis_specification_path)
    calibration_report = _load_json(calibration_report_path, UncalibratedDecisionReport)
    public = _load_json(public_manifest_path, PublicBenchmarkManifest)
    root = seal_root.resolve()
    probe_set = _load_json(root / "probe_summaries.json", ProbeSummarySet)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ExternalScoringError(f"Could not read Pixi lock: {pixi_lock_path}") from error

    conditions = FilesystemConditionCommitStore(root / "decision")
    global_seal = FilesystemGlobalDecisionSealStore(root / "decision", conditions).load()
    if global_seal is None:
        raise ExternalScoringError("External scoring requires a global decision seal")
    run_plan = _load_json(root / "run_plan.json", DecisionRunPlan)
    source_store = AtomicParquetDatasetStore(root / "datasets")
    condition_publication = read_condition_predictions(source_store).manifest
    criterion_publication = read_criterion_records(source_store).manifest
    _validate_sources(
        replay=replay,
        replay_plan=replay_plan,
        specification=specification,
        calibration_report=calibration_report,
        public=public,
        probe_set=probe_set,
        run_plan=run_plan,
        global_seal_hash=global_seal.seal_hash,
        global_seal_run_plan_hash=global_seal.run_plan_hash,
        condition_publication_hash=condition_publication.publication_hash,
        criterion_publication_hash=criterion_publication.publication_hash,
    )
    plan = _create_plan(
        replay=replay,
        replay_plan=replay_plan,
        specification=specification,
        calibration_report=calibration_report,
        public=public,
        probe_set=probe_set,
        pixi_lock_hash=pixi_lock_hash,
        scoring_code_revision=scoring_code_revision,
        created_at_utc=created_at_utc,
    )
    analysis_root = root / "analysis" / "scoring-v1"
    summary_path = analysis_root / "external_scoring_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path, ExternalScoringSummary)
        if summary.scoring_plan_hash != plan.plan_hash:
            raise ExternalScoringError("Existing scoring summary belongs to another plan")
        return summary
    write_immutable_json(analysis_root / "external_scoring_plan.json", plan)

    complete_statuses = tuple(
        status
        for status in global_seal.sample_statuses
        if status.status is SampleDecisionStatus.COMPLETE
    )
    if len(complete_statuses) != len(global_seal.sample_statuses):
        raise ExternalScoringError("External scoring requires all decision samples to be complete")
    criteria = _ordered_criteria(root / "criterion" / "records", complete_statuses)
    if created_at_utc <= max(record.revealed_at_utc for record in criteria):
        raise ExternalScoringError("External scoring must occur after every criterion reveal")
    probes = {record.case_id: record for record in probe_set.records}
    calibration = calibration_report.decision
    baselines: list[BaselineResult] = []
    repeats: list[RepeatMetric] = []
    scorer = BenchmarkScorer(calibration, scorer_version=plan.scorer_version)
    for criterion in criteria:
        key = criterion.key
        sample_baselines = (
            create_fixed_constant_baseline_result(
                key,
                score=plan.constant_baseline_score,
                baseline_version=plan.constant_baseline_version,
                calibration=calibration,
                created_at_utc=created_at_utc,
            ),
            create_probe_only_baseline_result(
                key,
                fit_role=BaselineFitRole.PREREGISTERED,
                fit_split_hash=calibration.calibration_data_hash,
                baseline_version=plan.probe_baseline_version,
                initial_tracker_state=load_initial_tracker_state(
                    benchmark_root,
                    public,
                    case_id=key.case_id,
                ),
                probe_evidence=probes[key.case_id],
                tracker_update=update_tracker,
                calibration=calibration,
                created_at_utc=created_at_utc,
            ),
        )
        baselines.extend(sample_baselines)
        commit_set = conditions.get(key)
        if commit_set is None:
            raise ExternalScoringError(f"Missing sealed conditions for {key.case_id}")
        repeats.extend(
            scorer.score(
                commit_set=commit_set,
                predictions=conditions.get_records(key),
                criterion=criterion,
                baselines=sample_baselines,
                created_at_utc=created_at_utc + timedelta(seconds=2),
            )
        )

    dataset_store = AtomicParquetDatasetStore(root / "datasets")
    baseline_publication = publish_baseline_results(
        dataset_store,
        tuple(baselines),
        global_seal=global_seal,
        published_at_utc=created_at_utc + timedelta(seconds=1),
    )
    repeat_publication = publish_repeat_metrics(
        dataset_store,
        tuple(repeats),
        global_seal=global_seal,
        published_at_utc=created_at_utc + timedelta(seconds=3),
    )
    model_route_ids = {status.key.model_route_id for status in complete_statuses}
    if len(model_route_ids) != 1:
        raise ExternalScoringError("External scoring supports one model route per run")
    aggregates = aggregate_case_metrics(
        public_manifest=public,
        run_id=run_plan.run_id,
        model_route_id=next(iter(model_route_ids)),
        repeats=tuple(repeats),
        planned_repeat_count=1,
        aggregation_version=specification.aggregation_version,
        created_at_utc=created_at_utc + timedelta(seconds=4),
    )
    aggregate_publication = publish_case_aggregates(
        dataset_store,
        aggregates,
        global_seal=global_seal,
        published_at_utc=created_at_utc + timedelta(seconds=5),
    )
    missing_case_ids = tuple(
        record.key.case_id for record in criteria if record.demonstrated_performance is None
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_scoring_summary.v1",
        "run_id": run_plan.run_id,
        "benchmark_version": public.benchmark_version,
        "scoring_plan_hash": plan.plan_hash,
        "criterion_count": len(criteria),
        "eligible_criterion_count": len(criteria) - len(missing_case_ids),
        "missing_criterion_count": len(missing_case_ids),
        "missing_case_ids": missing_case_ids,
        "baseline_count": len(baselines),
        "repeat_metric_count": len(repeats),
        "eligible_repeat_metric_count": sum(record.eligible for record in repeats),
        "case_aggregate_count": len(aggregates),
        "eligible_case_aggregate_count": sum(
            record.eligible_repeat_count > 0 for record in aggregates
        ),
        "baseline_publication_hash": baseline_publication.publication_hash,
        "repeat_publication_hash": repeat_publication.publication_hash,
        "aggregate_publication_hash": aggregate_publication.publication_hash,
        "selected_metric": calibration.primary_metric.value,
        "effect_interpretation_status": "not_computed",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc + timedelta(seconds=5),
        "gate_passed": True,
    }
    draft = ExternalScoringSummary.model_construct(
        _fields_set=set(content), **content, summary_hash="0" * 64
    )
    summary = ExternalScoringSummary.model_validate(
        {**content, "summary_hash": model_content_hash(draft, exclude={"summary_hash"})}
    )
    write_immutable_json(summary_path, summary)
    return summary


def _validate_sources(
    *,
    replay: ExternalReplayReport,
    replay_plan: ExternalReplayPlan,
    specification: AnalysisSpecification,
    calibration_report: UncalibratedDecisionReport,
    public: PublicBenchmarkManifest,
    probe_set: ProbeSummarySet,
    run_plan: DecisionRunPlan,
    global_seal_hash: Sha256,
    global_seal_run_plan_hash: Sha256,
    condition_publication_hash: Sha256,
    criterion_publication_hash: Sha256,
) -> None:
    specification_hash = analysis_specification_hash(specification)
    calibration = calibration_report.decision
    if not replay.gate_passed or replay.scored_dataset_status != "not_yet_created":
        raise ExternalScoringError("External scoring requires a passing pre-analysis replay")
    identities = {
        replay.run_id,
        run_plan.run_id,
        *(key.run_id for key in run_plan.planned_samples),
    }
    if len(identities) != 1:
        raise ExternalScoringError("External scoring sources belong to different runs")
    if replay.replay_plan_hash != replay_plan.plan_hash:
        raise ExternalScoringError("Replay report and replay plan identities differ")
    if replay_plan.global_seal_hash != global_seal_hash:
        raise ExternalScoringError("Replay report belongs to another global seal")
    if run_plan.plan_hash != global_seal_run_plan_hash:
        raise ExternalScoringError("Run plan differs from the globally sealed plan")
    if replay.condition_source_publication_hash != condition_publication_hash:
        raise ExternalScoringError("Replay condition source differs from the scoring dataset")
    if replay.criterion_source_publication_hash != criterion_publication_hash:
        raise ExternalScoringError("Replay criterion source differs from the scoring dataset")
    if public.benchmark_version != specification.benchmark_version:
        raise ExternalScoringError("Analysis specification belongs to another benchmark")
    if public.benchmark_version != run_plan.benchmark_version:
        raise ExternalScoringError("Public projection belongs to another run plan")
    if public.projection_hash != run_plan.provenance.public_manifest_hash:
        raise ExternalScoringError("Public projection hash differs from the sealed run")
    if probe_set.benchmark_version != public.benchmark_version:
        raise ExternalScoringError("Probe summaries belong to another benchmark")
    if {record.case_id for record in probe_set.records} != {case.case_id for case in public.cases}:
        raise ExternalScoringError("Probe summaries do not cover the public cases exactly")
    if calibration_report.benchmark_version != public.benchmark_version:
        raise ExternalScoringError("Calibration report belongs to another benchmark")
    if calibration_report.analysis_specification_hash != specification_hash:
        raise ExternalScoringError("Calibration report differs from the frozen analysis plan")
    if calibration.decision_hash != run_plan.provenance.calibration_decision_hash:
        raise ExternalScoringError("Calibration decision differs from the sealed run")
    if calibration.tracker_version != run_plan.provenance.tracker_version:
        raise ExternalScoringError("Calibration decision uses another tracker version")
    if calibration.policy_threshold != specification.policy_threshold:
        raise ExternalScoringError("Calibration threshold differs from the frozen analysis plan")
    if calibration.primary_metric is not PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE:
        raise ExternalScoringError("External v1 scoring requires the uncalibrated fallback metric")
    if calibration.primary_metric.value != specification.uncalibrated_fallback_metric:
        raise ExternalScoringError("Selected metric differs from the frozen fallback metric")
    expected_cases = tuple(case.case_id for case in public.cases)
    planned_cases = tuple(key.case_id for key in run_plan.planned_samples)
    if planned_cases != expected_cases:
        raise ExternalScoringError("External v1 scoring requires one ordered sample per case")
    if replay.condition_prediction_count != len(run_plan.planned_samples) * 4:
        raise ExternalScoringError("Replay prediction count differs from the scoring plan")
    if replay.criterion_record_count != len(run_plan.planned_samples):
        raise ExternalScoringError("Replay criterion count differs from the scoring plan")


def _create_plan(
    *,
    replay: ExternalReplayReport,
    replay_plan: ExternalReplayPlan,
    specification: AnalysisSpecification,
    calibration_report: UncalibratedDecisionReport,
    public: PublicBenchmarkManifest,
    probe_set: ProbeSummarySet,
    pixi_lock_hash: Sha256,
    scoring_code_revision: str,
    created_at_utc: datetime,
) -> ExternalScoringPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_scoring_plan.v1",
        "run_id": replay.run_id,
        "benchmark_version": public.benchmark_version,
        "replay_report_hash": replay.report_hash,
        "global_seal_hash": replay_plan.global_seal_hash,
        "public_projection_hash": public.projection_hash,
        "analysis_specification_hash": analysis_specification_hash(specification),
        "calibration_report_hash": calibration_report.report_hash,
        "calibration_decision_hash": calibration_report.decision.decision_hash,
        "probe_summary_set_hash": probe_set.set_hash,
        "scorer_version": "external-classification-scorer-v1",
        "constant_baseline_version": "constant-0.5-no-calibration-v1",
        "constant_baseline_score": 0.5,
        "probe_baseline_version": "probe-only-simple-v1",
        "aggregation_version": specification.aggregation_version,
        "scoring_code_revision": scoring_code_revision,
        "scoring_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ExternalScoringPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ExternalScoringPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _ordered_criteria(
    records_root: Path,
    statuses: tuple[SampleDecisionStatusRecord, ...],
) -> tuple[CriterionRecord, ...]:
    files = tuple(sorted(records_root.glob("*.json")))
    records = tuple(_load_json(path, CriterionRecord) for path in files)
    by_key = {record.key: record for record in records}
    if len(by_key) != len(records):
        raise ExternalScoringError("Criterion record set contains duplicate sample keys")
    try:
        ordered = tuple(by_key[status.key] for status in statuses)
    except KeyError as error:
        raise ExternalScoringError("Criterion records do not cover every sealed sample") from error
    if len(records) != len(ordered):
        raise ExternalScoringError("Criterion record set contains unexpected samples")
    return ordered


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ExternalScoringError(f"Could not verify external scoring source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("External scoring timestamp must be UTC")
