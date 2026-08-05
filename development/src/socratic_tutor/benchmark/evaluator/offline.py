"""Evaluator-only offline criterion, scoring, and case publication workflow."""

from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json, write_immutable_jsonl
from socratic_tutor.benchmark.common import RelativePath, Sha256
from socratic_tutor.benchmark.evaluator.aggregation import (
    CaseAggregate,
    CaseComparison,
    create_case_aggregate,
)
from socratic_tutor.benchmark.evaluator.criterion import (
    CriterionExecutionStatus,
    CriterionRecord,
    CriterionRecordFactory,
    CriterionRecordIndex,
    create_normalized_criterion_execution,
)
from socratic_tutor.benchmark.evaluator.datasets import (
    publish_baseline_results,
    publish_case_aggregates,
    publish_criterion_records,
    publish_repeat_metrics,
)
from socratic_tutor.benchmark.evaluator.gate import CriterionGate
from socratic_tutor.benchmark.evaluator.models import EvaluatorBenchmarkManifest
from socratic_tutor.benchmark.evaluator.requests import CriterionChannelRequestBuilder
from socratic_tutor.benchmark.evaluator.scoring import (
    BaselineFitRole,
    BaselineResult,
    BenchmarkScorer,
    CalibrationStatus,
    RepeatEstimator,
    RepeatMetric,
    create_calibration_decision,
    create_constant_baseline_result,
    create_probe_only_baseline_result,
    fit_constant_prevalence,
)
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import FilesystemConditionCommitStore
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunPlan,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionStatus,
)
from socratic_tutor.benchmark.public.models import (
    BenchmarkSplit,
    PublicArtifactClass,
    PublicBenchmarkManifest,
)
from socratic_tutor.benchmark.public.offline import (
    ProbeSummarySet,
    load_initial_tracker_state,
    read_verified_public_file,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
    create_recorded_response,
    replay_recorded_response,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracking import update_tracker


class OfflineCriterionError(ValueError):
    """An offline criterion or scoring phase violates a frozen boundary."""


class OfflineCriterionCase(ContractModel):
    """Authored criterion response and normalized execution for one development case."""

    case_id: str = Field(min_length=1)
    criterion_response: str = Field(min_length=1)
    criterion_passed: int = Field(ge=0)
    criterion_failed: int = Field(ge=0)

    @model_validator(mode="after")
    def require_criterion_result(self) -> "OfflineCriterionCase":
        if self.criterion_passed + self.criterion_failed == 0:
            raise ValueError("Offline criterion requires at least one test result")
        return self


class OfflineCriterionPlan(ContractModel):
    """Typed evaluator-only inputs decided before a development rehearsal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.offline_criterion_plan.v1"] = (
        "benchmark.offline_criterion_plan.v1"
    )
    mode: Literal["development_offline", "reference_heldout"] = "development_offline"
    run_id: str = Field(min_length=1)
    model_route: ModelRoute
    sampling: SamplingConfig
    system_prompt_version: str = Field(min_length=1)
    system_prompt_ref: RelativePath
    system_prompt_sha256: Sha256
    tracker_version: str = Field(min_length=1)
    calibration_status: CalibrationStatus
    policy_threshold: float = Field(ge=0.0, le=1.0)
    calibration_outcomes: tuple[bool, ...] = Field(min_length=2)
    calibration_owner: str = Field(min_length=1)
    scorer_version: str = Field(min_length=1)
    constant_baseline_version: str = Field(min_length=1)
    probe_baseline_version: str = Field(min_length=1)
    aggregation_version: str = Field(min_length=1)
    created_at_utc: datetime
    cases: tuple[OfflineCriterionCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "OfflineCriterionPlan":
        _require_utc(self.created_at_utc, "Offline criterion-plan creation time")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Offline criterion case IDs must be unique")
        return self


class CriterionGenerationSummary(ContractModel):
    """Machine-readable result of criterion reveal and nested scoring."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.criterion_generation_summary.v1"] = (
        "benchmark.criterion_generation_summary.v1"
    )
    benchmark_version: str
    run_id: str
    criterion_count: int = Field(ge=1)
    demonstrated_count: int = Field(ge=0)
    baseline_count: int = Field(ge=1)
    repeat_metric_count: int = Field(ge=1)
    case_aggregate_count: int = Field(ge=1)
    calibration_decision_hash: Sha256
    criterion_publication_hash: Sha256
    baseline_publication_hash: Sha256
    repeat_publication_hash: Sha256
    aggregate_publication_hash: Sha256


_ESTIMATORS_BY_COMPARISON: dict[
    CaseComparison,
    tuple[RepeatEstimator, RepeatEstimator],
] = {
    CaseComparison.PRIMARY_VALID: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.PROBE_INFORMED,
    ),
    CaseComparison.UNRELATED_CONTROL: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.UNRELATED_PROBE,
    ),
    CaseComparison.CORRUPTED_CONTROL: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.CORRUPTED_PROBE,
    ),
    CaseComparison.BASELINE_CONSTANT: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.CONSTANT_PREVALENCE,
    ),
    CaseComparison.BASELINE_PROBE_ONLY: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.PROBE_ONLY,
    ),
}


def run_offline_criterion_phase(
    *,
    public_manifest: PublicBenchmarkManifest,
    evaluator_manifest: EvaluatorBenchmarkManifest,
    benchmark_root: Path,
    plan: OfflineCriterionPlan,
    output_root: Path,
) -> CriterionGenerationSummary:
    """Reveal outcomes only after the global seal, then score and aggregate them."""

    root = output_root.resolve()
    run_plan = DecisionRunPlan.model_validate_json((root / "run_plan.json").read_bytes())
    _validate_criterion_scope(public_manifest, evaluator_manifest, plan, run_plan)
    decision_root = root / "decision"
    conditions = FilesystemConditionCommitStore(decision_root)
    global_store = FilesystemGlobalDecisionSealStore(decision_root, conditions)
    global_seal = global_store.load()
    if global_seal is None:
        raise OfflineCriterionError("Criterion phase requires a published global decision seal")
    prompt = read_verified_public_file(
        benchmark_root,
        public_manifest,
        plan.system_prompt_ref,
        expected_class=PublicArtifactClass.PROMPT,
    )
    if file_sha256(prompt) != plan.system_prompt_sha256:
        raise OfflineCriterionError("Criterion plan system-prompt hash does not match content")
    probe_set = ProbeSummarySet.model_validate_json((root / "probe_summaries.json").read_bytes())
    if probe_set.benchmark_version != public_manifest.benchmark_version:
        raise OfflineCriterionError("Probe summaries belong to another benchmark")
    probe_by_case = {record.case_id: record for record in probe_set.records}
    case_inputs = {case.case_id: case for case in plan.cases}

    calibration = create_calibration_decision(
        status=plan.calibration_status,
        policy_threshold=plan.policy_threshold,
        calibration_task_family_hash=canonical_sha256(
            {"schema_id": "benchmark.offline_calibration_families.v1"}
        ),
        calibration_data_hash=canonical_sha256(plan.calibration_outcomes),
        tracker_version=plan.tracker_version,
        diagnostics_hash=canonical_sha256(
            {
                "schema_id": "benchmark.offline_calibration_diagnostics.v1",
                "status": plan.calibration_status,
            }
        ),
        decision_owner=plan.calibration_owner,
        decided_at_utc=plan.created_at_utc,
    )
    if (
        run_plan.provenance.calibration_decision_hash is not None
        and run_plan.provenance.calibration_decision_hash != calibration.decision_hash
    ):
        raise OfflineCriterionError("Calibration decision differs from the decision run plan")
    write_immutable_json(root / "calibration_decision.json", calibration)

    index = CriterionRecordIndex()
    gate = CriterionGate(
        global_store,
        conditions,
        run_plan,
        criterion_record_exists=index.exists,
    )
    request_builder_records: list[RecordedGenerationResponse] = []
    criterion_records: list[CriterionRecord] = []
    for ordinal, status in enumerate(global_seal.sample_statuses, start=1):
        if status.status is not SampleDecisionStatus.COMPLETE:
            continue
        access = gate.open(
            gate.issue(status.key),
            manifest_loader=lambda: evaluator_manifest,
        )
        case_input = case_inputs[status.key.case_id]
        request_time = global_seal.sealed_at_utc + timedelta(minutes=1, seconds=ordinal)
        generation_spec = GenerationRequestSpec(
            run_id=plan.run_id,
            sample_id=status.key.sample_id,
            system_prompt_version=plan.system_prompt_version,
            system_prompt=prompt.decode("utf-8"),
            system_prompt_sha256=plan.system_prompt_sha256,
            model_route=plan.model_route,
            sampling=_sampling_for_ordinal(plan.sampling, ordinal),
        )
        request = CriterionChannelRequestBuilder(benchmark_root, access).build_criterion(
            case_id=status.key.case_id,
            spec=generation_spec,
        )
        response = create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.AUTHORED,
            final_response=case_input.criterion_response,
            captured_at_utc=request_time,
        )
        replayed = replay_recorded_response(request, response)
        request_builder_records.append(response)
        execution = create_normalized_criterion_execution(
            status=CriterionExecutionStatus.COMPLETED,
            test_bundle_sha256=next(
                item.test_bundle_sha256
                for item in evaluator_manifest.criteria
                if item.case_id == status.key.case_id
            ),
            requested_at_utc=request_time,
            operation_id=f"offline-{status.key.case_id}-{status.key.sample_id}",
            passed=case_input.criterion_passed,
            failed=case_input.criterion_failed,
            exit_code=0 if case_input.criterion_failed == 0 else 1,
            stdout_sha256=canonical_sha256(
                {"stream": "stdout", "request_hash": request.request_hash}
            ),
            stderr_sha256=canonical_sha256(
                {"stream": "stderr", "request_hash": request.request_hash}
            ),
            executed_at_utc=request_time + timedelta(seconds=1),
        )
        record = CriterionRecordFactory().build(
            access,
            request=request,
            response=replayed,
            execution=execution,
            revealed_at_utc=request_time + timedelta(seconds=2),
        )
        criterion_records.append(index.add(record))

    if not criterion_records:
        raise OfflineCriterionError("Offline criterion phase has no complete decision samples")
    write_immutable_jsonl(
        root / "criterion" / "recorded_responses.jsonl",
        tuple(request_builder_records),
    )
    dataset_store = AtomicParquetDatasetStore(root / "datasets")
    criterion_publication = publish_criterion_records(
        dataset_store,
        tuple(criterion_records),
        global_seal=global_seal,
        published_at_utc=max(record.revealed_at_utc for record in criterion_records)
        + timedelta(seconds=1),
    )

    fit_split_hash = canonical_sha256(
        {
            "schema_id": "benchmark.offline_calibration_split.v1",
            "outcomes": plan.calibration_outcomes,
        }
    )
    prevalence_fit = fit_constant_prevalence(
        plan.calibration_outcomes,
        fit_role=BaselineFitRole.CALIBRATION,
        fit_split_hash=fit_split_hash,
        baseline_version=plan.constant_baseline_version,
        fitted_at_utc=plan.created_at_utc - timedelta(seconds=1),
    )
    baselines: list[BaselineResult] = []
    repeats: list[RepeatMetric] = []
    for record in criterion_records:
        baseline_time = record.revealed_at_utc + timedelta(seconds=1)
        probe = probe_by_case[record.key.case_id]
        sample_baselines = (
            create_constant_baseline_result(
                record.key,
                fit=prevalence_fit,
                calibration=calibration,
                created_at_utc=baseline_time,
            ),
            create_probe_only_baseline_result(
                record.key,
                fit_role=BaselineFitRole.CALIBRATION,
                fit_split_hash=fit_split_hash,
                baseline_version=plan.probe_baseline_version,
                initial_tracker_state=load_initial_tracker_state(
                    benchmark_root,
                    public_manifest,
                    case_id=record.key.case_id,
                ),
                probe_evidence=probe,
                tracker_update=update_tracker,
                calibration=calibration,
                created_at_utc=baseline_time,
            ),
        )
        baselines.extend(sample_baselines)
        commit_set = conditions.get(record.key)
        if commit_set is None:
            raise OfflineCriterionError("Criterion record has no sealed decision set")
        repeats.extend(
            BenchmarkScorer(calibration, scorer_version=plan.scorer_version).score(
                commit_set=commit_set,
                predictions=conditions.get_records(record.key),
                criterion=record,
                baselines=sample_baselines,
                created_at_utc=baseline_time + timedelta(seconds=1),
            )
        )

    baseline_publication = publish_baseline_results(
        dataset_store,
        tuple(baselines),
        global_seal=global_seal,
        published_at_utc=max(record.created_at_utc for record in baselines) + timedelta(seconds=1),
    )
    repeat_publication = publish_repeat_metrics(
        dataset_store,
        tuple(repeats),
        global_seal=global_seal,
        published_at_utc=max(record.created_at_utc for record in repeats) + timedelta(seconds=1),
    )
    aggregate_time = max(record.created_at_utc for record in repeats) + timedelta(seconds=2)
    aggregates = aggregate_case_metrics(
        public_manifest=public_manifest,
        run_id=plan.run_id,
        model_route_id=f"{plan.model_route.provider}/{plan.model_route.model}",
        repeats=tuple(repeats),
        planned_repeat_count=len(run_plan.planned_samples) // len(public_manifest.cases),
        aggregation_version=plan.aggregation_version,
        created_at_utc=aggregate_time,
    )
    aggregate_publication = publish_case_aggregates(
        dataset_store,
        aggregates,
        global_seal=global_seal,
        published_at_utc=aggregate_time + timedelta(seconds=1),
    )
    summary = CriterionGenerationSummary(
        benchmark_version=public_manifest.benchmark_version,
        run_id=plan.run_id,
        criterion_count=len(criterion_records),
        demonstrated_count=sum(
            record.demonstrated_performance is True for record in criterion_records
        ),
        baseline_count=len(baselines),
        repeat_metric_count=len(repeats),
        case_aggregate_count=len(aggregates),
        calibration_decision_hash=calibration.decision_hash,
        criterion_publication_hash=criterion_publication.publication_hash,
        baseline_publication_hash=baseline_publication.publication_hash,
        repeat_publication_hash=repeat_publication.publication_hash,
        aggregate_publication_hash=aggregate_publication.publication_hash,
    )
    write_immutable_json(root / "criterion_summary.json", summary)
    return summary


def _validate_criterion_scope(
    public: PublicBenchmarkManifest,
    evaluator: EvaluatorBenchmarkManifest,
    plan: OfflineCriterionPlan,
    run_plan: DecisionRunPlan,
) -> None:
    splits = {case.split for case in public.cases}
    if plan.mode == "development_offline" and splits != {BenchmarkSplit.DEVELOPMENT}:
        raise OfflineCriterionError(
            "Offline criterion accepts development cases only; held-out generation is disabled"
        )
    if plan.mode == "reference_heldout":
        if splits != {BenchmarkSplit.HELD_OUT}:
            raise OfflineCriterionError(
                "Reference criterion requires an all-held-out benchmark projection"
            )
        if plan.model_route != ModelRoute(
            provider="recorded_fixture",
            model="authored-deterministic",
        ):
            raise OfflineCriterionError(
                "Reference criterion requires the recorded_fixture/authored-deterministic route"
            )
    expected_cases = tuple(case.case_id for case in public.cases)
    if tuple(case.case_id for case in plan.cases) != expected_cases:
        raise OfflineCriterionError("Offline criterion plan must cover every case in order")
    if tuple(item.case_id for item in evaluator.criteria) != expected_cases:
        raise OfflineCriterionError("Evaluator and public projections cover different cases")
    if evaluator.public_projection_hash != public.projection_hash:
        raise OfflineCriterionError("Evaluator projection belongs to another public projection")
    if plan.run_id != run_plan.run_id or run_plan.benchmark_version != public.benchmark_version:
        raise OfflineCriterionError("Criterion plan and decision run identities differ")
    if plan.created_at_utc >= run_plan.created_at_utc:
        raise OfflineCriterionError("Calibration decision must precede the decision run plan")
    route_id = f"{plan.model_route.provider}/{plan.model_route.model}"
    if any(key.model_route_id != route_id for key in run_plan.planned_samples):
        raise OfflineCriterionError("Criterion route differs from the decision route")
    if run_plan.provenance.tracker_version != plan.tracker_version:
        raise OfflineCriterionError("Criterion tracker version differs from decision provenance")


def aggregate_case_metrics(
    *,
    public_manifest: PublicBenchmarkManifest,
    run_id: str,
    model_route_id: str,
    repeats: tuple[RepeatMetric, ...],
    planned_repeat_count: int,
    aggregation_version: str,
    created_at_utc: datetime,
) -> tuple[CaseAggregate, ...]:
    aggregates: list[CaseAggregate] = []
    for case in public_manifest.cases:
        case_metrics = tuple(record for record in repeats if record.key.case_id == case.case_id)
        by_sample = _metrics_by_sample(case_metrics)
        for comparison, (left_estimator, right_estimator) in _ESTIMATORS_BY_COMPARISON.items():
            left_losses: list[float] = []
            right_losses: list[float] = []
            source_hashes: list[Sha256] = []
            for sample_metrics in by_sample.values():
                left = sample_metrics[left_estimator]
                right = sample_metrics[right_estimator]
                source_hashes.extend((left.record_hash, right.record_hash))
                if left.loss_value is not None and right.loss_value is not None:
                    left_losses.append(left.loss_value)
                    right_losses.append(right.loss_value)
            eligible = len(left_losses)
            aggregates.append(
                create_case_aggregate(
                    benchmark_version=public_manifest.benchmark_version,
                    run_id=run_id,
                    case_id=case.case_id,
                    model_route_id=model_route_id,
                    comparison=comparison,
                    eligible_repeat_count=eligible,
                    planned_repeat_count=planned_repeat_count,
                    mean_left_loss=fmean(left_losses) if left_losses else None,
                    mean_right_loss=fmean(right_losses) if right_losses else None,
                    missingness_rule="exclude_missing_criterion_and_report_denominator",
                    aggregation_version=aggregation_version,
                    source_repeat_hashes=tuple(source_hashes),
                    created_at_utc=created_at_utc,
                )
            )
    return tuple(aggregates)


def _metrics_by_sample(
    records: tuple[RepeatMetric, ...],
) -> dict[str, dict[RepeatEstimator, RepeatMetric]]:
    grouped: dict[str, dict[RepeatEstimator, RepeatMetric]] = {}
    for record in records:
        grouped.setdefault(record.key.sample_id, {})[record.estimator] = record
    expected = set(RepeatEstimator)
    if not grouped or any(set(metrics) != expected for metrics in grouped.values()):
        raise OfflineCriterionError("Case aggregation requires every estimator per sample")
    return grouped


def _sampling_for_ordinal(sampling: SamplingConfig, ordinal: int) -> SamplingConfig:
    return sampling.model_copy(update={"seed": (sampling.seed + ordinal - 1) % (2**32)})


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
