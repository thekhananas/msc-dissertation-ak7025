# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Outcome-bearing Arrow schemas and lifecycle-ordered Parquet publishers."""

from datetime import datetime
from typing import Any

import pyarrow as pa

from socratic_tutor.benchmark.evaluator.aggregation import CaseAggregate
from socratic_tutor.benchmark.evaluator.criterion import CriterionRecord
from socratic_tutor.benchmark.evaluator.scoring import (
    EXPECTED_BASELINES,
    EXPECTED_ESTIMATORS,
    BaselineResult,
    RepeatEstimator,
    RepeatMetric,
)
from socratic_tutor.benchmark.parquet import (
    AtomicParquetDatasetStore,
    BenchmarkDatasetKind,
    DatasetPublicationError,
    DatasetPublicationManifest,
    LoadedParquetDataset,
)
from socratic_tutor.benchmark.public import (
    BenchmarkSampleKey,
    GlobalDecisionSeal,
    SampleDecisionStatus,
)
from socratic_tutor.benchmark.public.prediction_arrow import prediction_arrow_schema


def criterion_arrow_schema() -> Any:
    """Return the exact schema for held-out transfer outcomes."""

    dictionary_string = pa.dictionary(pa.int8(), pa.string())
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            *_sample_key_fields(),
            pa.field("global_seal_hash", pa.string(), nullable=False),
            pa.field("condition_set_hash", pa.string(), nullable=False),
            pa.field("criterion_probe_id", pa.string(), nullable=False),
            pa.field("request_hash", pa.string(), nullable=False),
            pa.field("response_hash", pa.string(), nullable=True),
            pa.field("test_bundle_sha256", pa.string(), nullable=False),
            pa.field("rubric_sha256", pa.string(), nullable=False),
            pa.field("operation_id", pa.string(), nullable=True),
            pa.field("execution_status", dictionary_string, nullable=False),
            pa.field("execution_hash", pa.string(), nullable=False),
            pa.field("passed", pa.int32(), nullable=False),
            pa.field("failed", pa.int32(), nullable=False),
            pa.field("exit_code", pa.int32(), nullable=True),
            pa.field("timed_out", pa.bool_(), nullable=False),
            pa.field("resource_limited", pa.bool_(), nullable=False),
            pa.field("stdout_sha256", pa.string(), nullable=True),
            pa.field("stderr_sha256", pa.string(), nullable=True),
            pa.field("requested_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("executed_at_utc", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("demonstrated_performance", pa.bool_(), nullable=True),
            pa.field("missing_reason", dictionary_string, nullable=True),
            pa.field("revealed_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("record_hash", pa.string(), nullable=False),
        ],
        metadata={b"schema_id": b"benchmark.criterion_record.v1"},
    )


def baseline_arrow_schema() -> Any:
    """Return the exact schema for scoring-only baseline predictions."""

    dictionary_string = pa.dictionary(pa.int8(), pa.string())
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            *_sample_key_fields(),
            pa.field("baseline", dictionary_string, nullable=False),
            pa.field("fit_role", dictionary_string, nullable=False),
            pa.field("fit_split_hash", pa.string(), nullable=False),
            pa.field("baseline_version", pa.string(), nullable=False),
            pa.field("input_hash", pa.string(), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("policy_threshold", pa.float64(), nullable=False),
            pa.field("binary_decision", pa.bool_(), nullable=False),
            pa.field("calibration_decision_hash", pa.string(), nullable=False),
            pa.field("created_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("record_hash", pa.string(), nullable=False),
        ],
        metadata={b"schema_id": b"benchmark.baseline_result.v1"},
    )


def repeat_metric_arrow_schema() -> Any:
    """Return the exact schema for one estimator/sample scoring row."""

    dictionary_string = pa.dictionary(pa.int8(), pa.string())
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            *_sample_key_fields(),
            pa.field("estimator", dictionary_string, nullable=False),
            pa.field("estimator_version", pa.string(), nullable=False),
            pa.field("source_record_hash", pa.string(), nullable=False),
            pa.field("criterion_record_hash", pa.string(), nullable=False),
            pa.field("criterion_demonstrated_performance", pa.bool_(), nullable=True),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("binary_decision", pa.bool_(), nullable=False),
            pa.field("directive", dictionary_string, nullable=True),
            pa.field("selected_loss_name", dictionary_string, nullable=False),
            pa.field("loss_value", pa.float64(), nullable=True),
            pa.field("brier_loss", pa.float64(), nullable=True),
            pa.field("classification_error", pa.float64(), nullable=True),
            pa.field("unsafe_advancement", pa.bool_(), nullable=True),
            pa.field("false_confidence_acceptance", pa.bool_(), nullable=True),
            pa.field("false_confidence_detection", pa.bool_(), nullable=True),
            pa.field("action_disagreement_with_dialogue", pa.bool_(), nullable=True),
            pa.field("control_role", dictionary_string, nullable=False),
            pa.field("evidence_source_hash", pa.string(), nullable=True),
            pa.field("eligible", pa.bool_(), nullable=False),
            pa.field("missing_reason", dictionary_string, nullable=True),
            pa.field("latency_ms", pa.int64(), nullable=True),
            pa.field("input_tokens", pa.int64(), nullable=True),
            pa.field("output_tokens", pa.int64(), nullable=True),
            pa.field("retries", pa.int32(), nullable=True),
            pa.field("sandbox_time_ms", pa.int64(), nullable=True),
            pa.field("execution_failure", pa.bool_(), nullable=False),
            pa.field("cost_usd_micros", pa.int64(), nullable=True),
            pa.field("calibration_decision_hash", pa.string(), nullable=False),
            pa.field("scorer_version", pa.string(), nullable=False),
            pa.field("created_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("record_hash", pa.string(), nullable=False),
        ],
        metadata={b"schema_id": b"benchmark.repeat_metric.v1"},
    )


def case_aggregate_arrow_schema() -> Any:
    """Return the exact schema kept physically separate from repeat metrics."""

    dictionary_string = pa.dictionary(pa.int8(), pa.string())
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            pa.field("benchmark_version", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("model_route_id", pa.string(), nullable=False),
            pa.field("comparison", dictionary_string, nullable=False),
            pa.field("left_estimator", dictionary_string, nullable=False),
            pa.field("right_estimator", dictionary_string, nullable=False),
            pa.field("eligible_repeat_count", pa.int32(), nullable=False),
            pa.field("planned_repeat_count", pa.int32(), nullable=False),
            pa.field("mean_left_loss", pa.float64(), nullable=True),
            pa.field("mean_right_loss", pa.float64(), nullable=True),
            pa.field("case_effect", pa.float64(), nullable=True),
            pa.field("missing_repeat_count", pa.int32(), nullable=False),
            pa.field("missingness_rule", pa.string(), nullable=False),
            pa.field("aggregation_version", pa.string(), nullable=False),
            pa.field(
                "source_repeat_hashes",
                pa.list_(pa.field("element", pa.string(), nullable=True)),
                nullable=False,
            ),
            pa.field("created_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("record_hash", pa.string(), nullable=False),
        ],
        metadata={b"schema_id": b"benchmark.case_aggregate.v1"},
    )


def publish_criterion_records(
    store: AtomicParquetDatasetStore,
    records: tuple[CriterionRecord, ...],
    *,
    global_seal: GlobalDecisionSeal,
    published_at_utc: datetime,
) -> DatasetPublicationManifest:
    """Publish exactly one outcome or typed missingness row per complete sample."""

    _require_decision_publication(store, global_seal)
    validated = tuple(
        CriterionRecord.model_validate(record.model_dump(mode="python")) for record in records
    )
    expected_keys = _complete_sample_keys(global_seal)
    if tuple(record.key for record in validated) != expected_keys:
        raise DatasetPublicationError("Criterion rows do not match complete sealed samples")
    statuses = {
        status.key: status
        for status in global_seal.sample_statuses
        if status.status is SampleDecisionStatus.COMPLETE
    }
    for record in validated:
        status = statuses[record.key]
        if (
            record.global_seal_hash != global_seal.seal_hash
            or record.condition_set_hash != status.condition_set_hash
        ):
            raise DatasetPublicationError("Criterion row has the wrong decision barrier")
        if record.revealed_at_utc <= global_seal.sealed_at_utc:
            raise DatasetPublicationError("Criterion row does not follow the global seal")
    _require_publication_after(
        published_at_utc,
        tuple(record.revealed_at_utc for record in validated),
        "criterion reveal",
    )
    return store.publish(
        BenchmarkDatasetKind.CRITERION_RECORDS,
        benchmark_version=global_seal.benchmark_version,
        run_id=global_seal.run_id,
        schema=criterion_arrow_schema(),
        rows=tuple(criterion_arrow_row(record) for record in validated),
        source_record_hashes=tuple(record.record_hash for record in validated),
        decision_seal_hash=global_seal.seal_hash,
        published_at_utc=published_at_utc,
    )


def publish_baseline_results(
    store: AtomicParquetDatasetStore,
    records: tuple[BaselineResult, ...],
    *,
    global_seal: GlobalDecisionSeal,
    published_at_utc: datetime,
) -> DatasetPublicationManifest:
    """Publish two scoring-only baselines per complete sample after criterion publication."""

    _require_decision_publication(store, global_seal)
    _require_publication(
        store,
        BenchmarkDatasetKind.CRITERION_RECORDS,
        criterion_arrow_schema(),
        global_seal,
    )
    validated = tuple(
        BaselineResult.model_validate(record.model_dump(mode="python")) for record in records
    )
    expected = tuple(
        (key, baseline)
        for key in _complete_sample_keys(global_seal)
        for baseline in EXPECTED_BASELINES
    )
    if tuple((record.key, record.baseline) for record in validated) != expected:
        raise DatasetPublicationError("Baseline rows do not match complete sealed samples")
    _require_publication_after(
        published_at_utc,
        tuple(record.created_at_utc for record in validated),
        "baseline creation",
    )
    return store.publish(
        BenchmarkDatasetKind.BASELINE_RESULTS,
        benchmark_version=global_seal.benchmark_version,
        run_id=global_seal.run_id,
        schema=baseline_arrow_schema(),
        rows=tuple(baseline_arrow_row(record) for record in validated),
        source_record_hashes=tuple(record.record_hash for record in validated),
        decision_seal_hash=global_seal.seal_hash,
        published_at_utc=published_at_utc,
    )


def publish_repeat_metrics(
    store: AtomicParquetDatasetStore,
    records: tuple[RepeatMetric, ...],
    *,
    global_seal: GlobalDecisionSeal,
    published_at_utc: datetime,
) -> DatasetPublicationManifest:
    """Publish exact six-estimator score sets after criterion and baseline datasets."""

    decision = _require_decision_publication(store, global_seal)
    criteria = _require_publication(
        store,
        BenchmarkDatasetKind.CRITERION_RECORDS,
        criterion_arrow_schema(),
        global_seal,
    )
    baselines = _require_publication(
        store,
        BenchmarkDatasetKind.BASELINE_RESULTS,
        baseline_arrow_schema(),
        global_seal,
    )
    validated = tuple(
        RepeatMetric.model_validate(record.model_dump(mode="python")) for record in records
    )
    expected = tuple(
        (key, estimator)
        for key in _complete_sample_keys(global_seal)
        for estimator in EXPECTED_ESTIMATORS
    )
    if tuple((record.key, record.estimator) for record in validated) != expected:
        raise DatasetPublicationError("Repeat metrics do not match complete sealed samples")
    prediction_sources = {
        (_row_sample_key(row), str(row["condition"])): str(row["record_hash"])
        for row in decision.table.to_pylist()
    }
    criterion_sources = {
        _row_sample_key(row): str(row["record_hash"]) for row in criteria.table.to_pylist()
    }
    baseline_sources = {
        (_row_sample_key(row), str(row["baseline"])): str(row["record_hash"])
        for row in baselines.table.to_pylist()
    }
    for record in validated:
        key = _sample_key_values(record.key)
        expected_source = _repeat_estimator_source(
            record.estimator,
            key=key,
            prediction_sources=prediction_sources,
            baseline_sources=baseline_sources,
        )
        if record.source_record_hash != expected_source:
            raise DatasetPublicationError("Repeat metric has the wrong estimator source")
        if record.criterion_record_hash != criterion_sources.get(key):
            raise DatasetPublicationError("Repeat metric has the wrong criterion source")
    _require_publication_after(
        published_at_utc,
        tuple(record.created_at_utc for record in validated),
        "repeat scoring",
    )
    return store.publish(
        BenchmarkDatasetKind.REPEAT_METRICS,
        benchmark_version=global_seal.benchmark_version,
        run_id=global_seal.run_id,
        schema=repeat_metric_arrow_schema(),
        rows=tuple(repeat_metric_arrow_row(record) for record in validated),
        source_record_hashes=tuple(record.record_hash for record in validated),
        decision_seal_hash=global_seal.seal_hash,
        published_at_utc=published_at_utc,
    )


def publish_case_aggregates(
    store: AtomicParquetDatasetStore,
    records: tuple[CaseAggregate, ...],
    *,
    global_seal: GlobalDecisionSeal,
    published_at_utc: datetime,
) -> DatasetPublicationManifest:
    """Publish case rows only after their repeat-level source dataset exists."""

    repeats = _require_publication(
        store,
        BenchmarkDatasetKind.REPEAT_METRICS,
        repeat_metric_arrow_schema(),
        global_seal,
    )
    validated = tuple(
        CaseAggregate.model_validate(record.model_dump(mode="python")) for record in records
    )
    _require_same_run(validated, global_seal)
    keys = tuple(
        (
            record.benchmark_version,
            record.run_id,
            record.case_id,
            record.model_route_id,
            record.comparison,
        )
        for record in validated
    )
    if len(set(keys)) != len(keys):
        raise DatasetPublicationError("Case aggregate dataset has duplicate primary keys")
    repeat_rows = {str(row["record_hash"]): row for row in repeats.table.to_pylist()}
    for record in validated:
        _validate_case_sources(record, repeat_rows)
    _require_publication_after(
        published_at_utc,
        tuple(record.created_at_utc for record in validated),
        "case aggregation",
    )
    return store.publish(
        BenchmarkDatasetKind.CASE_AGGREGATES,
        benchmark_version=global_seal.benchmark_version,
        run_id=global_seal.run_id,
        schema=case_aggregate_arrow_schema(),
        rows=tuple(case_aggregate_arrow_row(record) for record in validated),
        source_record_hashes=tuple(record.record_hash for record in validated),
        decision_seal_hash=global_seal.seal_hash,
        published_at_utc=published_at_utc,
    )


def read_criterion_records(store: AtomicParquetDatasetStore) -> LoadedParquetDataset:
    return store.read(BenchmarkDatasetKind.CRITERION_RECORDS, schema=criterion_arrow_schema())


def read_baseline_results(store: AtomicParquetDatasetStore) -> LoadedParquetDataset:
    return store.read(BenchmarkDatasetKind.BASELINE_RESULTS, schema=baseline_arrow_schema())


def read_repeat_metrics(store: AtomicParquetDatasetStore) -> LoadedParquetDataset:
    return store.read(BenchmarkDatasetKind.REPEAT_METRICS, schema=repeat_metric_arrow_schema())


def read_case_aggregates(store: AtomicParquetDatasetStore) -> LoadedParquetDataset:
    return store.read(BenchmarkDatasetKind.CASE_AGGREGATES, schema=case_aggregate_arrow_schema())


def criterion_arrow_row(record: CriterionRecord) -> dict[str, object]:
    execution = record.execution
    return {
        "schema_version": record.schema_version,
        "schema_id": record.schema_id,
        **_sample_key_row(record.key),
        "global_seal_hash": record.global_seal_hash,
        "condition_set_hash": record.condition_set_hash,
        "criterion_probe_id": record.criterion_probe_id,
        "request_hash": record.request_hash,
        "response_hash": record.response_hash,
        "test_bundle_sha256": record.test_bundle_sha256,
        "rubric_sha256": record.rubric_sha256,
        "operation_id": execution.operation_id,
        "execution_status": execution.status.value,
        "execution_hash": execution.execution_hash,
        "passed": execution.passed,
        "failed": execution.failed,
        "exit_code": execution.exit_code,
        "timed_out": execution.timed_out,
        "resource_limited": execution.resource_limited,
        "stdout_sha256": execution.stdout_sha256,
        "stderr_sha256": execution.stderr_sha256,
        "requested_at_utc": execution.requested_at_utc,
        "executed_at_utc": execution.executed_at_utc,
        "demonstrated_performance": record.demonstrated_performance,
        "missing_reason": record.missing_reason.value if record.missing_reason else None,
        "revealed_at_utc": record.revealed_at_utc,
        "record_hash": record.record_hash,
    }


def baseline_arrow_row(record: BaselineResult) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "schema_id": record.schema_id,
        **_sample_key_row(record.key),
        "baseline": record.baseline.value,
        "fit_role": record.fit_role.value,
        "fit_split_hash": record.fit_split_hash,
        "baseline_version": record.baseline_version,
        "input_hash": record.input_hash,
        "score": record.score,
        "policy_threshold": record.policy_threshold,
        "binary_decision": record.binary_decision,
        "calibration_decision_hash": record.calibration_decision_hash,
        "created_at_utc": record.created_at_utc,
        "record_hash": record.record_hash,
    }

def repeat_metric_arrow_row(record: RepeatMetric) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "schema_id": record.schema_id,
        **_sample_key_row(record.key),
        "estimator": record.estimator.value,
        "estimator_version": record.estimator_version,
        "source_record_hash": record.source_record_hash,
        "criterion_record_hash": record.criterion_record_hash,
        "criterion_demonstrated_performance": record.criterion_demonstrated_performance,
        "score": record.score,
        "binary_decision": record.binary_decision,
        "directive": record.directive.value if record.directive else None,
        "selected_loss_name": record.selected_loss_name.value,
        "loss_value": record.loss_value,
        "brier_loss": record.brier_loss,
        "classification_error": record.classification_error,
        "unsafe_advancement": record.unsafe_advancement,
        "false_confidence_acceptance": record.false_confidence_acceptance,
        "false_confidence_detection": record.false_confidence_detection,
        "action_disagreement_with_dialogue": record.action_disagreement_with_dialogue,
        "control_role": record.control_role.value,
        "evidence_source_hash": record.evidence_source_hash,
        "eligible": record.eligible,
        "missing_reason": record.missing_reason.value if record.missing_reason else None,
        "latency_ms": record.latency_ms,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "retries": record.retries,
        "sandbox_time_ms": record.sandbox_time_ms,
        "execution_failure": record.execution_failure,
        "cost_usd_micros": record.cost_usd_micros,
        "calibration_decision_hash": record.calibration_decision_hash,
        "scorer_version": record.scorer_version,
        "created_at_utc": record.created_at_utc,
        "record_hash": record.record_hash,
    }


def case_aggregate_arrow_row(record: CaseAggregate) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "schema_id": record.schema_id,
        "benchmark_version": record.benchmark_version,
        "run_id": record.run_id,
        "case_id": record.case_id,
        "model_route_id": record.model_route_id,
        "comparison": record.comparison.value,
        "left_estimator": record.left_estimator.value,
        "right_estimator": record.right_estimator.value,
        "eligible_repeat_count": record.eligible_repeat_count,
        "planned_repeat_count": record.planned_repeat_count,
        "mean_left_loss": record.mean_left_loss,
        "mean_right_loss": record.mean_right_loss,
        "case_effect": record.case_effect,
        "missing_repeat_count": record.missing_repeat_count,
        "missingness_rule": record.missingness_rule,
        "aggregation_version": record.aggregation_version,
        "source_repeat_hashes": list(record.source_repeat_hashes),
        "created_at_utc": record.created_at_utc,
        "record_hash": record.record_hash,
    }


def _sample_key_fields() -> list[Any]:
    return [
        pa.field("benchmark_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("case_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("model_route_id", pa.string(), nullable=False),
    ]


def _sample_key_row(key: BenchmarkSampleKey) -> dict[str, object]:
    return {
        "benchmark_version": key.benchmark_version,
        "run_id": key.run_id,
        "case_id": key.case_id,
        "sample_id": key.sample_id,
        "model_route_id": key.model_route_id,
    }


def _sample_key_values(key: BenchmarkSampleKey) -> tuple[str, str, str, str, str]:
    return (
        key.benchmark_version,
        key.run_id,
        key.case_id,
        key.sample_id,
        key.model_route_id,
    )


def _row_sample_key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row["benchmark_version"]),
        str(row["run_id"]),
        str(row["case_id"]),
        str(row["sample_id"]),
        str(row["model_route_id"]),
    )


def _repeat_estimator_source(
    estimator: RepeatEstimator,
    *,
    key: tuple[str, str, str, str, str],
    prediction_sources: dict[tuple[tuple[str, str, str, str, str], str], str],
    baseline_sources: dict[tuple[tuple[str, str, str, str, str], str], str],
) -> str | None:
    if estimator is RepeatEstimator.CONSTANT_PREVALENCE:
        return baseline_sources.get((key, "constant_prevalence"))
    if estimator is RepeatEstimator.PROBE_ONLY:
        return baseline_sources.get((key, "probe_only"))
    return prediction_sources.get((key, estimator.value))


def _validate_case_sources(
    aggregate: CaseAggregate,
    repeat_rows: dict[str, dict[str, object]],
) -> None:
    grouped: dict[str, set[str]] = {}
    allowed_estimators = {
        aggregate.left_estimator.value,
        aggregate.right_estimator.value,
    }
    for source_hash in aggregate.source_repeat_hashes:
        row = repeat_rows.get(source_hash)
        if row is None:
            raise DatasetPublicationError("Case aggregate references an unpublished repeat metric")
        if (
            str(row["benchmark_version"]) != aggregate.benchmark_version
            or str(row["run_id"]) != aggregate.run_id
            or str(row["case_id"]) != aggregate.case_id
            or str(row["model_route_id"]) != aggregate.model_route_id
        ):
            raise DatasetPublicationError("Case aggregate references another case or route")
        estimator = str(row["estimator"])
        if estimator not in allowed_estimators:
            raise DatasetPublicationError("Case aggregate references the wrong estimator")
        grouped.setdefault(str(row["sample_id"]), set()).add(estimator)
    if any(estimators != allowed_estimators for estimators in grouped.values()):
        raise DatasetPublicationError("Case aggregate repeat sources are not paired")
    if aggregate.eligible_repeat_count > len(grouped):
        raise DatasetPublicationError("Case aggregate has more eligible repeats than sources")
    if len(grouped) > aggregate.planned_repeat_count:
        raise DatasetPublicationError("Case aggregate has more sources than planned repeats")


def _complete_sample_keys(seal: GlobalDecisionSeal) -> tuple[BenchmarkSampleKey, ...]:
    return tuple(
        status.key
        for status in seal.sample_statuses
        if status.status is SampleDecisionStatus.COMPLETE
    )


def _require_decision_publication(
    store: AtomicParquetDatasetStore,
    seal: GlobalDecisionSeal,
) -> LoadedParquetDataset:
    decision = store.read(
        BenchmarkDatasetKind.CONDITION_PREDICTIONS,
        schema=prediction_arrow_schema(),
    )
    _require_matching_seal(decision, seal)
    expected_hashes = tuple(
        item for status in seal.sample_statuses for item in status.prediction_hashes
    )
    if decision.manifest.source_record_hashes != expected_hashes:
        raise DatasetPublicationError("Decision publication differs from global accounting")
    return decision


def _require_publication(
    store: AtomicParquetDatasetStore,
    dataset: BenchmarkDatasetKind,
    schema: Any,
    seal: GlobalDecisionSeal,
) -> LoadedParquetDataset:
    loaded = store.read(dataset, schema=schema)
    _require_matching_seal(loaded, seal)
    return loaded


def _require_matching_seal(
    loaded: LoadedParquetDataset,
    seal: GlobalDecisionSeal,
) -> None:
    manifest = loaded.manifest
    if (
        manifest.benchmark_version != seal.benchmark_version
        or manifest.run_id != seal.run_id
        or manifest.decision_seal_hash != seal.seal_hash
    ):
        raise DatasetPublicationError("Published dataset belongs to another decision seal")


def _require_same_run(
    records: tuple[CaseAggregate, ...],
    seal: GlobalDecisionSeal,
) -> None:
    if any(
        record.benchmark_version != seal.benchmark_version or record.run_id != seal.run_id
        for record in records
    ):
        raise DatasetPublicationError("Case aggregate belongs to another benchmark run")


def _require_publication_after(
    published_at_utc: datetime,
    source_times: tuple[datetime, ...],
    source_name: str,
) -> None:
    if source_times and published_at_utc < max(source_times):
        raise DatasetPublicationError(f"Dataset publication precedes {source_name}")
