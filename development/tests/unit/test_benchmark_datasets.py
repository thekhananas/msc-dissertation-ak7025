"""Atomic benchmark Parquet publication and lifecycle-order tests."""

from datetime import timedelta
from pathlib import Path

import pytest

from socratic_tutor.benchmark.evaluator import (
    BenchmarkScorer,
    CalibrationStatus,
    CaseComparison,
    CriterionRecord,
    RepeatEstimator,
    RepeatMetric,
    create_case_aggregate,
    publish_baseline_results,
    publish_case_aggregates,
    publish_criterion_records,
    publish_repeat_metrics,
    read_baseline_results,
    read_case_aggregates,
    read_criterion_records,
    read_repeat_metrics,
    repeat_metric_hash,
)
from socratic_tutor.benchmark.parquet import (
    AtomicParquetDatasetStore,
    BenchmarkDatasetKind,
    DatasetNotPublishedError,
    DatasetPublicationError,
    DatasetPublicationStorageError,
    dataset_file_name,
    dataset_manifest_file_name,
)
from socratic_tutor.benchmark.public import (
    ConditionCommitStore,
    FilesystemGlobalDecisionSealStore,
    GlobalDecisionSeal,
    publish_condition_predictions,
    read_condition_predictions,
)
from socratic_tutor.benchmark.public.prediction_arrow import prediction_arrow_schema
from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    prediction_arrow_row,
)
from tests.unit.test_benchmark_scoring import (
    baselines_fixture,
    calibration_fixture,
    criterion_fixture,
)


class _InterruptingStore(AtomicParquetDatasetStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._interrupt = True

    def _after_data_publish(self, dataset: BenchmarkDatasetKind) -> None:
        if self._interrupt and dataset is BenchmarkDatasetKind.CONDITION_PREDICTIONS:
            self._interrupt = False
            raise RuntimeError("simulated process interruption")


def _global_seal(root: Path, conditions: ConditionCommitStore) -> GlobalDecisionSeal:
    seal = FilesystemGlobalDecisionSealStore(root, conditions).load()
    assert seal is not None
    return seal


def _decision_fixture(
    tmp_path: Path,
) -> tuple[
    CriterionRecord,
    ConditionCommitStore,
    GlobalDecisionSeal,
    tuple[DecisionPredictionRecord, ...],
]:
    criterion, conditions, _ = criterion_fixture(tmp_path)
    seal = _global_seal(tmp_path, conditions)
    predictions = conditions.get_records(criterion.key)
    return criterion, conditions, seal, predictions


def _replace_repeat_source(
    record: RepeatMetric,
    source_record_hash: str,
) -> RepeatMetric:
    content = record.model_dump(mode="python", exclude={"record_hash"})
    content["source_record_hash"] = source_record_hash
    draft = record.model_copy(
        update={
            "source_record_hash": source_record_hash,
            "record_hash": "0" * 64,
        }
    )
    return RepeatMetric.model_validate({**content, "record_hash": repeat_metric_hash(draft)})


def test_publishes_and_reads_separate_lifecycle_ordered_datasets(tmp_path: Path) -> None:
    criterion, conditions, seal, predictions = _decision_fixture(tmp_path / "source")
    root = tmp_path / "published"
    store = AtomicParquetDatasetStore(root)

    decision_manifest = publish_condition_predictions(
        store,
        predictions,
        global_seal=seal,
        published_at_utc=seal.sealed_at_utc + timedelta(microseconds=1),
    )
    criterion_manifest = publish_criterion_records(
        store,
        (criterion,),
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=1),
    )
    calibration = calibration_fixture(CalibrationStatus.UNCALIBRATED_SCORE)
    baselines = baselines_fixture(criterion, calibration)
    baseline_manifest = publish_baseline_results(
        store,
        baselines,
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
    )
    commit_set = conditions.get(criterion.key)
    assert commit_set is not None
    metrics = BenchmarkScorer(calibration, scorer_version="scorer-v1").score(
        commit_set=commit_set,
        predictions=predictions,
        criterion=criterion,
        baselines=baselines,
        created_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
    )
    repeat_manifest = publish_repeat_metrics(
        store,
        metrics,
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=3),
    )
    dialogue = next(item for item in metrics if item.estimator is RepeatEstimator.DIALOGUE_ONLY)
    probe = next(item for item in metrics if item.estimator is RepeatEstimator.PROBE_INFORMED)
    assert dialogue.loss_value is not None
    assert probe.loss_value is not None
    aggregate = create_case_aggregate(
        benchmark_version=criterion.key.benchmark_version,
        run_id=criterion.key.run_id,
        case_id=criterion.key.case_id,
        model_route_id=criterion.key.model_route_id,
        comparison=CaseComparison.PRIMARY_VALID,
        eligible_repeat_count=1,
        planned_repeat_count=1,
        mean_left_loss=dialogue.loss_value,
        mean_right_loss=probe.loss_value,
        missingness_rule="complete paired repeat only",
        aggregation_version="case-mean-v1",
        source_repeat_hashes=(dialogue.record_hash, probe.record_hash),
        created_at_utc=criterion.revealed_at_utc + timedelta(seconds=3),
    )
    aggregate_manifest = publish_case_aggregates(
        store,
        (aggregate,),
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=4),
    )

    assert decision_manifest.row_count == 4
    assert criterion_manifest.row_count == 1
    assert baseline_manifest.row_count == 2
    assert repeat_manifest.row_count == 6
    assert aggregate_manifest.row_count == 1
    assert read_condition_predictions(store).table.num_rows == 4
    assert read_criterion_records(store).table.column("demonstrated_performance").to_pylist() == [
        False
    ]
    assert read_baseline_results(store).table.num_rows == 2
    assert read_repeat_metrics(store).table.num_rows == 6
    assert read_case_aggregates(store).table.column("case_effect").to_pylist() == [1.0]
    assert set(root.glob("*.parquet")) == {
        root / dataset_file_name(dataset) for dataset in BenchmarkDatasetKind
    }
    assert set(root.glob("*.publication.json")) == {
        root / dataset_manifest_file_name(dataset) for dataset in BenchmarkDatasetKind
    }
    assert (
        publish_condition_predictions(
            store,
            predictions,
            global_seal=seal,
            published_at_utc=criterion.revealed_at_utc + timedelta(seconds=5),
        )
        == decision_manifest
    )


def test_interrupted_data_file_is_not_published_and_retry_repairs_it(tmp_path: Path) -> None:
    _, _, seal, predictions = _decision_fixture(tmp_path / "source")
    root = tmp_path / "published"
    interrupted = _InterruptingStore(root)

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        publish_condition_predictions(
            interrupted,
            predictions,
            global_seal=seal,
            published_at_utc=seal.sealed_at_utc + timedelta(seconds=1),
        )

    assert (root / dataset_file_name(BenchmarkDatasetKind.CONDITION_PREDICTIONS)).exists()
    assert not (
        root / dataset_manifest_file_name(BenchmarkDatasetKind.CONDITION_PREDICTIONS)
    ).exists()
    with pytest.raises(DatasetNotPublishedError):
        read_condition_predictions(interrupted)

    repaired = AtomicParquetDatasetStore(root)
    manifest = publish_condition_predictions(
        repaired,
        predictions,
        global_seal=seal,
        published_at_utc=seal.sealed_at_utc + timedelta(seconds=2),
    )
    assert manifest.row_count == 4
    assert read_condition_predictions(repaired).manifest == manifest


def test_outcome_publication_requires_the_decision_manifest(tmp_path: Path) -> None:
    criterion, _, seal, _ = _decision_fixture(tmp_path / "source")
    store = AtomicParquetDatasetStore(tmp_path / "published")

    with pytest.raises(DatasetNotPublishedError):
        publish_criterion_records(
            store,
            (criterion,),
            global_seal=seal,
            published_at_utc=criterion.revealed_at_utc + timedelta(seconds=1),
        )


def test_rejects_non_finite_rows_before_writing(tmp_path: Path) -> None:
    _, _, seal, predictions = _decision_fixture(tmp_path / "source")
    rows = [prediction_arrow_row(record) for record in predictions]
    rows[0]["tracker_mastery_probability"] = float("nan")
    store = AtomicParquetDatasetStore(tmp_path / "published")

    with pytest.raises(DatasetPublicationError, match="non-finite"):
        store.publish(
            BenchmarkDatasetKind.CONDITION_PREDICTIONS,
            benchmark_version=seal.benchmark_version,
            run_id=seal.run_id,
            schema=prediction_arrow_schema(),
            rows=rows,
            source_record_hashes=tuple(record.record_hash for record in predictions),
            decision_seal_hash=seal.seal_hash,
            published_at_utc=seal.sealed_at_utc + timedelta(seconds=1),
        )


def test_reader_rejects_parquet_modified_after_publication(tmp_path: Path) -> None:
    _, _, seal, predictions = _decision_fixture(tmp_path / "source")
    root = tmp_path / "published"
    store = AtomicParquetDatasetStore(root)
    publish_condition_predictions(
        store,
        predictions,
        global_seal=seal,
        published_at_utc=seal.sealed_at_utc + timedelta(seconds=1),
    )
    parquet_path = root / dataset_file_name(BenchmarkDatasetKind.CONDITION_PREDICTIONS)
    with parquet_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(DatasetPublicationStorageError, match="digest"):
        read_condition_predictions(store)


def test_rejects_crossed_repeat_and_case_lineage(tmp_path: Path) -> None:
    criterion, conditions, seal, predictions = _decision_fixture(tmp_path / "source")
    store = AtomicParquetDatasetStore(tmp_path / "published")
    publish_condition_predictions(
        store,
        predictions,
        global_seal=seal,
        published_at_utc=seal.sealed_at_utc + timedelta(seconds=1),
    )
    publish_criterion_records(
        store,
        (criterion,),
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=1),
    )
    calibration = calibration_fixture(CalibrationStatus.UNCALIBRATED_SCORE)
    baselines = baselines_fixture(criterion, calibration)
    publish_baseline_results(
        store,
        baselines,
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
    )
    commit_set = conditions.get(criterion.key)
    assert commit_set is not None
    metrics = BenchmarkScorer(calibration, scorer_version="scorer-v1").score(
        commit_set=commit_set,
        predictions=predictions,
        criterion=criterion,
        baselines=baselines,
        created_at_utc=criterion.revealed_at_utc + timedelta(seconds=2),
    )
    dialogue = next(item for item in metrics if item.estimator is RepeatEstimator.DIALOGUE_ONLY)
    probe = next(item for item in metrics if item.estimator is RepeatEstimator.PROBE_INFORMED)
    unrelated = next(item for item in metrics if item.estimator is RepeatEstimator.UNRELATED_PROBE)
    crossed_metrics = tuple(
        _replace_repeat_source(item, dialogue.source_record_hash)
        if item.estimator is RepeatEstimator.PROBE_INFORMED
        else item
        for item in metrics
    )

    with pytest.raises(DatasetPublicationError, match="wrong estimator source"):
        publish_repeat_metrics(
            store,
            crossed_metrics,
            global_seal=seal,
            published_at_utc=criterion.revealed_at_utc + timedelta(seconds=3),
        )

    publish_repeat_metrics(
        store,
        metrics,
        global_seal=seal,
        published_at_utc=criterion.revealed_at_utc + timedelta(seconds=3),
    )
    assert dialogue.loss_value is not None
    assert probe.loss_value is not None
    aggregate = create_case_aggregate(
        benchmark_version=criterion.key.benchmark_version,
        run_id=criterion.key.run_id,
        case_id=criterion.key.case_id,
        model_route_id=criterion.key.model_route_id,
        comparison=CaseComparison.PRIMARY_VALID,
        eligible_repeat_count=1,
        planned_repeat_count=1,
        mean_left_loss=dialogue.loss_value,
        mean_right_loss=probe.loss_value,
        missingness_rule="complete paired repeat only",
        aggregation_version="case-mean-v1",
        source_repeat_hashes=(dialogue.record_hash, unrelated.record_hash),
        created_at_utc=criterion.revealed_at_utc + timedelta(seconds=3),
    )

    with pytest.raises(DatasetPublicationError, match="wrong estimator"):
        publish_case_aggregates(
            store,
            (aggregate,),
            global_seal=seal,
            published_at_utc=criterion.revealed_at_utc + timedelta(seconds=4),
        )
