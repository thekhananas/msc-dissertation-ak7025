"""Decision-safe publication of globally sealed condition predictions."""

from datetime import datetime

from socratic_tutor.benchmark.parquet import (
    AtomicParquetDatasetStore,
    BenchmarkDatasetKind,
    DatasetPublicationError,
    DatasetPublicationManifest,
    LoadedParquetDataset,
)
from socratic_tutor.benchmark.public.global_seal import GlobalDecisionSeal
from socratic_tutor.benchmark.public.prediction_arrow import prediction_arrow_schema
from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    prediction_arrow_row,
)


def publish_condition_predictions(
    store: AtomicParquetDatasetStore,
    records: tuple[DecisionPredictionRecord, ...],
    *,
    global_seal: GlobalDecisionSeal,
    published_at_utc: datetime,
) -> DatasetPublicationManifest:
    """Publish every decision record accounted for by the immutable global seal."""

    if published_at_utc < global_seal.sealed_at_utc:
        raise DatasetPublicationError("Decision publication precedes the global seal")
    expected_hashes = tuple(
        record_hash
        for status in global_seal.sample_statuses
        for record_hash in status.prediction_hashes
    )
    actual_hashes = tuple(record.record_hash for record in records)
    if actual_hashes != expected_hashes:
        raise DatasetPublicationError(
            "Prediction records do not match global decision-seal accounting"
        )
    cursor = 0
    for status in global_seal.sample_statuses:
        sample_records = records[cursor : cursor + len(status.prediction_hashes)]
        cursor += len(sample_records)
        if any(
            record.benchmark_version != global_seal.benchmark_version for record in sample_records
        ):
            raise DatasetPublicationError("Prediction belongs to another benchmark version")
        if any(record.run_id != global_seal.run_id for record in sample_records):
            raise DatasetPublicationError("Prediction belongs to another run")
        if any(record.committed_at_utc >= global_seal.sealed_at_utc for record in sample_records):
            raise DatasetPublicationError("Prediction does not precede the global seal")
        if any(
            (
                record.benchmark_version,
                record.run_id,
                record.case_id,
                record.sample_id,
                record.model_route_id,
            )
            != (
                status.key.benchmark_version,
                status.key.run_id,
                status.key.case_id,
                status.key.sample_id,
                status.key.model_route_id,
            )
            for record in sample_records
        ):
            raise DatasetPublicationError("Prediction identity differs from global accounting")
    _require_unique_prediction_keys(records)
    return store.publish(
        BenchmarkDatasetKind.CONDITION_PREDICTIONS,
        benchmark_version=global_seal.benchmark_version,
        run_id=global_seal.run_id,
        schema=prediction_arrow_schema(),
        rows=tuple(prediction_arrow_row(record) for record in records),
        source_record_hashes=actual_hashes,
        decision_seal_hash=global_seal.seal_hash,
        published_at_utc=published_at_utc,
    )


def read_condition_predictions(
    store: AtomicParquetDatasetStore,
) -> LoadedParquetDataset:
    """Read condition predictions only through their final publication manifest."""

    return store.read(
        BenchmarkDatasetKind.CONDITION_PREDICTIONS,
        schema=prediction_arrow_schema(),
    )


def _require_unique_prediction_keys(records: tuple[DecisionPredictionRecord, ...]) -> None:
    keys = tuple(
        (
            record.benchmark_version,
            record.run_id,
            record.case_id,
            record.sample_id,
            record.model_route_id,
            record.condition,
        )
        for record in records
    )
    if len(set(keys)) != len(keys):
        raise DatasetPublicationError("Prediction dataset contains duplicate primary keys")
