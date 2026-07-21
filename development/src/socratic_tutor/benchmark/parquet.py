# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Atomic, manifest-gated Parquet publication shared by benchmark phases."""

import fcntl
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel


class BenchmarkDatasetKind(StrEnum):
    """Canonical physical datasets produced by the benchmark pipeline."""

    CONDITION_PREDICTIONS = "condition_predictions"
    CRITERION_RECORDS = "criterion_records"
    BASELINE_RESULTS = "baseline_results"
    REPEAT_METRICS = "repeat_metrics"
    CASE_AGGREGATES = "case_aggregates"


_DATASET_SCHEMA_IDS: dict[BenchmarkDatasetKind, str] = {
    BenchmarkDatasetKind.CONDITION_PREDICTIONS: "benchmark.condition_prediction.v1",
    BenchmarkDatasetKind.CRITERION_RECORDS: "benchmark.criterion_record.v1",
    BenchmarkDatasetKind.BASELINE_RESULTS: "benchmark.baseline_result.v1",
    BenchmarkDatasetKind.REPEAT_METRICS: "benchmark.repeat_metric.v1",
    BenchmarkDatasetKind.CASE_AGGREGATES: "benchmark.case_aggregate.v1",
}


class DatasetPublicationError(ValueError):
    """Dataset rows, schema, lineage, or lifecycle order are invalid."""


class DatasetNotPublishedError(DatasetPublicationError):
    """A final publication manifest does not exist for the requested dataset."""


class DatasetPublicationConflictError(DatasetPublicationError):
    """An immutable dataset name was already published with different content."""


class DatasetPublicationStorageError(RuntimeError):
    """A published dataset is missing, corrupt, or inconsistent with its manifest."""


class DatasetPublicationManifest(ContractModel):
    """Final marker that makes one immutable Parquet file readable."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.dataset_publication.v1"] = "benchmark.dataset_publication.v1"
    dataset: BenchmarkDatasetKind
    dataset_schema_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    parquet_file: str = Field(min_length=1)
    parquet_sha256: Sha256
    arrow_schema_hash: Sha256
    row_count: int = Field(ge=0)
    source_record_hashes: tuple[Sha256, ...]
    source_record_root_hash: Sha256
    decision_seal_hash: Sha256
    published_at_utc: datetime
    publication_hash: Sha256

    @model_validator(mode="after")
    def validate_publication(self) -> "DatasetPublicationManifest":
        _require_utc(self.published_at_utc, "Dataset publication time")
        if self.dataset_schema_id != dataset_schema_id(self.dataset):
            raise ValueError("Dataset kind and schema ID do not match")
        if self.parquet_file != dataset_file_name(self.dataset):
            raise ValueError("Dataset publication uses a non-canonical Parquet filename")
        if len(self.source_record_hashes) != self.row_count:
            raise ValueError("Every published row requires one source record hash")
        if len(set(self.source_record_hashes)) != len(self.source_record_hashes):
            raise ValueError("Published source record hashes must be unique")
        if self.source_record_root_hash != source_record_root_hash(self.source_record_hashes):
            raise ValueError("Source record root does not match the published records")
        if self.publication_hash != dataset_publication_hash(self):
            raise ValueError("Dataset publication hash does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class LoadedParquetDataset:
    """Verified manifest and Arrow table returned together."""

    manifest: DatasetPublicationManifest
    table: Any


def dataset_schema_id(dataset: BenchmarkDatasetKind) -> str:
    """Return the record schema required for one physical dataset."""

    return _DATASET_SCHEMA_IDS[dataset]


def dataset_file_name(dataset: BenchmarkDatasetKind) -> str:
    """Return the canonical Parquet filename for one dataset kind."""

    return f"{dataset.value}.parquet"


def dataset_manifest_file_name(dataset: BenchmarkDatasetKind) -> str:
    """Return the final marker filename written after its Parquet file."""

    return f"{dataset.value}.publication.json"


def source_record_root_hash(record_hashes: tuple[Sha256, ...]) -> Sha256:
    """Bind the exact ordered record identities represented by a dataset."""

    return canonical_sha256(
        {
            "schema_id": "benchmark.source_record_root.v1",
            "record_hashes": record_hashes,
        }
    )


def arrow_schema_hash(schema: Any) -> Sha256:
    """Hash field order, types, nullability, and schema metadata."""

    metadata = schema.metadata or {}
    return canonical_sha256(
        {
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in schema
            ],
            "metadata": {
                key.decode("utf-8"): value.decode("utf-8")
                for key, value in sorted(metadata.items())
            },
        }
    )


def dataset_publication_hash(publication: DatasetPublicationManifest) -> Sha256:
    """Hash every publication field except its self-referential digest."""

    return model_content_hash(publication, exclude={"publication_hash"})


class AtomicParquetDatasetStore:
    """Publish one Parquet file then atomically expose its final manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks_dir = self.root / ".locks"
        self._locks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def publish(
        self,
        dataset: BenchmarkDatasetKind,
        *,
        benchmark_version: str,
        run_id: str,
        schema: Any,
        rows: Sequence[Mapping[str, object]],
        source_record_hashes: tuple[Sha256, ...],
        decision_seal_hash: Sha256,
        published_at_utc: datetime,
    ) -> DatasetPublicationManifest:
        """Materialize validated rows and publish the manifest as the commit point."""

        _require_utc(published_at_utc, "Dataset publication time")
        _validate_arrow_schema(dataset, schema)
        frozen_rows = tuple(dict(row) for row in rows)
        _validate_rows(schema, frozen_rows, source_record_hashes)
        table = pa.Table.from_pylist(list(frozen_rows), schema=schema)
        if not table.schema.equals(schema, check_metadata=True):
            raise DatasetPublicationError("Materialized table does not match its Arrow schema")

        with self._dataset_lock(dataset):
            manifest_path = self._manifest_path(dataset)
            if manifest_path.exists():
                loaded = self.read(dataset, schema=schema)
                existing = loaded.manifest
                if (
                    existing.benchmark_version == benchmark_version
                    and existing.run_id == run_id
                    and existing.source_record_hashes == source_record_hashes
                    and existing.decision_seal_hash == decision_seal_hash
                ):
                    return existing
                raise DatasetPublicationConflictError(
                    f"Dataset was already published with different content: {dataset.value}"
                )

            parquet_path = self._parquet_path(dataset)
            temporary = parquet_path.with_name(f".{parquet_path.name}.{uuid4().hex}.tmp")
            try:
                pq.write_table(
                    table,
                    temporary,
                    compression="zstd",
                    version="2.6",
                    use_dictionary=True,
                )
                _fsync_file(temporary)
                written = pq.read_table(temporary)
                if not written.schema.equals(schema, check_metadata=True):
                    raise DatasetPublicationStorageError(
                        "Temporary Parquet schema changed during serialization"
                    )
                os.replace(temporary, parquet_path)
                _fsync_directory(self.root)
                self._after_data_publish(dataset)

                parquet_hash = file_sha256(parquet_path.read_bytes())
                content = {
                    "schema_version": 1,
                    "schema_id": "benchmark.dataset_publication.v1",
                    "dataset": dataset,
                    "dataset_schema_id": dataset_schema_id(dataset),
                    "benchmark_version": benchmark_version,
                    "run_id": run_id,
                    "parquet_file": dataset_file_name(dataset),
                    "parquet_sha256": parquet_hash,
                    "arrow_schema_hash": arrow_schema_hash(schema),
                    "row_count": table.num_rows,
                    "source_record_hashes": source_record_hashes,
                    "source_record_root_hash": source_record_root_hash(source_record_hashes),
                    "decision_seal_hash": decision_seal_hash,
                    "published_at_utc": published_at_utc,
                }
                draft = DatasetPublicationManifest.model_construct(
                    _fields_set=set(content),
                    **content,
                    publication_hash="0" * 64,
                )
                manifest = DatasetPublicationManifest.model_validate(
                    {
                        **content,
                        "publication_hash": dataset_publication_hash(draft),
                    }
                )
                self._atomic_write_manifest(manifest_path, manifest)
                return manifest
            finally:
                temporary.unlink(missing_ok=True)

    def read(self, dataset: BenchmarkDatasetKind, *, schema: Any) -> LoadedParquetDataset:
        """Read only when the final manifest and Parquet content both verify."""

        _validate_arrow_schema(dataset, schema)
        manifest_path = self._manifest_path(dataset)
        if not manifest_path.exists():
            raise DatasetNotPublishedError(f"Dataset is not published: {dataset.value}")
        try:
            manifest = DatasetPublicationManifest.model_validate_json(manifest_path.read_bytes())
        except (OSError, ValueError) as error:
            raise DatasetPublicationStorageError(
                f"Invalid dataset publication manifest: {dataset.value}"
            ) from error
        if manifest.dataset is not dataset:
            raise DatasetPublicationStorageError("Publication manifest names another dataset")
        if manifest.arrow_schema_hash != arrow_schema_hash(schema):
            raise DatasetPublicationStorageError("Publication Arrow schema hash differs")

        parquet_path = self._parquet_path(dataset)
        if not parquet_path.exists():
            raise DatasetPublicationStorageError("Published Parquet file is missing")
        try:
            if file_sha256(parquet_path.read_bytes()) != manifest.parquet_sha256:
                raise DatasetPublicationStorageError("Published Parquet digest differs")
            table = pq.read_table(parquet_path)
        except (OSError, ValueError) as error:
            raise DatasetPublicationStorageError(
                f"Could not read published Parquet dataset: {dataset.value}"
            ) from error
        if not table.schema.equals(schema, check_metadata=True):
            raise DatasetPublicationStorageError("Published Parquet schema differs")
        if table.num_rows != manifest.row_count:
            raise DatasetPublicationStorageError("Published Parquet row count differs")
        row_hashes = tuple(table.column("record_hash").to_pylist())
        if row_hashes != manifest.source_record_hashes:
            raise DatasetPublicationStorageError("Published row identities differ")
        return LoadedParquetDataset(manifest=manifest, table=table)

    def _after_data_publish(self, dataset: BenchmarkDatasetKind) -> None:
        """Fault-injection hook called before the publication manifest is written."""

    def _atomic_write_manifest(
        self,
        path: Path,
        manifest: DatasetPublicationManifest,
    ) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(manifest.model_dump_json(indent=2).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def _parquet_path(self, dataset: BenchmarkDatasetKind) -> Path:
        return self.root / dataset_file_name(dataset)

    def _manifest_path(self, dataset: BenchmarkDatasetKind) -> Path:
        return self.root / dataset_manifest_file_name(dataset)

    def _dataset_lock(self, dataset: BenchmarkDatasetKind) -> "_FileLock":
        return _FileLock(self._locks_dir / f"{dataset.value}.lock", self._lock)


class _FileLock:
    def __init__(self, path: Path, thread_lock: RLock) -> None:
        self._path = path
        self._thread_lock = thread_lock
        self._handle: Any = None

    def __enter__(self) -> None:
        self._thread_lock.acquire()
        try:
            self._handle = self._path.open("a", encoding="utf-8")
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
            raise

    def __exit__(self, *_args: object) -> None:
        try:
            if self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                self._handle.close()
        finally:
            self._thread_lock.release()


def _validate_arrow_schema(dataset: BenchmarkDatasetKind, schema: Any) -> None:
    metadata = schema.metadata or {}
    declared = metadata.get(b"schema_id", b"").decode("utf-8")
    if declared != dataset_schema_id(dataset):
        raise DatasetPublicationError("Arrow metadata has the wrong record schema ID")
    if "record_hash" not in schema.names:
        raise DatasetPublicationError("Published Arrow schema requires record_hash")


def _validate_rows(
    schema: Any,
    rows: tuple[dict[str, object], ...],
    source_record_hashes: tuple[Sha256, ...],
) -> None:
    if len(rows) != len(source_record_hashes):
        raise DatasetPublicationError("Every row requires one source record hash")
    if len(set(source_record_hashes)) != len(source_record_hashes):
        raise DatasetPublicationError("Source record hashes must be unique")
    expected_columns = tuple(schema.names)
    for index, row in enumerate(rows):
        if tuple(row) != expected_columns:
            raise DatasetPublicationError(
                f"Row {index} does not match the exact Arrow column order"
            )
        if row.get("record_hash") != source_record_hashes[index]:
            raise DatasetPublicationError(f"Row {index} has another source record hash")
        _reject_non_finite(row, row_number=index)


def _reject_non_finite(value: object, *, row_number: int) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DatasetPublicationError(f"Row {row_number} contains a non-finite value")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_non_finite(nested, row_number=row_number)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_non_finite(nested, row_number=row_number)


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
