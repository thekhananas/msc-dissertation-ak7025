# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Typed boundary around PyArrow's untyped prediction-table API."""

from typing import Any

import pyarrow as pa

from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    prediction_arrow_row,
)
from socratic_tutor.contracts import ContractModel


class PredictionArrowSnapshot(ContractModel):
    """Plain-data proof that records materialize under the declared Arrow schema."""

    schema_id: str
    columns: tuple[str, ...]
    field_types: tuple[str, ...]
    nullable_fields: tuple[str, ...]
    row_count: int
    conditions: tuple[str, ...]
    uncertainty_null_count: int


def prediction_arrow_schema() -> Any:
    """Return the exact Arrow schema later used by the decision Parquet writer."""

    dictionary_string = pa.dictionary(pa.int8(), pa.string())
    return pa.schema(
        [
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            pa.field("benchmark_version", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("model_route_id", pa.string(), nullable=False),
            pa.field("condition", dictionary_string, nullable=False),
            pa.field("case_content_hash", pa.string(), nullable=False),
            pa.field("initial_state_hash", pa.string(), nullable=False),
            pa.field("public_record_hash", pa.string(), nullable=False),
            pa.field("public_evidence_hash", pa.string(), nullable=False),
            pa.field("additional_evidence_hash", pa.string(), nullable=True),
            pa.field("tracker_mastery_probability", pa.float64(), nullable=False),
            pa.field("tracker_uncertainty", pa.float64(), nullable=True),
            pa.field("tracker_uncertainty_method", dictionary_string, nullable=False),
            pa.field("tracker_misconception_probability", pa.float64(), nullable=True),
            pa.field("directive", dictionary_string, nullable=False),
            pa.field("decision_rationale", pa.string(), nullable=False),
            pa.field("policy_target_concept", pa.string(), nullable=False),
            pa.field("policy_propensity", pa.float64(), nullable=False),
            pa.field("tracker_version", pa.string(), nullable=False),
            pa.field("policy_version", pa.string(), nullable=False),
            pa.field("observation_schema", pa.string(), nullable=False),
            pa.field("input_hash", pa.string(), nullable=False),
            pa.field("committed_at_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("record_hash", pa.string(), nullable=False),
        ],
        metadata={b"schema_id": b"benchmark.condition_prediction.v1"},
    )


def prediction_arrow_snapshot(
    records: tuple[DecisionPredictionRecord, ...],
) -> PredictionArrowSnapshot:
    """Materialize records and return a typed summary for schema verification."""

    schema = prediction_arrow_schema()
    table = pa.Table.from_pylist(
        [prediction_arrow_row(record) for record in records],
        schema=schema,
    )
    if table.schema != schema:
        raise ValueError("Materialized prediction table does not match its Arrow schema")
    metadata = table.schema.metadata or {}
    schema_id = metadata.get(b"schema_id", b"").decode("ascii")
    return PredictionArrowSnapshot(
        schema_id=schema_id,
        columns=tuple(table.column_names),
        field_types=tuple(f"{field.name}: {field.type}" for field in table.schema),
        nullable_fields=tuple(field.name for field in table.schema if field.nullable),
        row_count=table.num_rows,
        conditions=tuple(table.column("condition").to_pylist()),
        uncertainty_null_count=table.column("tracker_uncertainty").null_count,
    )
