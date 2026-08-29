# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Deterministic tabular helpers for the CSEDM study."""

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from socratic_tutor.csedm_study.features import CSEDMAdapterError


def parquet_bytes(schema: Any, rows: tuple[dict[str, object], ...]) -> bytes:
    """Serialise validated rows with the study's fixed Parquet settings."""

    table = pa.Table.from_pylist(list(rows), schema=schema)
    if not table.schema.equals(schema, check_metadata=True):
        raise CSEDMAdapterError("CSEDM table does not match its Arrow schema")
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", version="2.6", use_dictionary=False)
    return sink.getvalue().to_pybytes()
