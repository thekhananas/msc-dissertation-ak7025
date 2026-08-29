"""Shared, read-only access to the frozen CSEDM archive tables."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass

from pydantic import Field

from socratic_tutor.contracts import ContractModel

PREDICT_COLUMNS = (
    "SubjectID",
    "ProblemID",
    "StartOrder",
    "FirstCorrect",
    "EverCorrect",
    "UsedHint",
    "Attempts",
)
MAIN_TABLE_COLUMNS = (
    "EventType",
    "EventID",
    "Order",
    "SubjectID",
    "ToolInstances",
    "CodeStateID",
    "ServerTimestamp",
    "ProblemID",
    "Correct",
)
CODE_STATE_COLUMNS = ("CodeStateID", "Code")
FOLD_COUNT = 10
UNAVAILABLE_SOURCE_VALUES = frozenset({"", "NA"})

type CsvRow = dict[str, str]
type TargetKey = tuple[str, str, str]


class CSEDMInventoryError(ValueError):
    """The supplied archive does not satisfy the frozen source contract."""


class TableInventory(ContractModel):
    """Aggregate structure of one source table."""

    member: str
    columns: tuple[str, ...]
    row_count: int = Field(ge=0)
    missing_by_column: dict[str, int]


@dataclass(frozen=True)
class LoadedTable:
    """Validated CSV rows and their publication-safe summary."""

    summary: TableInventory
    rows: tuple[CsvRow, ...]


def read_table(
    archive: zipfile.ZipFile,
    member: str,
    expected_columns: tuple[str, ...],
) -> LoadedTable:
    """Read one named archive table after checking its exact columns."""

    try:
        with archive.open(member) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            columns = tuple(reader.fieldnames or ())
            if columns != expected_columns:
                raise CSEDMInventoryError(
                    f"Unexpected columns in {member}: expected {expected_columns}, found {columns}"
                )
            rows = tuple(_normalise_row(row, columns, member) for row in reader)
    except KeyError as error:
        raise CSEDMInventoryError(f"CSEDM archive is missing required member: {member}") from error
    missing = {
        column: sum(is_unavailable_source_value(row[column]) for row in rows) for column in columns
    }
    return LoadedTable(
        summary=TableInventory(
            member=member,
            columns=columns,
            row_count=len(rows),
            missing_by_column=missing,
        ),
        rows=rows,
    )


def target_key(row: CsvRow) -> TargetKey:
    """Return the stable source identity of one prediction target."""

    return row["SubjectID"], row["ProblemID"], row["StartOrder"]


def is_unavailable_source_value(value: str) -> bool:
    """Return whether the archive marks a value as unavailable."""

    return value.strip().upper() in UNAVAILABLE_SOURCE_VALUES


def _normalise_row(
    row: dict[str | None, str | None], columns: tuple[str, ...], member: str
) -> CsvRow:
    if None in row or any(row.get(column) is None for column in columns):
        raise CSEDMInventoryError(f"Malformed CSV row in {member}")
    return {column: row[column] or "" for column in columns}
