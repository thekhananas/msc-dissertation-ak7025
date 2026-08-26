"""Stable CSV serialisation shared by generated research outputs."""

import csv
from collections.abc import Iterable
from io import StringIO


def csv_bytes(
    headers: tuple[str, ...],
    rows: Iterable[tuple[object, ...]],
) -> bytes:
    """Serialise tabular data as UTF-8 CSV with Unix line endings."""
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")
