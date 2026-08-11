"""Small atomic helpers for immutable benchmark command artifacts."""

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from socratic_tutor.benchmark.hashing import canonical_json_bytes


class ArtifactConflictError(ValueError):
    """An immutable artifact path already contains different content."""


def artifact_locations(output_root: Path) -> dict[str, Any]:
    """Describe files written beneath an output root for command-line users."""

    working_directory = Path.cwd().resolve()
    resolved_root = output_root.resolve()
    files = sorted(path for path in resolved_root.rglob("*") if path.is_file())
    relative_files: list[str] = []
    for path in files:
        try:
            relative_files.append(path.relative_to(working_directory).as_posix())
        except ValueError:
            relative_files.append(str(path))
    return {
        "working_directory": str(working_directory),
        "output_root": str(resolved_root),
        "files": relative_files,
    }


def immutable_json_bytes(value: BaseModel | dict[str, object]) -> bytes:
    """Return stable, human-readable JSON terminated by one newline."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def write_immutable_json(path: Path, value: BaseModel | dict[str, object]) -> None:
    """Create an immutable JSON artifact or accept an exact retry."""

    write_immutable_bytes(path, immutable_json_bytes(value))


def write_immutable_jsonl(path: Path, records: tuple[BaseModel, ...]) -> None:
    """Create canonical JSONL or accept an exact retry."""

    content = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    write_immutable_bytes(path, content)


def write_immutable_bytes(path: Path, content: bytes) -> None:
    """Publish bytes atomically while rejecting a conflicting existing file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ArtifactConflictError(f"Could not verify existing artifact: {path}") from error
        if existing != content:
            raise ArtifactConflictError(f"Immutable artifact already differs: {path}")
        return

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
